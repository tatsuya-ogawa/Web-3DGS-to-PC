#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { basename } from "node:path";
import {
  estimateIsoThreshold,
  extractPointCloudFromVoxels,
  loadGaussianSplats,
  voxelizeGaussianDensity,
  writePointCloudPly,
} from "./index.js";

interface CliArgs {
  input?: string;
  output: string;
  resolution: number;
  voxelSize?: number;
  sigmaRadius: number;
  iso?: number;
  isoPercentile: number;
  maxPoints?: number;
  maxSplats?: number;
  minOpacity: number;
  jitter: number;
  densityScale: number;
  seed: number;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    printHelp();
    process.exitCode = 1;
    return;
  }

  const startedAt = Date.now();
  console.log(`Loading ${args.input}`);
  const splats = await loadGaussianSplats(args.input);
  console.log(`Loaded ${splats.length.toLocaleString()} splats`);

  const grid = voxelizeGaussianDensity(splats, {
    resolution: args.resolution,
    voxelSize: args.voxelSize,
    sigmaRadius: args.sigmaRadius,
    minOpacity: args.minOpacity,
    maxSplats: args.maxSplats,
    densityScale: args.densityScale,
  });
  console.log(
    `Voxelized ${grid.dims.join("x")} grid (${grid.density.length.toLocaleString()} voxels, size ${grid.voxelSize.toPrecision(5)})`,
  );

  const isoThreshold = args.iso ?? estimateIsoThreshold(grid, args.isoPercentile);
  console.log(`Using iso threshold ${isoThreshold.toPrecision(5)}`);

  const pointCloud = extractPointCloudFromVoxels(grid, {
    isoThreshold,
    maxPoints: args.maxPoints,
    jitter: args.jitter,
    seed: args.seed,
  });
  const output = args.output;
  await writeFile(output, writePointCloudPly(pointCloud));
  const points = pointCloud.positions.length / 3;
  console.log(`Wrote ${points.toLocaleString()} points to ${output}`);
  console.log(`Done in ${((Date.now() - startedAt) / 1000).toFixed(2)}s`);
}

function parseArgs(argv: string[]): CliArgs {
  const args: CliArgs = {
    output: "3dgs_voxel_pc.ply",
    resolution: 128,
    sigmaRadius: 3,
    isoPercentile: 0.85,
    minOpacity: 0,
    jitter: 0.35,
    densityScale: 1,
    seed: 1,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      const value = argv[++i];
      if (value === undefined) {
        throw new Error(`Missing value for ${arg}`);
      }
      return value;
    };

    switch (arg) {
      case "-i":
      case "--input":
        args.input = next();
        break;
      case "-o":
      case "--output":
        args.output = next();
        break;
      case "--resolution":
        args.resolution = parseInteger(next(), arg);
        break;
      case "--voxel-size":
        args.voxelSize = parsePositiveNumber(next(), arg);
        break;
      case "--sigma-radius":
        args.sigmaRadius = parsePositiveNumber(next(), arg);
        break;
      case "--iso":
        args.iso = parsePositiveNumber(next(), arg);
        break;
      case "--iso-percentile":
        args.isoPercentile = parsePercentile(next(), arg);
        break;
      case "--max-points":
        args.maxPoints = parseInteger(next(), arg);
        break;
      case "--max-splats":
        args.maxSplats = parseInteger(next(), arg);
        break;
      case "--min-opacity":
        args.minOpacity = parsePercentile(next(), arg);
        break;
      case "--jitter":
        args.jitter = parsePositiveNumber(next(), arg, true);
        break;
      case "--density-scale":
        args.densityScale = parsePositiveNumber(next(), arg);
        break;
      case "--seed":
        args.seed = parseInteger(next(), arg);
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        if (!args.input && !arg.startsWith("-")) {
          args.input = arg;
          if (args.output === "3dgs_voxel_pc.ply") {
            args.output = defaultOutputName(arg);
          }
        } else {
          throw new Error(`Unknown argument ${arg}`);
        }
    }
  }

  return args;
}

function defaultOutputName(input: string): string {
  const name = basename(input).replace(/\.[^.]+$/, "");
  return `${name}_voxel_pc.ply`;
}

function parseInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parsePositiveNumber(value: string, name: string, allowZero = false): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || (allowZero ? parsed < 0 : parsed <= 0)) {
    throw new Error(`${name} must be ${allowZero ? "non-negative" : "positive"}`);
  }
  return parsed;
}

function parsePercentile(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${name} must be between 0 and 1`);
  }
  return parsed;
}

function printHelp(): void {
  console.log(`Usage:
  web-3dgs-to-pc <input.splat|input.ply> [options]
  npm run convert -- <input.splat|input.ply> [options]

Options:
  -o, --output <path>          Output binary PLY path
  --resolution <n>             Longest-axis voxel resolution (default: 128)
  --voxel-size <number>        Explicit voxel size instead of resolution
  --sigma-radius <number>      Gaussian support radius in sigmas (default: 3)
  --iso <number>               Density threshold; overrides --iso-percentile
  --iso-percentile <0..1>      Non-zero density percentile for threshold (default: 0.85)
  --max-points <n>             Limit output points after compaction
  --max-splats <n>             Limit input splats for testing
  --min-opacity <0..1>         Ignore low-opacity splats
  --jitter <number>            Voxel-size-relative point jitter (default: 0.35)
  --density-scale <number>     Density multiplier (default: 1)
  --seed <n>                   Deterministic jitter/shuffle seed
`);
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
