#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { basename, relative, resolve, sep } from "node:path";
import { createServer, type ViteDevServer } from "vite";
import { chromium, type Browser } from "playwright-core";

interface HeadlessCliArgs {
  input?: string;
  output?: string;
  meshOutput?: string;
  resolution: number;
  sigmaRadius: number;
  iso?: number;
  isoPercentile: number;
  maxPoints: number;
  maxSplats?: number;
  minOpacity: number;
  densityScale: number;
  atomicScale: number;
  jitter: number;
  seed: number;
  extSplats: boolean;
  boundsQuantile: number;
  headed: boolean;
  timeoutMs: number;
  smoothIterations: number;
  noColor: boolean;
}

interface HeadlessResult {
  base64Ply: string;
  meshBase64Ply?: string;
  inputSplats: number;
  usedSplats: number;
  points: number;
  meshVertices?: number;
  meshFaces?: number;
  dims: [number, number, number];
  voxelSize: number;
  isoThreshold: number;
  nonZeroVoxels: number;
  selectedVoxels: number;
  adapterInfo: string;
  timings: Record<string, number>;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  if (!args.input) {
    printHelp();
    process.exitCode = 1;
    return;
  }

  const cwd = process.cwd();
  const inputPath = resolve(cwd, args.input);
  const outputPath = resolve(cwd, args.output ?? defaultOutputName(inputPath));
  const meshOutputPath = args.meshOutput ? resolve(cwd, args.meshOutput) : undefined;
  const fileUrl = toServedPath(cwd, inputPath);

  let server: ViteDevServer | undefined;
  let browser: Browser | undefined;
  try {
    server = await createServer({
      root: cwd,
      logLevel: "silent",
      server: {
        host: "127.0.0.1",
        port: 0,
        strictPort: false,
        fs: { allow: [cwd] },
      },
    });
    await server.listen();
    const localUrl = server.resolvedUrls?.local[0];
    if (!localUrl) {
      throw new Error("Failed to resolve Vite dev server URL");
    }

    browser = await chromium.launch({
      channel: "chrome",
      headless: !args.headed,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=WebGPU",
        "--disable-gpu-sandbox",
      ],
    });
    const page = await browser.newPage();
    page.setDefaultTimeout(args.timeoutMs);
    page.on("console", (message) => {
      const text = message.text();
      if (message.type() === "error") {
        console.error(`[browser] ${text}`);
      } else if (text.startsWith("[headless]")) {
        console.log(text);
      }
    });
    page.on("pageerror", (error) => console.error(`[browser] ${error.message}`));
    
    // Load headless.html from its relocated subproject path
    await page.goto(`${localUrl}tests/headless/headless.html`);

    console.log(`Spark/WebGPU headless input: ${args.input}`);
    const result = await page.evaluate(
      async (options) => window.runSogToPointCloud(options),
      {
        fileUrl,
        fileName: basename(inputPath),
        resolution: args.resolution,
        sigmaRadius: args.sigmaRadius,
        iso: args.iso,
        isoPercentile: args.isoPercentile,
        maxPoints: args.maxPoints,
        maxSplats: args.maxSplats,
        minOpacity: args.minOpacity,
        densityScale: args.densityScale,
        atomicScale: args.atomicScale,
        jitter: args.jitter,
        seed: args.seed,
        extSplats: args.extSplats,
        boundsQuantile: args.boundsQuantile,
        mesh: Boolean(meshOutputPath),
        smoothIterations: args.smoothIterations,
        noColor: args.noColor,
      } as any,
    ) as HeadlessResult;

    await writeFile(outputPath, Buffer.from(result.base64Ply, "base64"));
    if (meshOutputPath) {
      if (!result.meshBase64Ply) {
        throw new Error("Mesh output was requested, but the browser did not return a mesh");
      }
      await writeFile(meshOutputPath, Buffer.from(result.meshBase64Ply, "base64"));
    }
    console.log(`Adapter: ${result.adapterInfo || "unknown"}`);
    console.log(`Splats: ${result.usedSplats.toLocaleString()} / ${result.inputSplats.toLocaleString()}`);
    console.log(`Grid: ${result.dims.join("x")} voxelSize=${result.voxelSize.toPrecision(5)}`);
    console.log(
      `Density: iso=${result.isoThreshold.toPrecision(5)} nonzero=${result.nonZeroVoxels.toLocaleString()} selected=${result.selectedVoxels.toLocaleString()}`,
    );
    console.log(`Wrote ${result.points.toLocaleString()} points to ${outputPath}`);
    if (meshOutputPath) {
      console.log(
        `Wrote ${result.meshVertices?.toLocaleString() ?? "0"} mesh vertices / ${result.meshFaces?.toLocaleString() ?? "0"} faces to ${meshOutputPath}`,
      );
    }
    console.log(formatTimings(result.timings));
  } finally {
    await browser?.close().catch(() => undefined);
    await server?.close().catch(() => undefined);
  }
}

function parseArgs(argv: string[]): HeadlessCliArgs {
  const args: HeadlessCliArgs = {
    resolution: 64,
    sigmaRadius: 3,
    isoPercentile: 0.85,
    maxPoints: 50000,
    minOpacity: 0,
    densityScale: 1,
    atomicScale: 512,
    jitter: 0.35,
    seed: 1,
    extSplats: true,
    boundsQuantile: 0.01,
    headed: false,
    timeoutMs: 180000,
    smoothIterations: 0,
    noColor: false,
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
      case "--mesh-output":
        args.meshOutput = next();
        break;
      case "--resolution":
        args.resolution = parseInteger(next(), arg);
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
      case "--density-scale":
        args.densityScale = parsePositiveNumber(next(), arg);
        break;
      case "--atomic-scale":
        args.atomicScale = parsePositiveNumber(next(), arg);
        break;
      case "--jitter":
        args.jitter = parsePositiveNumber(next(), arg);
        break;
      case "--seed":
        args.seed = parseInteger(next(), arg);
        break;
      case "--packed-splats":
        args.extSplats = false;
        break;
      case "--bounds-quantile":
        args.boundsQuantile = parsePercentile(next(), arg);
        break;
      case "--smooth-iterations":
        const parsedSmooth = Number(next());
        if (!Number.isInteger(parsedSmooth) || parsedSmooth < 0) {
          throw new Error(`${arg} must be a non-negative integer`);
        }
        args.smoothIterations = parsedSmooth;
        break;
      case "--no-color":
        args.noColor = true;
        break;
      case "--headed":
        args.headed = true;
        break;
      case "--timeout-ms":
        args.timeoutMs = parseInteger(next(), arg);
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        if (!args.input && !arg.startsWith("-")) {
          args.input = arg;
        } else {
          throw new Error(`Unknown argument ${arg}`);
        }
    }
  }

  return args;
}

function toServedPath(cwd: string, inputPath: string): string {
  const rel = relative(cwd, inputPath);
  if (rel.startsWith("..") || rel.includes(`..${sep}`)) {
    throw new Error(`Input must be inside the workspace: ${inputPath}`);
  }
  return `/${rel.split(sep).map(encodeURIComponent).join("/")}`;
}

function defaultOutputName(inputPath: string): string {
  return `${basename(inputPath).replace(/\.[^.]+$/, "")}_webgpu_pc.ply`;
}

function parseInteger(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function parsePositiveNumber(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be positive`);
  }
  return parsed;
}

// Percentile parser
function parsePercentile(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
    throw new Error(`${name} must be between 0 and 1`);
  }
  return parsed;
}

// Help printer
function printHelp(): void {
  console.log(`Usage:
  npm run headless:run -- splats/model.sog [options]

Options:
  -o, --output <path>        Output binary PLY path
  --mesh-output <path>       Optional binary PLY triangle mesh from the density iso-surface
  --resolution <n>           Longest-axis voxel resolution (default: 64)
  --sigma-radius <number>    Gaussian support radius in sigmas (default: 3)
  --iso <number>             Density threshold; overrides --iso-percentile
  --iso-percentile <0..1>    Non-zero density percentile for threshold (default: 0.85)
  --max-points <n>           Limit output points after compaction (default: 50000)
  --max-splats <n>           Limit decoded splats sent to WebGPU for testing
  --min-opacity <0..1>       Ignore low-opacity splats in WebGPU
  --density-scale <number>   Density multiplier (default: 1)
  --atomic-scale <number>    Fixed-point atomic scale (default: 512)
  --jitter <number>          Voxel-size-relative output jitter (default: 0.35)
  --seed <n>                 Deterministic WebGPU sampling seed (default: 1)
  --packed-splats            Use Spark packed splats instead of extSplats
  --bounds-quantile <0..1>   Trim center bounds by quantile to ignore outliers (default: 0.01)
  --smooth-iterations <n>    3D Laplacian smoothing iterations (default: 0)
  --no-color                 Disable color computation (Fast Mode)
  --headed                   Launch visible Chrome for debugging
  --timeout-ms <n>           Playwright timeout (default: 180000)
`);
}

function formatTimings(timings: Record<string, number>): string {
  const parts = Object.entries(timings).map(([key, value]) => `${key}=${(value / 1000).toFixed(2)}s`);
  return `Timings: ${parts.join(" ")}`;
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
