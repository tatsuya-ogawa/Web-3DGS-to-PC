import { inverseRotateVec3, mulberry32, quatToMatrix, rotateVec3 } from "./math.js";
import type {
  Bounds,
  ExtractPointCloudOptions,
  GaussianSplat,
  PointCloud,
  Vec3,
  VoxelGrid,
  VoxelizeOptions,
} from "./types.js";

export function computeBounds(splats: readonly GaussianSplat[], sigmaRadius = 3, padding = 0): Bounds {
  const min: Vec3 = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max: Vec3 = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

  for (const splat of splats) {
    const rot = quatToMatrix(splat.rotation);
    const axisX = rotateVec3(rot, [splat.scales[0] * sigmaRadius, 0, 0]);
    const axisY = rotateVec3(rot, [0, splat.scales[1] * sigmaRadius, 0]);
    const axisZ = rotateVec3(rot, [0, 0, splat.scales[2] * sigmaRadius]);
    const extent: Vec3 = [
      Math.abs(axisX[0]) + Math.abs(axisY[0]) + Math.abs(axisZ[0]),
      Math.abs(axisX[1]) + Math.abs(axisY[1]) + Math.abs(axisZ[1]),
      Math.abs(axisX[2]) + Math.abs(axisY[2]) + Math.abs(axisZ[2]),
    ];

    for (let axis = 0; axis < 3; axis += 1) {
      min[axis] = Math.min(min[axis], splat.center[axis] - extent[axis] - padding);
      max[axis] = Math.max(max[axis], splat.center[axis] + extent[axis] + padding);
    }
  }

  if (!Number.isFinite(min[0])) {
    return { min: [-1, -1, -1], max: [1, 1, 1] };
  }

  return { min, max };
}

export function voxelizeGaussianDensity(
  splats: readonly GaussianSplat[],
  options: VoxelizeOptions = {},
): VoxelGrid {
  const sigmaRadius = options.sigmaRadius ?? 3;
  const densityScale = options.densityScale ?? 1;
  const minOpacity = options.minOpacity ?? 0;
  const maxSplats = options.maxSplats ?? splats.length;
  const activeSplats = splats.slice(0, maxSplats).filter((splat) => splat.opacity >= minOpacity);
  const roughBounds = computeBounds(activeSplats, sigmaRadius);

  const extent: Vec3 = [
    roughBounds.max[0] - roughBounds.min[0],
    roughBounds.max[1] - roughBounds.min[1],
    roughBounds.max[2] - roughBounds.min[2],
  ];
  const longest = Math.max(extent[0], extent[1], extent[2], Number.EPSILON);
  const voxelSize = options.voxelSize ?? longest / (options.resolution ?? 128);
  const padding = (options.paddingVoxels ?? 2) * voxelSize;
  const bounds = computeBounds(activeSplats, sigmaRadius, padding);

  const dims: Vec3 = [
    Math.max(1, Math.ceil((bounds.max[0] - bounds.min[0]) / voxelSize)),
    Math.max(1, Math.ceil((bounds.max[1] - bounds.min[1]) / voxelSize)),
    Math.max(1, Math.ceil((bounds.max[2] - bounds.min[2]) / voxelSize)),
  ];
  const voxelCount = dims[0] * dims[1] * dims[2];
  const density = new Float32Array(voxelCount);
  const weight = new Float32Array(voxelCount);
  const color = new Float32Array(voxelCount * 3);

  for (const splat of activeSplats) {
    accumulateSplat(splat, { dims, bounds, voxelSize, density, weight, color }, sigmaRadius, densityScale);
  }

  return { dims, bounds, voxelSize, density, weight, color };
}

function accumulateSplat(
  splat: GaussianSplat,
  grid: Pick<VoxelGrid, "dims" | "bounds" | "voxelSize" | "density" | "weight" | "color">,
  sigmaRadius: number,
  densityScale: number,
): void {
  const rot = quatToMatrix(splat.rotation);
  const axes = [
    rotateVec3(rot, [splat.scales[0] * sigmaRadius, 0, 0]),
    rotateVec3(rot, [0, splat.scales[1] * sigmaRadius, 0]),
    rotateVec3(rot, [0, 0, splat.scales[2] * sigmaRadius]),
  ];
  const extent: Vec3 = [
    Math.abs(axes[0][0]) + Math.abs(axes[1][0]) + Math.abs(axes[2][0]),
    Math.abs(axes[0][1]) + Math.abs(axes[1][1]) + Math.abs(axes[2][1]),
    Math.abs(axes[0][2]) + Math.abs(axes[1][2]) + Math.abs(axes[2][2]),
  ];

  const minVoxel: Vec3 = [
    clampIndex(Math.floor((splat.center[0] - extent[0] - grid.bounds.min[0]) / grid.voxelSize), grid.dims[0]),
    clampIndex(Math.floor((splat.center[1] - extent[1] - grid.bounds.min[1]) / grid.voxelSize), grid.dims[1]),
    clampIndex(Math.floor((splat.center[2] - extent[2] - grid.bounds.min[2]) / grid.voxelSize), grid.dims[2]),
  ];
  const maxVoxel: Vec3 = [
    clampIndex(Math.ceil((splat.center[0] + extent[0] - grid.bounds.min[0]) / grid.voxelSize), grid.dims[0]),
    clampIndex(Math.ceil((splat.center[1] + extent[1] - grid.bounds.min[1]) / grid.voxelSize), grid.dims[1]),
    clampIndex(Math.ceil((splat.center[2] + extent[2] - grid.bounds.min[2]) / grid.voxelSize), grid.dims[2]),
  ];

  const sx = Math.max(splat.scales[0], 1e-6);
  const sy = Math.max(splat.scales[1], 1e-6);
  const sz = Math.max(splat.scales[2], 1e-6);
  const sigmaSq = sigmaRadius * sigmaRadius;

  for (let z = minVoxel[2]; z <= maxVoxel[2]; z += 1) {
    const pz = grid.bounds.min[2] + (z + 0.5) * grid.voxelSize;
    for (let y = minVoxel[1]; y <= maxVoxel[1]; y += 1) {
      const py = grid.bounds.min[1] + (y + 0.5) * grid.voxelSize;
      for (let x = minVoxel[0]; x <= maxVoxel[0]; x += 1) {
        const px = grid.bounds.min[0] + (x + 0.5) * grid.voxelSize;
        const local = inverseRotateVec3(rot, [
          px - splat.center[0],
          py - splat.center[1],
          pz - splat.center[2],
        ]);
        const mahalanobisSq =
          (local[0] / sx) ** 2 +
          (local[1] / sy) ** 2 +
          (local[2] / sz) ** 2;
        if (mahalanobisSq > sigmaSq) {
          continue;
        }

        const contribution = splat.opacity * densityScale * Math.exp(-0.5 * mahalanobisSq);
        const index = voxelIndex(x, y, z, grid.dims);
        grid.density[index] += contribution;
        grid.weight[index] += contribution;
        const colorIndex = index * 3;
        grid.color[colorIndex] += contribution * splat.color[0];
        grid.color[colorIndex + 1] += contribution * splat.color[1];
        grid.color[colorIndex + 2] += contribution * splat.color[2];
      }
    }
  }
}

export function extractPointCloudFromVoxels(grid: VoxelGrid, options: ExtractPointCloudOptions): PointCloud {
  const candidates: number[] = [];
  for (let i = 0; i < grid.density.length; i += 1) {
    if (grid.density[i] >= options.isoThreshold) {
      candidates.push(i);
    }
  }

  const rng = mulberry32(options.seed ?? 1);
  const maxPoints = options.maxPoints ?? candidates.length;
  shufflePrefix(candidates, Math.min(maxPoints, candidates.length), rng);
  const pointCount = Math.min(maxPoints, candidates.length);
  const positions = new Float32Array(pointCount * 3);
  const colors = new Uint8Array(pointCount * 3);
  const jitter = options.jitter ?? 0;

  for (let i = 0; i < pointCount; i += 1) {
    const index = candidates[i];
    const [x, y, z] = indexToVoxel(index, grid.dims);
    const jx = (rng() - 0.5) * jitter * grid.voxelSize;
    const jy = (rng() - 0.5) * jitter * grid.voxelSize;
    const jz = (rng() - 0.5) * jitter * grid.voxelSize;
    positions[i * 3] = grid.bounds.min[0] + (x + 0.5) * grid.voxelSize + jx;
    positions[i * 3 + 1] = grid.bounds.min[1] + (y + 0.5) * grid.voxelSize + jy;
    positions[i * 3 + 2] = grid.bounds.min[2] + (z + 0.5) * grid.voxelSize + jz;

    const colorIndex = index * 3;
    const invW = grid.weight[index] > 0 ? 1 / grid.weight[index] : 0;
    colors[i * 3] = Math.max(0, Math.min(255, Math.round(grid.color[colorIndex] * invW)));
    colors[i * 3 + 1] = Math.max(0, Math.min(255, Math.round(grid.color[colorIndex + 1] * invW)));
    colors[i * 3 + 2] = Math.max(0, Math.min(255, Math.round(grid.color[colorIndex + 2] * invW)));
  }

  return { positions, colors };
}

export function estimateIsoThreshold(grid: VoxelGrid, percentile = 0.85): number {
  const nonzero: number[] = [];
  for (const value of grid.density) {
    if (value > 0) {
      nonzero.push(value);
    }
  }
  if (nonzero.length === 0) {
    return Number.POSITIVE_INFINITY;
  }
  nonzero.sort((a, b) => a - b);
  const index = Math.min(nonzero.length - 1, Math.max(0, Math.floor(nonzero.length * percentile)));
  return nonzero[index];
}

function voxelIndex(x: number, y: number, z: number, dims: Vec3): number {
  return x + y * dims[0] + z * dims[0] * dims[1];
}

function indexToVoxel(index: number, dims: Vec3): Vec3 {
  const xy = dims[0] * dims[1];
  const z = Math.floor(index / xy);
  const rest = index - z * xy;
  const y = Math.floor(rest / dims[0]);
  const x = rest - y * dims[0];
  return [x, y, z];
}

function clampIndex(value: number, dim: number): number {
  return Math.max(0, Math.min(dim - 1, value));
}

function shufflePrefix<T>(items: T[], prefixLength: number, rng: () => number): void {
  for (let i = 0; i < prefixLength; i += 1) {
    const j = i + Math.floor(rng() * (items.length - i));
    const tmp = items[i];
    items[i] = items[j];
    items[j] = tmp;
  }
}
