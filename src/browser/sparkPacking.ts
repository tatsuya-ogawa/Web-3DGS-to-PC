import type { PackedSplatsForGpu } from "./webgpuVoxel.js";

export interface SplatDataSource {
  numSplats: number;
  forEachSplat(
    callback: (
      index: number,
      center: { x: number; y: number; z: number },
      scale: { x: number; y: number; z: number },
      quaternion: { x: number; y: number; z: number; w: number },
      opacity: number,
      color: { r: number; g: number; b: number }
    ) => void
  ): void;
}

export interface SplatMeshInstance extends SplatDataSource {
  initialized: Promise<void>;
  dispose(): void;
}

export interface SplatMeshConstructor {
  new (options: { fileBytes: Uint8Array; fileName: string; extSplats: boolean }): SplatMeshInstance;
}

export interface PackedSplatLoadResult {
  packed: PackedSplatsForGpu;
  inputSplats: number;
  usedSplats: number;
}

export async function decodePackedSplats(
  SplatMeshClass: SplatMeshConstructor,
  fileBytes: Uint8Array,
  fileName: string,
  extSplats: boolean,
  maxSplats: number | undefined,
  boundsQuantile: number,
): Promise<PackedSplatLoadResult> {
  const mesh = new SplatMeshClass({
    fileBytes,
    fileName,
    extSplats,
  });
  await mesh.initialized;

  try {
    const inputSplats = mesh.numSplats;
    const usedSplats = Math.min(inputSplats, maxSplats ?? inputSplats);
    const packed = packSplats(mesh, usedSplats, boundsQuantile);
    return { packed, inputSplats, usedSplats };
  } finally {
    mesh.dispose();
  }
}

export function packSplats(mesh: SplatDataSource, maxSplats: number, boundsQuantile: number): PackedSplatsForGpu {
  const centers = new Float32Array(maxSplats * 4);
  const scales = new Float32Array(maxSplats * 4);
  const quaternions = new Float32Array(maxSplats * 4);
  const colors = new Float32Array(maxSplats * 4);
  const boundsMin: [number, number, number] = [
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  ];
  const boundsMax: [number, number, number] = [
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
    Number.NEGATIVE_INFINITY,
  ];

  let count = 0;
  mesh.forEachSplat((
    index: number,
    center: { x: number; y: number; z: number },
    scale: { x: number; y: number; z: number },
    quaternion: { x: number; y: number; z: number; w: number },
    opacity: number,
    color: { r: number; g: number; b: number },
  ) => {
    if (index >= maxSplats) {
      return;
    }
    const base = index * 4;
    centers[base] = center.x;
    centers[base + 1] = center.y;
    centers[base + 2] = center.z;
    centers[base + 3] = 1;
    scales[base] = Math.max(1e-6, scale.x);
    scales[base + 1] = Math.max(1e-6, scale.y);
    scales[base + 2] = Math.max(1e-6, scale.z);
    scales[base + 3] = 0;
    quaternions[base] = quaternion.x;
    quaternions[base + 1] = quaternion.y;
    quaternions[base + 2] = quaternion.z;
    quaternions[base + 3] = quaternion.w;
    colors[base] = color.r * 255;
    colors[base + 1] = color.g * 255;
    colors[base + 2] = color.b * 255;
    colors[base + 3] = opacity;

    boundsMin[0] = Math.min(boundsMin[0], center.x);
    boundsMin[1] = Math.min(boundsMin[1], center.y);
    boundsMin[2] = Math.min(boundsMin[2], center.z);
    boundsMax[0] = Math.max(boundsMax[0], center.x);
    boundsMax[1] = Math.max(boundsMax[1], center.y);
    boundsMax[2] = Math.max(boundsMax[2], center.z);
    count = Math.max(count, index + 1);
  });

  if (count === 0) {
    throw new Error("Spark decoded zero splats");
  }

  if (boundsQuantile > 0) {
    const robust = estimateQuantileBounds(centers, count, boundsQuantile);
    boundsMin[0] = robust.min[0];
    boundsMin[1] = robust.min[1];
    boundsMin[2] = robust.min[2];
    boundsMax[0] = robust.max[0];
    boundsMax[1] = robust.max[1];
    boundsMax[2] = robust.max[2];
  }

  const padding = Math.max(
    boundsMax[0] - boundsMin[0],
    boundsMax[1] - boundsMin[1],
    boundsMax[2] - boundsMin[2],
  ) * 0.02;
  for (let axis = 0; axis < 3; axis += 1) {
    boundsMin[axis] -= padding;
    boundsMax[axis] += padding;
  }

  return {
    centers: centers.subarray(0, count * 4),
    scales: scales.subarray(0, count * 4),
    quaternions: quaternions.subarray(0, count * 4),
    colors: colors.subarray(0, count * 4),
    boundsMin,
    boundsMax,
    count,
  };
}

function estimateQuantileBounds(
  centers: Float32Array,
  count: number,
  quantile: number,
): { min: [number, number, number]; max: [number, number, number] } {
  const sampleCount = Math.min(count, 200_000);
  const stride = Math.max(1, Math.floor(count / sampleCount));
  const values = [
    new Float32Array(sampleCount),
    new Float32Array(sampleCount),
    new Float32Array(sampleCount),
  ];
  let written = 0;
  for (let i = 0; i < count && written < sampleCount; i += stride) {
    const base = i * 4;
    values[0][written] = centers[base];
    values[1][written] = centers[base + 1];
    values[2][written] = centers[base + 2];
    written += 1;
  }

  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  const lowIndex = Math.max(0, Math.min(written - 1, Math.floor(written * quantile)));
  const highIndex = Math.max(0, Math.min(written - 1, Math.floor(written * (1 - quantile))));

  for (let axis = 0; axis < 3; axis += 1) {
    const axisValues = values[axis].subarray(0, written);
    axisValues.sort();
    min[axis] = axisValues[lowIndex];
    max[axis] = axisValues[highIndex];
  }

  return { min, max };
}
