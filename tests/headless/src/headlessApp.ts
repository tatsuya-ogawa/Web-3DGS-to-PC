import { SplatMesh } from "@sparkjsdev/spark";
import {
  bytesToBase64,
  writePointCloudPlyBytes,
  writeTriangleMeshPlyBytes,
  decodePackedSplats,
  voxelizeWithWebGpu,
  type WebGpuVoxelOptions,
} from "web-3dgs-to-pc/browser";

export interface HeadlessRunOptions {
  fileUrl: string;
  fileName: string;
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
  mesh: boolean;
  smoothIterations: number;
  noColor: boolean;
}

export interface HeadlessRunResult {
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

declare global {
  interface Window {
    runSogToPointCloud(options: HeadlessRunOptions): Promise<HeadlessRunResult>;
  }
}

window.runSogToPointCloud = async (options: HeadlessRunOptions): Promise<HeadlessRunResult> => {
  const timings: Record<string, number> = {};
  const mark = performance.now();
  const response = await fetch(options.fileUrl);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${options.fileUrl}: ${response.status} ${response.statusText}`);
  }
  const fileBytes = new Uint8Array(await response.arrayBuffer());
  timings.fetchMs = performance.now() - mark;

  const loadStarted = performance.now();
  const {
    packed,
    inputSplats,
    usedSplats,
  } = await decodePackedSplats(
    SplatMesh,
    fileBytes,
    options.fileName,
    options.extSplats,
    options.maxSplats,
    options.boundsQuantile,
  );
  timings.sparkLoadMs = performance.now() - loadStarted;

  const gpuStarted = performance.now();
  const gpuOptions: WebGpuVoxelOptions = {
    resolution: options.resolution,
    sigmaRadius: options.sigmaRadius,
    iso: options.iso,
    isoPercentile: options.isoPercentile,
    maxPoints: options.maxPoints,
    densityScale: options.densityScale,
    minOpacity: options.minOpacity,
    atomicScale: options.atomicScale,
    jitter: options.jitter,
    seed: options.seed,
    mesh: options.mesh,
    smoothIterations: options.smoothIterations,
    noColor: options.noColor,
  };
  const pointCloud = await voxelizeWithWebGpu(packed, gpuOptions);
  timings.webgpuMs = performance.now() - gpuStarted;

  const encodeStarted = performance.now();
  const base64Ply = bytesToBase64(writePointCloudPlyBytes(pointCloud));
  const meshBase64Ply = pointCloud.mesh ? bytesToBase64(writeTriangleMeshPlyBytes(pointCloud.mesh)) : undefined;
  timings.encodeMs = performance.now() - encodeStarted;

  return {
    base64Ply,
    meshBase64Ply,
    inputSplats,
    usedSplats,
    points: pointCloud.positions.length / 3,
    meshVertices: pointCloud.mesh ? pointCloud.mesh.positions.length / 3 : undefined,
    meshFaces: pointCloud.mesh ? pointCloud.mesh.indices.length / 3 : undefined,
    dims: pointCloud.dims,
    voxelSize: pointCloud.voxelSize,
    isoThreshold: pointCloud.isoThreshold,
    nonZeroVoxels: pointCloud.nonZeroVoxels,
    selectedVoxels: pointCloud.selectedVoxels,
    adapterInfo: pointCloud.adapterInfo,
    timings,
  };
};
