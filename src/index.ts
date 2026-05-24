export type {
  Bounds,
  Color,
  ExtractPointCloudOptions,
  GaussianSplat,
  PointCloud,
  Quat,
  Vec3,
  VoxelGrid,
  VoxelizeOptions,
} from "./core/types.js";
export {
  computeBounds,
  estimateIsoThreshold,
  extractPointCloudFromVoxels,
  voxelizeGaussianDensity,
} from "./core/voxel.js";
export type { IsoSurfaceGrid, TriangleMesh } from "./core/marchingTetrahedra.js";
export { extractMarchingTetrahedraMesh } from "./core/marchingTetrahedra.js";
export { extractDualContouringMesh } from "./core/dualContouring.js";
export { parseSplat } from "./io/splat.js";
export { parseGaussianPly, writePointCloudPly } from "./io/ply.js";
export { loadGaussianSplats } from "./io/load.js";

// Browser/WebGPU and Splat packing exports
export type {
  PackedSplatsForGpu,
  WebGpuVoxelOptions,
  WebGpuVoxelResult,
} from "./browser/webgpuVoxel.js";
export {
  voxelizeWithWebGpu,
  voxelizeChunkWithDevice,
} from "./browser/webgpuVoxel.js";

export type {
  PackedSplatLoadResult,
} from "./browser/sparkPacking.js";
export {
  decodePackedSplats,
  packSplats,
} from "./browser/sparkPacking.js";

export type {
  BrowserPointCloud,
  BrowserTriangleMesh,
} from "./browser/ply.js";
export {
  writePointCloudPlyBytes,
  writeTriangleMeshPlyBytes,
  bytesToBase64,
} from "./browser/ply.js";

// Dynamic Chunking Terrain physics exports
export type {
  ChunkGenerationMode,
  ChunkVoxelManagerStatus,
  SupportFillMode,
  VoxelGrid as ChunkVoxelGrid,
  HeightField,
  SurfaceHeightQuery,
} from "./core/chunkManager.js";
export {
  ChunkVoxelManager,
  SplatSpatialBinning,
} from "./core/chunkManager.js";
