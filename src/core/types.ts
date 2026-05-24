export type Vec3 = [number, number, number];
export type Quat = [number, number, number, number];
export type Color = [number, number, number];

export interface GaussianSplat {
  center: Vec3;
  scales: Vec3;
  rotation: Quat;
  opacity: number;
  color: Color;
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
}

export interface VoxelGrid {
  dims: Vec3;
  bounds: Bounds;
  voxelSize: number;
  density: Float32Array;
  weight: Float32Array;
  color: Float32Array;
}

export interface VoxelizeOptions {
  resolution?: number;
  voxelSize?: number;
  paddingVoxels?: number;
  sigmaRadius?: number;
  densityScale?: number;
  minOpacity?: number;
  maxSplats?: number;
}

export interface PointCloud {
  positions: Float32Array;
  colors: Uint8Array;
}

export interface ExtractPointCloudOptions {
  isoThreshold: number;
  maxPoints?: number;
  jitter?: number;
  seed?: number;
}
