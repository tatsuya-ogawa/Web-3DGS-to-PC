import { extractMarchingTetrahedraMesh, type TriangleMesh } from "../core/marchingTetrahedra.js";
import { extractDualContouringMesh } from "../core/dualContouring.js";

export interface PackedSplatsForGpu {
  centers: Float32Array;
  scales: Float32Array;
  quaternions: Float32Array;
  colors: Float32Array;
  boundsMin: [number, number, number];
  boundsMax: [number, number, number];
  count: number;
}

export interface WebGpuVoxelOptions {
  resolution: number;
  sigmaRadius: number;
  iso?: number;
  isoPercentile: number;
  maxPoints: number;
  densityScale: number;
  minOpacity: number;
  atomicScale: number;
  jitter: number;
  seed: number;
  mesh?: boolean;
  meshType?: "tetrahedra" | "dual";
  returnVoxels?: boolean;
  smoothIterations?: number;
  noColor?: boolean;
}

export interface WebGpuVoxelResult {
  positions: Float32Array;
  colors: Uint8Array;
  isoThreshold: number;
  dims: [number, number, number];
  voxelSize: number;
  nonZeroVoxels: number;
  selectedVoxels: number;
  adapterInfo: string;
  mesh?: TriangleMesh;
  voxelCenters?: Float32Array;
  voxelColors?: Uint8Array;
}

const VOXEL_SHADER = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

@group(0) @binding(0) var<storage, read> centers: array<vec4f>;
@group(0) @binding(1) var<storage, read> scales: array<vec4f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;
@group(0) @binding(3) var<storage, read> colors: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> colorAccum: array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params: Params;

fn clampI32(value: i32, low: i32, high: i32) -> i32 {
  return min(max(value, low), high);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.numSplats) {
    return;
  }

  let center = centers[i].xyz;
  let scale = max(scales[i].xyz, vec3f(0.000001));
  let quatRaw = quats[i];
  let qLen = max(length(quatRaw), 0.000001);
  let q = quatRaw / qLen;
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;
  let color = colors[i];
  let opacity = color.a;
  if (opacity < params.controls.z) {
    return;
  }

  let m00 = 1.0 - 2.0 * (y * y + z * z);
  let m01 = 2.0 * (x * y - w * z);
  let m02 = 2.0 * (x * z + w * y);
  let m10 = 2.0 * (x * y + w * z);
  let m11 = 1.0 - 2.0 * (x * x + z * z);
  let m12 = 2.0 * (y * z - w * x);
  let m20 = 2.0 * (x * z - w * y);
  let m21 = 2.0 * (y * z + w * x);
  let m22 = 1.0 - 2.0 * (x * x + y * y);

  let sigmaRadius = params.controls.x;
  let sx = scale.x * sigmaRadius;
  let sy = scale.y * sigmaRadius;
  let sz = scale.z * sigmaRadius;
  let extent = vec3f(
    abs(m00 * sx) + abs(m01 * sy) + abs(m02 * sz),
    abs(m10 * sx) + abs(m11 * sy) + abs(m12 * sz),
    abs(m20 * sx) + abs(m21 * sy) + abs(m22 * sz),
  );

  let boundsMin = params.boundsMinAndVoxel.xyz;
  let boundsMax = params.boundsMaxAndPad.xyz;
  if (
    center.x < boundsMin.x || center.y < boundsMin.y || center.z < boundsMin.z ||
    center.x > boundsMax.x || center.y > boundsMax.y || center.z > boundsMax.z
  ) {
    return;
  }
  let voxelSize = params.boundsMinAndVoxel.w;
  let minVoxel = vec3i(
    clampI32(i32(floor((center.x - extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
    clampI32(i32(floor((center.y - extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
    clampI32(i32(floor((center.z - extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
  );
  let maxVoxel = vec3i(
    clampI32(i32(ceil((center.x + extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
    clampI32(i32(ceil((center.y + extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
    clampI32(i32(ceil((center.z + extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
  );

  let sigmaSq = sigmaRadius * sigmaRadius;
  let atomicScale = params.controls.w;
  let densityScale = params.controls.y;
  for (var vz = minVoxel.z; vz <= maxVoxel.z; vz = vz + 1) {
    let pz = boundsMin.z + (f32(vz) + 0.5) * voxelSize;
    for (var vy = minVoxel.y; vy <= maxVoxel.y; vy = vy + 1) {
      let py = boundsMin.y + (f32(vy) + 0.5) * voxelSize;
      for (var vx = minVoxel.x; vx <= maxVoxel.x; vx = vx + 1) {
        let px = boundsMin.x + (f32(vx) + 0.5) * voxelSize;
        let d = vec3f(px, py, pz) - center;
        let local = vec3f(
          m00 * d.x + m10 * d.y + m20 * d.z,
          m01 * d.x + m11 * d.y + m21 * d.z,
          m02 * d.x + m12 * d.y + m22 * d.z,
        );
        let mahalanobisSq =
          (local.x / scale.x) * (local.x / scale.x) +
          (local.y / scale.y) * (local.y / scale.y) +
          (local.z / scale.z) * (local.z / scale.z);
        if (mahalanobisSq > sigmaSq) {
          continue;
        }

        let contribution = opacity * densityScale * exp(-0.5 * mahalanobisSq);
        let scaled = u32(max(0.0, contribution * atomicScale + 0.5));
        if (scaled == 0u) {
          continue;
        }
        let index = u32(vx) + u32(vy) * params.dimX + u32(vz) * params.dimX * params.dimY;
        atomicAdd(&density[index], scaled);
        let colorBase = index * 3u;
        atomicAdd(&colorAccum[colorBase], u32(max(0.0, contribution * color.r * atomicScale + 0.5)));
        atomicAdd(&colorAccum[colorBase + 1u], u32(max(0.0, contribution * color.g * atomicScale + 0.5)));
        atomicAdd(&colorAccum[colorBase + 2u], u32(max(0.0, contribution * color.b * atomicScale + 0.5)));
      }
    }
  }
}
`;

const EXTRACT_SHADER = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

struct ExtractParams {
  maxSelected: u32,
  threshold: u32,
  _pad0: u32,
  _pad1: u32,
};

struct Counter {
  value: atomic<u32>,
};

@group(0) @binding(0) var<storage, read> density: array<u32>;
@group(0) @binding(1) var<storage, read_write> counter: Counter;
@group(0) @binding(2) var<storage, read_write> selectedIndices: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<uniform> extractParams: ExtractParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let voxelCount = params.dimX * params.dimY * params.dimZ;
  if (index >= voxelCount) {
    return;
  }
  let d = density[index];
  if (d < extractParams.threshold || d == 0u) {
    return;
  }
  let out = atomicAdd(&counter.value, 1u);
  if (out >= extractParams.maxSelected) {
    return;
  }
  selectedIndices[out] = index;
}
`;

const SAMPLE_SHADER = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

struct SampleParams {
  pointCount: u32,
  selectedCount: u32,
  seed: u32,
  jitterBits: u32,
};

@group(0) @binding(0) var<storage, read> density: array<u32>;
@group(0) @binding(1) var<storage, read> colorAccum: array<u32>;
@group(0) @binding(2) var<storage, read> selectedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> outColors: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<uniform> sampleParams: SampleParams;

fn hash32(v: u32) -> u32 {
  var x = v;
  x = ((x >> 16u) ^ x) * 0x7feb352du;
  x = ((x >> 15u) ^ x) * 0x846ca68bu;
  x = (x >> 16u) ^ x;
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash32(seed) & 0x00ffffffu) / 16777216.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pointIndex = gid.x;
  if (pointIndex >= sampleParams.pointCount || sampleParams.selectedCount == 0u) {
    return;
  }
  let selectedSlot = hash32(pointIndex + sampleParams.seed * 1664525u) % sampleParams.selectedCount;
  let index = selectedIndices[selectedSlot];
  let xy = params.dimX * params.dimY;
  let z = index / xy;
  let rest = index - z * xy;
  let y = rest / params.dimX;
  let x = rest - y * params.dimX;
  let boundsMin = params.boundsMinAndVoxel.xyz;
  let voxelSize = params.boundsMinAndVoxel.w;
  let jitter = bitcast<f32>(sampleParams.jitterBits);
  let j = vec3f(
    (rand01(pointIndex * 3u + sampleParams.seed + 11u) - 0.5) * jitter,
    (rand01(pointIndex * 3u + sampleParams.seed + 17u) - 0.5) * jitter,
    (rand01(pointIndex * 3u + sampleParams.seed + 23u) - 0.5) * jitter,
  );
  positions[pointIndex] = vec4f(
    boundsMin.x + (f32(x) + 0.5 + j.x) * voxelSize,
    boundsMin.y + (f32(y) + 0.5 + j.y) * voxelSize,
    boundsMin.z + (f32(z) + 0.5 + j.z) * voxelSize,
    1.0,
  );

  let d = density[index];
  let colorBase = index * 3u;
  let r = min(255u, (colorAccum[colorBase] + d / 2u) / d);
  let g = min(255u, (colorAccum[colorBase + 1u] + d / 2u) / d);
  let b = min(255u, (colorAccum[colorBase + 2u] + d / 2u) / d);
  outColors[pointIndex] = r | (g << 8u) | (b << 16u);
}
`;

const SMOOTH_SHADER = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

@group(0) @binding(0) var<storage, read> densityIn: array<u32>;
@group(0) @binding(1) var<storage, read> colorAccumIn: array<u32>;
@group(0) @binding(2) var<storage, read_write> densityOut: array<u32>;
@group(0) @binding(3) var<storage, read_write> colorAccumOut: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

fn getIndex(x: u32, y: u32, z: u32) -> u32 {
  return x + y * params.dimX + z * params.dimX * params.dimY;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let voxelCount = params.dimX * params.dimY * params.dimZ;
  if (index >= voxelCount) {
    return;
  }

  let xy = params.dimX * params.dimY;
  let z = index / xy;
  let rest = index - z * xy;
  let y = rest / params.dimX;
  let x = rest - y * params.dimX;

  let val_c = f32(densityIn[index]);
  
  var count = 1.0;
  var sumDensity = val_c;
  var sumR = f32(colorAccumIn[index * 3u]);
  var sumG = f32(colorAccumIn[index * 3u + 1u]);
  var sumB = f32(colorAccumIn[index * 3u + 2u]);

  let dimX_i = i32(params.dimX);
  let dimY_i = i32(params.dimY);
  let dimZ_i = i32(params.dimZ);
  let x_i = i32(x);
  let y_i = i32(y);
  let z_i = i32(z);

  let dx = array<i32, 6>(-1, 1, 0, 0, 0, 0);
  let dy = array<i32, 6>(0, 0, -1, 1, 0, 0);
  let dz = array<i32, 6>(0, 0, 0, 0, -1, 1);

  for (var d = 0u; d < 6u; d = d + 1u) {
    let nx = x_i + dx[d];
    let ny = y_i + dy[d];
    let nz = z_i + dz[d];
    if (nx >= 0 && nx < dimX_i && ny >= 0 && ny < dimY_i && nz >= 0 && nz < dimZ_i) {
      let nIdx = getIndex(u32(nx), u32(ny), u32(nz));
      sumDensity = sumDensity + f32(densityIn[nIdx]);
      sumR = sumR + f32(colorAccumIn[nIdx * 3u]);
      sumG = sumG + f32(colorAccumIn[nIdx * 3u + 1u]);
      sumB = sumB + f32(colorAccumIn[nIdx * 3u + 2u]);
      count = count + 1.0;
    }
  }

  let lambda = 0.15;
  let avgDensity = sumDensity / count;
  let avgR = sumR / count;
  let avgG = sumG / count;
  let avgB = sumB / count;

  let smoothedDensity = val_c * (1.0 - lambda) + avgDensity * lambda;
  let smoothedR = f32(colorAccumIn[index * 3u]) * (1.0 - lambda) + avgR * lambda;
  let smoothedG = f32(colorAccumIn[index * 3u + 1u]) * (1.0 - lambda) + avgG * lambda;
  let smoothedB = f32(colorAccumIn[index * 3u + 2u]) * (1.0 - lambda) + avgB * lambda;

  densityOut[index] = u32(smoothedDensity + 0.5);
  colorAccumOut[index * 3u] = u32(smoothedR + 0.5);
  colorAccumOut[index * 3u + 1u] = u32(smoothedG + 0.5);
  colorAccumOut[index * 3u + 2u] = u32(smoothedB + 0.5);
}
`;

const VOXEL_SHADER_NO_COLOR = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

@group(0) @binding(0) var<storage, read> centers: array<vec4f>;
@group(0) @binding(1) var<storage, read> scales: array<vec4f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;
@group(0) @binding(3) var<storage, read> colors: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

fn clampI32(value: i32, low: i32, high: i32) -> i32 {
  return min(max(value, low), high);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.numSplats) {
    return;
  }

  let center = centers[i].xyz;
  let scale = max(scales[i].xyz, vec3f(0.000001));
  let quatRaw = quats[i];
  let qLen = max(length(quatRaw), 0.000001);
  let q = quatRaw / qLen;
  let x = q.x;
  let y = q.y;
  let z = q.z;
  let w = q.w;
  let color = colors[i];
  let opacity = color.a;
  if (opacity < params.controls.z) {
    return;
  }

  let m00 = 1.0 - 2.0 * (y * y + z * z);
  let m01 = 2.0 * (x * y - w * z);
  let m02 = 2.0 * (x * z + w * y);
  let m10 = 2.0 * (x * y + w * z);
  let m11 = 1.0 - 2.0 * (x * x + z * z);
  let m12 = 2.0 * (y * z - w * x);
  let m20 = 2.0 * (x * z - w * y);
  let m21 = 2.0 * (y * z + w * x);
  let m22 = 1.0 - 2.0 * (x * x + y * y);

  let sigmaRadius = params.controls.x;
  let sx = scale.x * sigmaRadius;
  let sy = scale.y * sigmaRadius;
  let sz = scale.z * sigmaRadius;
  let extent = vec3f(
    abs(m00 * sx) + abs(m01 * sy) + abs(m02 * sz),
    abs(m10 * sx) + abs(m11 * sy) + abs(m12 * sz),
    abs(m20 * sx) + abs(m21 * sy) + abs(m22 * sz),
  );

  let boundsMin = params.boundsMinAndVoxel.xyz;
  let boundsMax = params.boundsMaxAndPad.xyz;
  if (
    center.x < boundsMin.x || center.y < boundsMin.y || center.z < boundsMin.z ||
    center.x > boundsMax.x || center.y > boundsMax.y || center.z > boundsMax.z
  ) {
    return;
  }
  let voxelSize = params.boundsMinAndVoxel.w;
  let minVoxel = vec3i(
    clampI32(i32(floor((center.x - extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
    clampI32(i32(floor((center.y - extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
    clampI32(i32(floor((center.z - extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
  );
  let maxVoxel = vec3i(
    clampI32(i32(ceil((center.x + extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
    clampI32(i32(ceil((center.y + extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
    clampI32(i32(ceil((center.z + extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
  );

  let sigmaSq = sigmaRadius * sigmaRadius;
  let atomicScale = params.controls.w;
  let densityScale = params.controls.y;
  for (var vz = minVoxel.z; vz <= maxVoxel.z; vz = vz + 1) {
    let pz = boundsMin.z + (f32(vz) + 0.5) * voxelSize;
    for (var vy = minVoxel.y; vy <= maxVoxel.y; vy = vy + 1) {
      let py = boundsMin.y + (f32(vy) + 0.5) * voxelSize;
      for (var vx = minVoxel.x; vx <= maxVoxel.x; vx = vx + 1) {
        let px = boundsMin.x + (f32(vx) + 0.5) * voxelSize;
        let d = vec3f(px, py, pz) - center;
        let local = vec3f(
          m00 * d.x + m10 * d.y + m20 * d.z,
          m01 * d.x + m11 * d.y + m21 * d.z,
          m02 * d.x + m12 * d.y + m22 * d.z,
        );
        let mahalanobisSq =
          (local.x / scale.x) * (local.x / scale.x) +
          (local.y / scale.y) * (local.y / scale.y) +
          (local.z / scale.z) * (local.z / scale.z);
        if (mahalanobisSq > sigmaSq) {
          continue;
        }

        let contribution = opacity * densityScale * exp(-0.5 * mahalanobisSq);
        let scaled = u32(max(0.0, contribution * atomicScale + 0.5));
        if (scaled == 0u) {
          continue;
        }
        let index = u32(vx) + u32(vy) * params.dimX + u32(vz) * params.dimX * params.dimY;
        atomicAdd(&density[index], scaled);
      }
    }
  }
}
`;

const CHUNK_VOXEL_SHADER_WITH_MIN_HEIGHT = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

struct MinHeight {
  yVoxel: atomic<u32>,
};

@group(0) @binding(0) var<storage, read> centers: array<vec4f>;
@group(0) @binding(1) var<storage, read> scales: array<vec4f>;
@group(0) @binding(2) var<storage, read> quats: array<vec4f>;
@group(0) @binding(3) var<storage, read> colors: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> density: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> minHeight: MinHeight;

var<workgroup> groupMinY: array<u32, 64>;

fn clampI32(value: i32, low: i32, high: i32) -> i32 {
  return min(max(value, low), high);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let sentinel = 0xffffffffu;
  var localMinY = sentinel;
  let i = gid.x;

  if (i < params.numSplats) {
    let center = centers[i].xyz;
    let scale = max(scales[i].xyz, vec3f(0.000001));
    let quatRaw = quats[i];
    let qLen = max(length(quatRaw), 0.000001);
    let q = quatRaw / qLen;
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    let color = colors[i];
    let opacity = color.a;

    if (opacity >= params.controls.z) {
      let m00 = 1.0 - 2.0 * (y * y + z * z);
      let m01 = 2.0 * (x * y - w * z);
      let m02 = 2.0 * (x * z + w * y);
      let m10 = 2.0 * (x * y + w * z);
      let m11 = 1.0 - 2.0 * (x * x + z * z);
      let m12 = 2.0 * (y * z - w * x);
      let m20 = 2.0 * (x * z - w * y);
      let m21 = 2.0 * (y * z + w * x);
      let m22 = 1.0 - 2.0 * (x * x + y * y);

      let sigmaRadius = params.controls.x;
      let sx = scale.x * sigmaRadius;
      let sy = scale.y * sigmaRadius;
      let sz = scale.z * sigmaRadius;
      let extent = vec3f(
        abs(m00 * sx) + abs(m01 * sy) + abs(m02 * sz),
        abs(m10 * sx) + abs(m11 * sy) + abs(m12 * sz),
        abs(m20 * sx) + abs(m21 * sy) + abs(m22 * sz),
      );

      let boundsMin = params.boundsMinAndVoxel.xyz;
      let voxelSize = params.boundsMinAndVoxel.w;
      let minVoxel = vec3i(
        clampI32(i32(floor((center.x - extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
        clampI32(i32(floor((center.y - extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
        clampI32(i32(floor((center.z - extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
      );
      let maxVoxel = vec3i(
        clampI32(i32(ceil((center.x + extent.x - boundsMin.x) / voxelSize)), 0, i32(params.dimX) - 1),
        clampI32(i32(ceil((center.y + extent.y - boundsMin.y) / voxelSize)), 0, i32(params.dimY) - 1),
        clampI32(i32(ceil((center.z + extent.z - boundsMin.z) / voxelSize)), 0, i32(params.dimZ) - 1),
      );

      let sigmaSq = sigmaRadius * sigmaRadius;
      let atomicScale = params.controls.w;
      let densityScale = params.controls.y;
      let minHeightThreshold = max(1u, u32(params.boundsMaxAndPad.w + 0.5));

      for (var vz = minVoxel.z; vz <= maxVoxel.z; vz = vz + 1) {
        let pz = boundsMin.z + (f32(vz) + 0.5) * voxelSize;
        for (var vy = minVoxel.y; vy <= maxVoxel.y; vy = vy + 1) {
          let py = boundsMin.y + (f32(vy) + 0.5) * voxelSize;
          for (var vx = minVoxel.x; vx <= maxVoxel.x; vx = vx + 1) {
            let px = boundsMin.x + (f32(vx) + 0.5) * voxelSize;
            let d = vec3f(px, py, pz) - center;
            let local = vec3f(
              m00 * d.x + m10 * d.y + m20 * d.z,
              m01 * d.x + m11 * d.y + m21 * d.z,
              m02 * d.x + m12 * d.y + m22 * d.z,
            );
            let mahalanobisSq =
              (local.x / scale.x) * (local.x / scale.x) +
              (local.y / scale.y) * (local.y / scale.y) +
              (local.z / scale.z) * (local.z / scale.z);
            if (mahalanobisSq > sigmaSq) {
              continue;
            }

            let contribution = opacity * densityScale * exp(-0.5 * mahalanobisSq);
            let scaled = u32(max(0.0, contribution * atomicScale + 0.5));
            if (scaled == 0u) {
              continue;
            }
            let index = u32(vx) + u32(vy) * params.dimX + u32(vz) * params.dimX * params.dimY;
            atomicAdd(&density[index], scaled);
            if (scaled >= minHeightThreshold) {
              localMinY = min(localMinY, u32(vy));
            }
          }
        }
      }
    }
  }

  groupMinY[lid.x] = localMinY;
  workgroupBarrier();

  for (var stride = 32u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) {
      groupMinY[lid.x] = min(groupMinY[lid.x], groupMinY[lid.x + stride]);
    }
    workgroupBarrier();
  }

  if (lid.x == 0u && groupMinY[0] != sentinel) {
    atomicMin(&minHeight.yVoxel, groupMinY[0]);
  }
}
`;

const SMOOTH_SHADER_NO_COLOR = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

@group(0) @binding(0) var<storage, read> densityIn: array<u32>;
@group(0) @binding(1) var<storage, read_write> densityOut: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn getIndex(x: u32, y: u32, z: u32) -> u32 {
  return x + y * params.dimX + z * params.dimX * params.dimY;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  let voxelCount = params.dimX * params.dimY * params.dimZ;
  if (index >= voxelCount) {
    return;
  }

  let xy = params.dimX * params.dimY;
  let z = index / xy;
  let rest = index - z * xy;
  let y = rest / params.dimX;
  let x = rest - y * params.dimX;

  let val_c = f32(densityIn[index]);
  
  var count = 1.0;
  var sumDensity = val_c;

  let dimX_i = i32(params.dimX);
  let dimY_i = i32(params.dimY);
  let dimZ_i = i32(params.dimZ);
  let x_i = i32(x);
  let y_i = i32(y);
  let z_i = i32(z);

  let dx = array<i32, 6>(-1, 1, 0, 0, 0, 0);
  let dy = array<i32, 6>(0, 0, -1, 1, 0, 0);
  let dz = array<i32, 6>(0, 0, 0, 0, -1, 1);

  for (var d = 0u; d < 6u; d = d + 1u) {
    let nx = x_i + dx[d];
    let ny = y_i + dy[d];
    let nz = z_i + dz[d];
    if (nx >= 0 && nx < dimX_i && ny >= 0 && ny < dimY_i && nz >= 0 && nz < dimZ_i) {
      let nIdx = getIndex(u32(nx), u32(ny), u32(nz));
      sumDensity = sumDensity + f32(densityIn[nIdx]);
      count = count + 1.0;
    }
  }

  let lambda = 0.15;
  let avgDensity = sumDensity / count;
  let smoothedDensity = val_c * (1.0 - lambda) + avgDensity * lambda;

  densityOut[index] = u32(smoothedDensity + 0.5);
}
`;

const SAMPLE_SHADER_NO_COLOR = /* wgsl */ `
struct Params {
  numSplats: u32,
  dimX: u32,
  dimY: u32,
  dimZ: u32,
  boundsMinAndVoxel: vec4f,
  boundsMaxAndPad: vec4f,
  controls: vec4f,
};

struct SampleParams {
  pointCount: u32,
  selectedCount: u32,
  seed: u32,
  jitterBits: u32,
};

@group(0) @binding(0) var<storage, read> density: array<u32>;
@group(0) @binding(1) var<storage, read> selectedIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> positions: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> outColors: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<uniform> sampleParams: SampleParams;

fn hash32(v: u32) -> u32 {
  var x = v;
  x = ((x >> 16u) ^ x) * 0x7feb352du;
  x = ((x >> 15u) ^ x) * 0x846ca68bu;
  x = (x >> 16u) ^ x;
  return x;
}

fn rand01(seed: u32) -> f32 {
  return f32(hash32(seed) & 0x00ffffffu) / 16777216.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let pointIndex = gid.x;
  if (pointIndex >= sampleParams.pointCount || sampleParams.selectedCount == 0u) {
    return;
  }
  let selectedSlot = hash32(pointIndex + sampleParams.seed * 1664525u) % sampleParams.selectedCount;
  let index = selectedIndices[selectedSlot];
  let xy = params.dimX * params.dimY;
  let z = index / xy;
  let rest = index - z * xy;
  let y = rest / params.dimX;
  let x = rest - y * params.dimX;
  let boundsMin = params.boundsMinAndVoxel.xyz;
  let voxelSize = params.boundsMinAndVoxel.w;
  let jitter = bitcast<f32>(sampleParams.jitterBits);
  let j = vec3f(
    (rand01(pointIndex * 3u + sampleParams.seed + 11u) - 0.5) * jitter,
    (rand01(pointIndex * 3u + sampleParams.seed + 17u) - 0.5) * jitter,
    (rand01(pointIndex * 3u + sampleParams.seed + 23u) - 0.5) * jitter,
  );
  positions[pointIndex] = vec4f(
    boundsMin.x + (f32(x) + 0.5 + j.x) * voxelSize,
    boundsMin.y + (f32(y) + 0.5 + j.y) * voxelSize,
    boundsMin.z + (f32(z) + 0.5 + j.z) * voxelSize,
    1.0,
  );

  // Write a constant light grey color (0xc8c8c8)
  outColors[pointIndex] = 0xc8c8c8u;
}
`;

export async function voxelizeWithWebGpu(
  splats: PackedSplatsForGpu,
  options: WebGpuVoxelOptions,
): Promise<WebGpuVoxelResult> {
  const gpu = navigator.gpu;
  if (!gpu) {
    throw new Error("navigator.gpu is unavailable in this browser");
  }

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("WebGPU adapter is unavailable");
  }
  const device = await adapter.requestDevice();
  const adapterInfo = await describeAdapter(adapter);

  const extent = [
    splats.boundsMax[0] - splats.boundsMin[0],
    splats.boundsMax[1] - splats.boundsMin[1],
    splats.boundsMax[2] - splats.boundsMin[2],
  ];
  const longest = Math.max(extent[0], extent[1], extent[2], Number.EPSILON);
  const voxelSize = longest / options.resolution;
  const dims: [number, number, number] = [
    Math.max(1, Math.ceil(extent[0] / voxelSize)),
    Math.max(1, Math.ceil(extent[1] / voxelSize)),
    Math.max(1, Math.ceil(extent[2] / voxelSize)),
  ];
  const voxelCount = dims[0] * dims[1] * dims[2];
  const maxStorageBufferBindingSize = device.limits.maxStorageBufferBindingSize;
  if (voxelCount * 3 * 4 > maxStorageBufferBindingSize) {
    throw new Error(
      `Grid color buffer (${voxelCount * 3 * 4} bytes) exceeds maxStorageBufferBindingSize ${maxStorageBufferBindingSize}. Reduce --resolution.`,
    );
  }

  const centersBuffer = createStorageBuffer(device, splats.centers, GPUBufferUsage.STORAGE);
  const scalesBuffer = createStorageBuffer(device, splats.scales, GPUBufferUsage.STORAGE);
  const quatsBuffer = createStorageBuffer(device, splats.quaternions, GPUBufferUsage.STORAGE);
  const colorsBuffer = createStorageBuffer(device, splats.colors, GPUBufferUsage.STORAGE);
  const densityBuffer = device.createBuffer({
    size: align4(voxelCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const noColor = options.noColor ?? false;
  const colorAccumBuffer = noColor
    ? undefined
    : device.createBuffer({
        size: align4(voxelCount * 3 * 4),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });

  const paramsBuffer = createParamsBuffer(device, splats.count, dims, splats.boundsMin, splats.boundsMax, voxelSize, options);

  const voxelPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: noColor ? VOXEL_SHADER_NO_COLOR : VOXEL_SHADER }),
      entryPoint: "main",
    },
  });

  const voxelBindGroup = device.createBindGroup({
    layout: voxelPipeline.getBindGroupLayout(0),
    entries: noColor
      ? [
          { binding: 0, resource: { buffer: centersBuffer } },
          { binding: 1, resource: { buffer: scalesBuffer } },
          { binding: 2, resource: { buffer: quatsBuffer } },
          { binding: 3, resource: { buffer: colorsBuffer } },
          { binding: 4, resource: { buffer: densityBuffer } },
          { binding: 5, resource: { buffer: paramsBuffer } },
        ]
      : [
          { binding: 0, resource: { buffer: centersBuffer } },
          { binding: 1, resource: { buffer: scalesBuffer } },
          { binding: 2, resource: { buffer: quatsBuffer } },
          { binding: 3, resource: { buffer: colorsBuffer } },
          { binding: 4, resource: { buffer: densityBuffer } },
          { binding: 5, resource: { buffer: colorAccumBuffer! } },
          { binding: 6, resource: { buffer: paramsBuffer } },
        ],
  });
  runComputePass(device, voxelPipeline, voxelBindGroup, Math.ceil(splats.count / 64));

  // 3D Laplacian grid smoothing pass (optional)
  const smoothIterations = options.smoothIterations ?? 0;
  if (smoothIterations > 0) {
    const tempDensityBuffer = device.createBuffer({
      size: align4(voxelCount * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const tempColorAccumBuffer = noColor
      ? undefined
      : device.createBuffer({
          size: align4(voxelCount * 3 * 4),
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

    const smoothPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: noColor ? SMOOTH_SHADER_NO_COLOR : SMOOTH_SHADER }),
        entryPoint: "main",
      },
    });

    const bindGroupPing = device.createBindGroup({
      layout: smoothPipeline.getBindGroupLayout(0),
      entries: noColor
        ? [
            { binding: 0, resource: { buffer: densityBuffer } },
            { binding: 1, resource: { buffer: tempDensityBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ]
        : [
            { binding: 0, resource: { buffer: densityBuffer } },
            { binding: 1, resource: { buffer: colorAccumBuffer! } },
            { binding: 2, resource: { buffer: tempDensityBuffer } },
            { binding: 3, resource: { buffer: tempColorAccumBuffer! } },
            { binding: 4, resource: { buffer: paramsBuffer } },
          ],
    });

    const bindGroupPong = device.createBindGroup({
      layout: smoothPipeline.getBindGroupLayout(0),
      entries: noColor
        ? [
            { binding: 0, resource: { buffer: tempDensityBuffer } },
            { binding: 1, resource: { buffer: densityBuffer } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ]
        : [
            { binding: 0, resource: { buffer: tempDensityBuffer } },
            { binding: 1, resource: { buffer: tempColorAccumBuffer! } },
            { binding: 2, resource: { buffer: densityBuffer } },
            { binding: 3, resource: { buffer: colorAccumBuffer! } },
            { binding: 4, resource: { buffer: paramsBuffer } },
          ],
    });

    const workgroupCount = Math.ceil(voxelCount / 64);
    for (let iter = 0; iter < smoothIterations; iter += 1) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(smoothPipeline);
      if (iter % 2 === 0) {
        pass.setBindGroup(0, bindGroupPing);
      } else {
        pass.setBindGroup(0, bindGroupPong);
      }
      pass.dispatchWorkgroups(workgroupCount);
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();

    if (smoothIterations % 2 !== 0) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(tempDensityBuffer, 0, densityBuffer, 0, align4(voxelCount * 4));
      if (!noColor) {
        encoder.copyBufferToBuffer(tempColorAccumBuffer!, 0, colorAccumBuffer!, 0, align4(voxelCount * 3 * 4));
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }

    tempDensityBuffer.destroy();
    if (tempColorAccumBuffer) {
      tempColorAccumBuffer.destroy();
    }
  } else {
    await device.queue.onSubmittedWorkDone();
  }

  const densityRead = new Uint32Array(await readBuffer(device, densityBuffer, voxelCount * 4));
  const thresholdAtomic = options.iso == null
    ? estimateAtomicThreshold(densityRead, options.isoPercentile)
    : Math.max(1, Math.round(options.iso * options.atomicScale));
  const nonZeroVoxels = countAbove(densityRead, 1);
  const selectedVoxels = countAbove(densityRead, thresholdAtomic);

  const needColors = !noColor && ((options.mesh && selectedVoxels > 0) || (options.returnVoxels && selectedVoxels > 0));
  const colorAccumRead = needColors
    ? new Uint32Array(await readBuffer(device, colorAccumBuffer!, voxelCount * 3 * 4))
    : undefined;

  let mesh = undefined;
  if (options.mesh && selectedVoxels > 0) {
    const meshGrid = {
      dims,
      boundsMin: splats.boundsMin,
      voxelSize,
      density: densityRead,
      colorAccum: colorAccumRead,
      isoThreshold: thresholdAtomic,
    };
    mesh = options.meshType === "dual"
      ? extractDualContouringMesh(meshGrid)
      : extractMarchingTetrahedraMesh(meshGrid);
  }

  let voxelCenters: Float32Array | undefined;
  let voxelColors: Uint8Array | undefined;

  if (options.returnVoxels && selectedVoxels > 0) {
    voxelCenters = new Float32Array(selectedVoxels * 3);
    voxelColors = new Uint8Array(selectedVoxels * 3);
    let count = 0;
    const xy = dims[0] * dims[1];
    for (let i = 0; i < voxelCount; i += 1) {
      const d = densityRead[i];
      if (d >= thresholdAtomic) {
        const vz = Math.floor(i / xy);
        const rest = i - vz * xy;
        const vy = Math.floor(rest / dims[0]);
        const vx = rest - vy * dims[0];

        voxelCenters[count * 3] = splats.boundsMin[0] + (vx + 0.5) * voxelSize;
        voxelCenters[count * 3 + 1] = splats.boundsMin[1] + (vy + 0.5) * voxelSize;
        voxelCenters[count * 3 + 2] = splats.boundsMin[2] + (vz + 0.5) * voxelSize;

        if (colorAccumRead) {
          const base = i * 3;
          voxelColors[count * 3] = Math.max(0, Math.min(255, Math.round(colorAccumRead[base] / d)));
          voxelColors[count * 3 + 1] = Math.max(0, Math.min(255, Math.round(colorAccumRead[base + 1] / d)));
          voxelColors[count * 3 + 2] = Math.max(0, Math.min(255, Math.round(colorAccumRead[base + 2] / d)));
        } else {
          voxelColors[count * 3] = 200;
          voxelColors[count * 3 + 1] = 200;
          voxelColors[count * 3 + 2] = 200;
        }
        count += 1;
      }
    }
  }

  if (selectedVoxels === 0) {
    return {
      positions: new Float32Array(),
      colors: new Uint8Array(),
      isoThreshold: thresholdAtomic / options.atomicScale,
      dims,
      voxelSize,
      nonZeroVoxels,
      selectedVoxels,
      adapterInfo,
      mesh,
      voxelCenters: new Float32Array(),
      voxelColors: new Uint8Array(),
    };
  }

  const counterBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const selectedIndicesBuffer = device.createBuffer({
    size: align4(voxelCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const positionsBuffer = device.createBuffer({
    size: align4(options.maxPoints * 16),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const outColorsBuffer = device.createBuffer({
    size: align4(options.maxPoints * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const extractParamsBuffer = createExtractParamsBuffer(device, voxelCount, thresholdAtomic);
  const extractPipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: EXTRACT_SHADER }),
      entryPoint: "main",
    },
  });
  const extractBindGroup = device.createBindGroup({
    layout: extractPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: densityBuffer } },
      { binding: 1, resource: { buffer: counterBuffer } },
      { binding: 2, resource: { buffer: selectedIndicesBuffer } },
      { binding: 3, resource: { buffer: paramsBuffer } },
      { binding: 4, resource: { buffer: extractParamsBuffer } },
    ],
  });
  const pointCount = options.maxPoints;
  const sampleParamsBuffer = createSampleParamsBuffer(device, pointCount, selectedVoxels, options.seed, options.jitter);
  const samplePipeline = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({ code: noColor ? SAMPLE_SHADER_NO_COLOR : SAMPLE_SHADER }),
      entryPoint: "main",
    },
  });
  const sampleBindGroup = device.createBindGroup({
    layout: samplePipeline.getBindGroupLayout(0),
    entries: noColor
      ? [
          { binding: 0, resource: { buffer: densityBuffer } },
          { binding: 1, resource: { buffer: selectedIndicesBuffer } },
          { binding: 2, resource: { buffer: positionsBuffer } },
          { binding: 3, resource: { buffer: outColorsBuffer } },
          { binding: 4, resource: { buffer: paramsBuffer } },
          { binding: 5, resource: { buffer: sampleParamsBuffer } },
        ]
      : [
          { binding: 0, resource: { buffer: densityBuffer } },
          { binding: 1, resource: { buffer: colorAccumBuffer! } },
          { binding: 2, resource: { buffer: selectedIndicesBuffer } },
          { binding: 3, resource: { buffer: positionsBuffer } },
          { binding: 4, resource: { buffer: outColorsBuffer } },
          { binding: 5, resource: { buffer: paramsBuffer } },
          { binding: 6, resource: { buffer: sampleParamsBuffer } },
        ],
  });

  const encoder = device.createCommandEncoder();
  const extractPass = encoder.beginComputePass();
  extractPass.setPipeline(extractPipeline);
  extractPass.setBindGroup(0, extractBindGroup);
  extractPass.dispatchWorkgroups(Math.ceil(voxelCount / 64));
  extractPass.end();
  const samplePass = encoder.beginComputePass();
  samplePass.setPipeline(samplePipeline);
  samplePass.setBindGroup(0, sampleBindGroup);
  samplePass.dispatchWorkgroups(Math.ceil(pointCount / 64));
  samplePass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  const positionsRead = new Float32Array(await readBuffer(device, positionsBuffer, pointCount * 16));
  const colorsRead = new Uint32Array(await readBuffer(device, outColorsBuffer, pointCount * 4));
  const positions = new Float32Array(pointCount * 3);
  const colors = new Uint8Array(pointCount * 3);

  for (let i = 0; i < pointCount; i += 1) {
    positions[i * 3] = positionsRead[i * 4];
    positions[i * 3 + 1] = positionsRead[i * 4 + 1];
    positions[i * 3 + 2] = positionsRead[i * 4 + 2];
    const packed = colorsRead[i];
    colors[i * 3] = packed & 0xff;
    colors[i * 3 + 1] = (packed >>> 8) & 0xff;
    colors[i * 3 + 2] = (packed >>> 16) & 0xff;
  }

  return {
    positions,
    colors,
    isoThreshold: thresholdAtomic / options.atomicScale,
    dims,
    voxelSize,
    nonZeroVoxels,
    selectedVoxels,
    adapterInfo,
    mesh,
    voxelCenters,
    voxelColors,
  };
}

function createStorageBuffer(device: GPUDevice, data: Float32Array, usage: number): GPUBuffer {
  const buffer = device.createBuffer({
    size: align4(data.byteLength),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buffer;
}

function createParamsBuffer(
  device: GPUDevice,
  count: number,
  dims: [number, number, number],
  boundsMin: [number, number, number],
  boundsMax: [number, number, number],
  voxelSize: number,
  options: Pick<WebGpuVoxelOptions, "sigmaRadius" | "densityScale" | "minOpacity" | "atomicScale"> & {
    minHeightAtomicThreshold?: number;
  },
): GPUBuffer {
  const bytes = new ArrayBuffer(64);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = count;
  u32[1] = dims[0];
  u32[2] = dims[1];
  u32[3] = dims[2];
  f32[4] = boundsMin[0];
  f32[5] = boundsMin[1];
  f32[6] = boundsMin[2];
  f32[7] = voxelSize;
  f32[8] = boundsMax[0];
  f32[9] = boundsMax[1];
  f32[10] = boundsMax[2];
  f32[11] = options.minHeightAtomicThreshold ?? 0;
  f32[12] = options.sigmaRadius;
  f32[13] = options.densityScale;
  f32[14] = options.minOpacity;
  f32[15] = options.atomicScale;

  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function createExtractParamsBuffer(device: GPUDevice, maxPoints: number, threshold: number): GPUBuffer {
  const data = new Uint32Array([maxPoints, threshold, 0, 0]);
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createSampleParamsBuffer(
  device: GPUDevice,
  pointCount: number,
  selectedCount: number,
  seed: number,
  jitter: number,
): GPUBuffer {
  const bytes = new ArrayBuffer(16);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = pointCount;
  u32[1] = selectedCount;
  u32[2] = seed >>> 0;
  f32[3] = jitter;
  const buffer = device.createBuffer({
    size: bytes.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, bytes);
  return buffer;
}

function runComputePass(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  workgroups: number,
): void {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(workgroups);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const readSize = align4(size);
  const readback = device.createBuffer({
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, readSize);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  return readback.getMappedRange(0, readSize).slice(0);
}

function estimateAtomicThreshold(values: Uint32Array, percentile: number): number {
  const nonzero: number[] = [];
  for (const value of values) {
    if (value > 0) {
      nonzero.push(value);
    }
  }
  if (nonzero.length === 0) {
    return 1;
  }
  nonzero.sort((a, b) => a - b);
  const index = Math.min(nonzero.length - 1, Math.max(0, Math.floor(nonzero.length * percentile)));
  return Math.max(1, nonzero[index]);
}

function countAbove(values: Uint32Array, threshold: number): number {
  let count = 0;
  for (const value of values) {
    if (value >= threshold) {
      count += 1;
    }
  }
  return count;
}

async function describeAdapter(adapter: GPUAdapter): Promise<string> {
  if ("info" in adapter) {
    const info = adapter.info;
    return [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ");
  }
  if ("requestAdapterInfo" in adapter) {
    const info = await (adapter as GPUAdapter & { requestAdapterInfo(): Promise<GPUAdapterInfo> }).requestAdapterInfo();
    return [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" / ");
  }
  return "unknown";
}

function align4(value: number): number {
  return Math.ceil(value / 4) * 4;
}

const chunkPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();
const chunkSmoothPipelineCache = new WeakMap<GPUDevice, GPUComputePipeline>();

export interface ChunkVoxelizeOptions {
  sigmaRadius?: number;
  densityScale?: number;
  minOpacity?: number;
  atomicScale?: number;
  smoothIterations?: number;
  minHeightAtomicThreshold?: number;
  voxelSize?: number;
}

export interface ChunkVoxelizeResult {
  density: Uint32Array;
  dims: [number, number, number];
  voxelSize: number;
  minSurfaceHeight?: number;
}

export async function voxelizeChunkWithDevice(
  device: GPUDevice,
  splats: PackedSplatsForGpu,
  boundsMin: [number, number, number],
  boundsMax: [number, number, number],
  resolution: number,
  options: ChunkVoxelizeOptions = {},
): Promise<ChunkVoxelizeResult> {
  const sigmaRadius = options.sigmaRadius ?? 3;
  const densityScale = options.densityScale ?? 1.0;
  const minOpacity = options.minOpacity ?? 0.0;
  const atomicScale = options.atomicScale ?? 512;
  const smoothIterations = options.smoothIterations ?? 0;
  const minHeightAtomicThreshold = options.minHeightAtomicThreshold ?? 1;
  const extent = [
    boundsMax[0] - boundsMin[0],
    boundsMax[1] - boundsMin[1],
    boundsMax[2] - boundsMin[2],
  ];
  const longest = Math.max(extent[0], extent[1], extent[2], Number.EPSILON);
  const voxelSize = options.voxelSize ?? longest / resolution;
  const dims: [number, number, number] = [
    Math.max(1, Math.ceil(extent[0] / voxelSize)),
    Math.max(1, Math.ceil(extent[1] / voxelSize)),
    Math.max(1, Math.ceil(extent[2] / voxelSize)),
  ];
  const voxelCount = dims[0] * dims[1] * dims[2];

  const centersBuffer = createStorageBuffer(device, splats.centers, GPUBufferUsage.STORAGE);
  const scalesBuffer = createStorageBuffer(device, splats.scales, GPUBufferUsage.STORAGE);
  const quatsBuffer = createStorageBuffer(device, splats.quaternions, GPUBufferUsage.STORAGE);
  const colorsBuffer = createStorageBuffer(device, splats.colors, GPUBufferUsage.STORAGE);
  const densityBuffer = device.createBuffer({
    size: align4(voxelCount * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const minHeightBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(minHeightBuffer, 0, new Uint32Array([0xffffffff]));

  const paramsBuffer = createParamsBuffer(
    device,
    splats.count,
    dims,
    boundsMin,
    boundsMax,
    voxelSize,
    {
      sigmaRadius,
      densityScale,
      minOpacity,
      atomicScale,
      minHeightAtomicThreshold,
    },
  );

  let voxelPipeline = chunkPipelineCache.get(device);
  if (!voxelPipeline) {
    voxelPipeline = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: CHUNK_VOXEL_SHADER_WITH_MIN_HEIGHT }),
        entryPoint: "main",
      },
    });
    chunkPipelineCache.set(device, voxelPipeline);
  }

  const voxelBindGroup = device.createBindGroup({
    layout: voxelPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: centersBuffer } },
      { binding: 1, resource: { buffer: scalesBuffer } },
      { binding: 2, resource: { buffer: quatsBuffer } },
      { binding: 3, resource: { buffer: colorsBuffer } },
      { binding: 4, resource: { buffer: densityBuffer } },
      { binding: 5, resource: { buffer: paramsBuffer } },
      { binding: 6, resource: { buffer: minHeightBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(voxelPipeline);
  pass.setBindGroup(0, voxelBindGroup);
  pass.dispatchWorkgroups(Math.ceil(splats.count / 64));
  pass.end();
  device.queue.submit([encoder.finish()]);

  if (smoothIterations > 0) {
    const tempDensityBuffer = device.createBuffer({
      size: align4(voxelCount * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    let smoothPipeline = chunkSmoothPipelineCache.get(device);
    if (!smoothPipeline) {
      smoothPipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: SMOOTH_SHADER_NO_COLOR }),
          entryPoint: "main",
        },
      });
      chunkSmoothPipelineCache.set(device, smoothPipeline);
    }

    const bindGroupPing = device.createBindGroup({
      layout: smoothPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: densityBuffer } },
        { binding: 1, resource: { buffer: tempDensityBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });
    const bindGroupPong = device.createBindGroup({
      layout: smoothPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tempDensityBuffer } },
        { binding: 1, resource: { buffer: densityBuffer } },
        { binding: 2, resource: { buffer: paramsBuffer } },
      ],
    });

    const smoothWorkgroups = Math.ceil(voxelCount / 64);
    for (let iter = 0; iter < smoothIterations; iter += 1) {
      const smoothEncoder = device.createCommandEncoder();
      const smoothPass = smoothEncoder.beginComputePass();
      smoothPass.setPipeline(smoothPipeline);
      smoothPass.setBindGroup(0, iter % 2 === 0 ? bindGroupPing : bindGroupPong);
      smoothPass.dispatchWorkgroups(smoothWorkgroups);
      smoothPass.end();
      device.queue.submit([smoothEncoder.finish()]);
    }

    if (smoothIterations % 2 !== 0) {
      const copyEncoder = device.createCommandEncoder();
      copyEncoder.copyBufferToBuffer(tempDensityBuffer, 0, densityBuffer, 0, align4(voxelCount * 4));
      device.queue.submit([copyEncoder.finish()]);
    }

    await device.queue.onSubmittedWorkDone();
    tempDensityBuffer.destroy();
  }

  const readback = new Uint32Array(await readBuffer(device, densityBuffer, voxelCount * 4));
  const minHeightReadback = new Uint32Array(await readBuffer(device, minHeightBuffer, 4));
  const minSurfaceHeight =
    minHeightReadback[0] === 0xffffffff ? undefined : boundsMin[1] + (minHeightReadback[0] + 1) * voxelSize;

  centersBuffer.destroy();
  scalesBuffer.destroy();
  quatsBuffer.destroy();
  colorsBuffer.destroy();
  densityBuffer.destroy();
  minHeightBuffer.destroy();
  paramsBuffer.destroy();

  return { density: readback, dims, voxelSize, minSurfaceHeight };
}
