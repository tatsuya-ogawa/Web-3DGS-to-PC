import type { Quat, Vec3 } from "./types.js";

export function normalizeQuat(q: Quat): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len];
}

export function quatToMatrix(qIn: Quat): Float32Array {
  const q = normalizeQuat(qIn);
  const r = q[0];
  const x = q[1];
  const y = q[2];
  const z = q[3];

  return new Float32Array([
    1 - 2 * (y * y + z * z),
    2 * (x * y - r * z),
    2 * (x * z + r * y),
    2 * (x * y + r * z),
    1 - 2 * (x * x + z * z),
    2 * (y * z - r * x),
    2 * (x * z - r * y),
    2 * (y * z + r * x),
    1 - 2 * (x * x + y * y),
  ]);
}

export function rotateVec3(m: Float32Array, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

export function inverseRotateVec3(m: Float32Array, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}
