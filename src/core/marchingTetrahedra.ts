import type { Vec3 } from "./types.js";

export interface IsoSurfaceGrid {
  dims: Vec3;
  boundsMin: Vec3;
  voxelSize: number;
  density: Uint32Array;
  colorAccum?: Uint32Array;
  isoThreshold: number;
}

export interface TriangleMesh {
  positions: Float32Array;
  colors: Uint8Array;
  indices: Uint32Array;
}

interface GridSample {
  position: Vec3;
  color: Vec3;
  density: number;
}

interface MeshVertex {
  position: Vec3;
  color: Vec3;
}

const CORNER_OFFSETS: readonly Vec3[] = [
  [0, 0, 0],
  [1, 0, 0],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 1, 1],
  [0, 1, 1],
];

const TETS: readonly (readonly number[])[] = [
  [0, 5, 1, 6],
  [0, 1, 2, 6],
  [0, 2, 3, 6],
  [0, 3, 7, 6],
  [0, 7, 4, 6],
  [0, 4, 5, 6],
];

export function extractMarchingTetrahedraMesh(grid: IsoSurfaceGrid): TriangleMesh {
  const [dimX, dimY, dimZ] = grid.dims;
  if (dimX < 2 || dimY < 2 || dimZ < 2) {
    return emptyMesh();
  }

  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  const corners = new Array<GridSample>(8);

  for (let z = 0; z < dimZ - 1; z += 1) {
    for (let y = 0; y < dimY - 1; y += 1) {
      for (let x = 0; x < dimX - 1; x += 1) {
        for (let c = 0; c < CORNER_OFFSETS.length; c += 1) {
          const offset = CORNER_OFFSETS[c];
          corners[c] = readSample(grid, x + offset[0], y + offset[1], z + offset[2]);
        }
        for (const tet of TETS) {
          polygonizeTet(tet, corners, grid.isoThreshold, positions, colors, indices);
        }
      }
    }
  }

  return {
    positions: new Float32Array(positions),
    colors: new Uint8Array(colors),
    indices: new Uint32Array(indices),
  };
}

function polygonizeTet(
  tet: readonly number[],
  corners: readonly GridSample[],
  isoThreshold: number,
  positions: number[],
  colors: number[],
  indices: number[],
): void {
  const inside: number[] = [];
  const outside: number[] = [];
  for (const cornerIndex of tet) {
    if (corners[cornerIndex].density >= isoThreshold) {
      inside.push(cornerIndex);
    } else {
      outside.push(cornerIndex);
    }
  }

  if (inside.length === 0 || inside.length === 4) {
    return;
  }

  if (inside.length === 1) {
    const a = corners[inside[0]];
    addTriangle(
      interpolate(a, corners[outside[0]], isoThreshold),
      interpolate(a, corners[outside[1]], isoThreshold),
      interpolate(a, corners[outside[2]], isoThreshold),
      positions,
      colors,
      indices,
    );
    return;
  }

  if (inside.length === 3) {
    const a = corners[outside[0]];
    addTriangle(
      interpolate(a, corners[inside[2]], isoThreshold),
      interpolate(a, corners[inside[1]], isoThreshold),
      interpolate(a, corners[inside[0]], isoThreshold),
      positions,
      colors,
      indices,
    );
    return;
  }

  const a = corners[inside[0]];
  const b = corners[inside[1]];
  const c = corners[outside[0]];
  const d = corners[outside[1]];
  const ac = interpolate(a, c, isoThreshold);
  const bc = interpolate(b, c, isoThreshold);
  const bd = interpolate(b, d, isoThreshold);
  const ad = interpolate(a, d, isoThreshold);
  addTriangle(ac, bc, bd, positions, colors, indices);
  addTriangle(ac, bd, ad, positions, colors, indices);
}

function readSample(grid: IsoSurfaceGrid, x: number, y: number, z: number): GridSample {
  const index = gridIndex(grid.dims, x, y, z);
  const density = grid.density[index];
  return {
    position: [
      grid.boundsMin[0] + (x + 0.5) * grid.voxelSize,
      grid.boundsMin[1] + (y + 0.5) * grid.voxelSize,
      grid.boundsMin[2] + (z + 0.5) * grid.voxelSize,
    ],
    color: sampleColor(grid, index),
    density,
  };
}

function interpolate(a: GridSample, b: GridSample, isoThreshold: number): MeshVertex {
  const delta = b.density - a.density;
  const t = Math.abs(delta) <= Number.EPSILON
    ? 0.5
    : clamp01((isoThreshold - a.density) / delta);
  return {
    position: [
      lerp(a.position[0], b.position[0], t),
      lerp(a.position[1], b.position[1], t),
      lerp(a.position[2], b.position[2], t),
    ],
    color: [
      lerp(a.color[0], b.color[0], t),
      lerp(a.color[1], b.color[1], t),
      lerp(a.color[2], b.color[2], t),
    ],
  };
}

function addTriangle(
  a: MeshVertex,
  b: MeshVertex,
  c: MeshVertex,
  positions: number[],
  colors: number[],
  indices: number[],
): void {
  if (triangleAreaSquared(a.position, b.position, c.position) < 1e-18) {
    return;
  }

  const base = positions.length / 3;
  pushVertex(a, positions, colors);
  pushVertex(b, positions, colors);
  pushVertex(c, positions, colors);
  indices.push(base, base + 1, base + 2);
}

function pushVertex(vertex: MeshVertex, positions: number[], colors: number[]): void {
  positions.push(vertex.position[0], vertex.position[1], vertex.position[2]);
  colors.push(
    clampByte(Math.round(vertex.color[0])),
    clampByte(Math.round(vertex.color[1])),
    clampByte(Math.round(vertex.color[2])),
  );
}

function sampleColor(grid: IsoSurfaceGrid, index: number): Vec3 {
  const density = grid.density[index];
  if (!grid.colorAccum || density === 0) {
    return [180, 180, 180];
  }
  const base = index * 3;
  return [
    clampByte(Math.round(grid.colorAccum[base] / density)),
    clampByte(Math.round(grid.colorAccum[base + 1] / density)),
    clampByte(Math.round(grid.colorAccum[base + 2] / density)),
  ];
}

function triangleAreaSquared(a: Vec3, b: Vec3, c: Vec3): number {
  const abx = b[0] - a[0];
  const aby = b[1] - a[1];
  const abz = b[2] - a[2];
  const acx = c[0] - a[0];
  const acy = c[1] - a[1];
  const acz = c[2] - a[2];
  const crossX = aby * acz - abz * acy;
  const crossY = abz * acx - abx * acz;
  const crossZ = abx * acy - aby * acx;
  return crossX * crossX + crossY * crossY + crossZ * crossZ;
}

function gridIndex(dims: Vec3, x: number, y: number, z: number): number {
  return x + y * dims[0] + z * dims[0] * dims[1];
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function clampByte(value: number): number {
  return Math.max(0, Math.min(255, value));
}

function emptyMesh(): TriangleMesh {
  return {
    positions: new Float32Array(),
    colors: new Uint8Array(),
    indices: new Uint32Array(),
  };
}
