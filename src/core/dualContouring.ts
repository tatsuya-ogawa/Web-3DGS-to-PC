import type { Vec3 } from "./types.js";
import type { IsoSurfaceGrid, TriangleMesh } from "./marchingTetrahedra.js";

const CORNER_OFFSETS: readonly Vec3[] = [
  [0, 0, 0], // C0
  [1, 0, 0], // C1
  [1, 1, 0], // C2
  [0, 1, 0], // C3
  [0, 0, 1], // C4
  [1, 0, 1], // C5
  [1, 1, 1], // C6
  [0, 1, 1], // C7
];

// The 12 edges of a voxel cell connecting corners
const CELL_EDGES: readonly (readonly [number, number])[] = [
  [0, 1], [3, 2], [4, 5], [7, 6], // X edges
  [0, 3], [1, 2], [4, 7], [5, 6], // Y edges
  [0, 4], [1, 5], [2, 6], [3, 7], // Z edges
];

export function extractDualContouringMesh(grid: IsoSurfaceGrid): TriangleMesh {
  const [dimX, dimY, dimZ] = grid.dims;
  if (dimX < 2 || dimY < 2 || dimZ < 2) {
    return {
      positions: new Float32Array(),
      colors: new Uint8Array(),
      indices: new Uint32Array(),
    };
  }

  const voxelCount = dimX * dimY * dimZ;
  const isoThreshold = grid.isoThreshold;

  // Track vertex index generated for each active cell
  // Number of cells = (dimX - 1) * (dimY - 1) * (dimZ - 1)
  const cellVertexIndices = new Int32Array((dimX - 1) * (dimY - 1) * (dimZ - 1)).fill(-1);

  const positions: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];

  const getGridIndex = (x: number, y: number, z: number): number => {
    return x + y * dimX + z * dimX * dimY;
  };

  const getCellIndex = (x: number, y: number, z: number): number => {
    return x + y * (dimX - 1) + z * (dimX - 1) * (dimY - 1);
  };

  const getCornerDensity = (x: number, y: number, z: number): number => {
    return grid.density[getGridIndex(x, y, z)];
  };

  const getCornerColor = (x: number, y: number, z: number): [number, number, number] => {
    if (!grid.colorAccum) {
      return [180, 180, 180];
    }
    const idx = getGridIndex(x, y, z);
    const d = grid.density[idx];
    if (d === 0) {
      return [180, 180, 180];
    }
    const base = idx * 3;
    return [
      grid.colorAccum[base] / d,
      grid.colorAccum[base + 1] / d,
      grid.colorAccum[base + 2] / d,
    ];
  };

  // Step 1: Scan all voxel cells to generate a vertex in each boundary-crossing cell
  for (let z = 0; z < dimZ - 1; z += 1) {
    for (let y = 0; y < dimY - 1; y += 1) {
      for (let x = 0; x < dimX - 1; x += 1) {
        // Evaluate densities at 8 corners of the current cell
        let mask = 0;
        const densities = new Float32Array(8);
        for (let c = 0; c < 8; c += 1) {
          const offset = CORNER_OFFSETS[c];
          const d = getCornerDensity(x + offset[0], y + offset[1], z + offset[2]);
          densities[c] = d;
          if (d >= isoThreshold) {
            mask |= 1 << c;
          }
        }

        // Skip completely inside or completely outside cells
        if (mask === 0 || mask === 255) {
          continue;
        }

        // Compute vertex by averaging all edge crossings (Dual Centroid)
        let sumX = 0;
        let sumY = 0;
        let sumZ = 0;
        let sumR = 0;
        let sumG = 0;
        let sumB = 0;
        let crossings = 0;

        for (const [c1, c2] of CELL_EDGES) {
          const c1Inside = (mask & (1 << c1)) !== 0;
          const c2Inside = (mask & (1 << c2)) !== 0;
          if (c1Inside !== c2Inside) {
            // Edge is crossed! Linearly interpolate the crossing position and color
            const d1 = densities[c1];
            const d2 = densities[c2];
            const t = Math.abs(d2 - d1) <= Number.EPSILON ? 0.5 : (isoThreshold - d1) / (d2 - d1);
            const tClamped = Math.max(0, Math.min(1, t));

            const o1 = CORNER_OFFSETS[c1];
            const o2 = CORNER_OFFSETS[c2];

            const px = grid.boundsMin[0] + (x + o1[0] + (o2[0] - o1[0]) * tClamped + 0.5) * grid.voxelSize;
            const py = grid.boundsMin[1] + (y + o1[1] + (o2[1] - o1[1]) * tClamped + 0.5) * grid.voxelSize;
            const pz = grid.boundsMin[2] + (z + o1[2] + (o2[2] - o1[2]) * tClamped + 0.5) * grid.voxelSize;

            const col1 = getCornerColor(x + o1[0], y + o1[1], z + o1[2]);
            const col2 = getCornerColor(x + o2[0], y + o2[1], z + o2[2]);

            sumX += px;
            sumY += py;
            sumZ += pz;
            sumR += col1[0] + (col2[0] - col1[0]) * tClamped;
            sumG += col1[1] + (col2[1] - col1[1]) * tClamped;
            sumB += col1[2] + (col2[2] - col1[2]) * tClamped;
            crossings += 1;
          }
        }

        if (crossings > 0) {
          const vertIdx = positions.length / 3;
          positions.push(sumX / crossings, sumY / crossings, sumZ / crossings);
          colors.push(
            Math.max(0, Math.min(255, Math.round(sumR / crossings))),
            Math.max(0, Math.min(255, Math.round(sumG / crossings))),
            Math.max(0, Math.min(255, Math.round(sumB / crossings))),
          );
          cellVertexIndices[getCellIndex(x, y, z)] = vertIdx;
        }
      }
    }
  }

  // Step 2: Scan grid edges and output quads for crossed edges
  for (let z = 0; z < dimZ - 1; z += 1) {
    for (let y = 0; y < dimY - 1; y += 1) {
      for (let x = 0; x < dimX - 1; x += 1) {
        const d_c = getCornerDensity(x, y, z);
        const startInside = d_c >= isoThreshold;

        // X-axis edge
        if (x < dimX - 2 && y >= 1 && z >= 1) {
          const d_x = getCornerDensity(x + 1, y, z);
          if (startInside !== (d_x >= isoThreshold)) {
            const p0 = cellVertexIndices[getCellIndex(x, y, z)];
            const p1 = cellVertexIndices[getCellIndex(x, y - 1, z)];
            const p2 = cellVertexIndices[getCellIndex(x, y - 1, z - 1)];
            const p3 = cellVertexIndices[getCellIndex(x, y, z - 1)];
            if (p0 >= 0 && p1 >= 0 && p2 >= 0 && p3 >= 0) {
              if (startInside) {
                indices.push(p0, p3, p2, p0, p2, p1);
              } else {
                indices.push(p0, p1, p2, p0, p2, p3);
              }
            }
          }
        }

        // Y-axis edge
        if (y < dimY - 2 && x >= 1 && z >= 1) {
          const d_y = getCornerDensity(x, y + 1, z);
          if (startInside !== (d_y >= isoThreshold)) {
            const p0 = cellVertexIndices[getCellIndex(x, y, z)];
            const p1 = cellVertexIndices[getCellIndex(x, y, z - 1)];
            const p2 = cellVertexIndices[getCellIndex(x - 1, y, z - 1)];
            const p3 = cellVertexIndices[getCellIndex(x - 1, y, z)];
            if (p0 >= 0 && p1 >= 0 && p2 >= 0 && p3 >= 0) {
              if (startInside) {
                indices.push(p0, p1, p2, p0, p2, p3);
              } else {
                indices.push(p0, p3, p2, p0, p2, p1);
              }
            }
          }
        }

        // Z-axis edge
        if (z < dimZ - 2 && x >= 1 && y >= 1) {
          const d_z = getCornerDensity(x, y, z + 1);
          if (startInside !== (d_z >= isoThreshold)) {
            const p0 = cellVertexIndices[getCellIndex(x, y, z)];
            const p1 = cellVertexIndices[getCellIndex(x - 1, y, z)];
            const p2 = cellVertexIndices[getCellIndex(x - 1, y - 1, z)];
            const p3 = cellVertexIndices[getCellIndex(x, y - 1, z)];
            if (p0 >= 0 && p1 >= 0 && p2 >= 0 && p3 >= 0) {
              if (startInside) {
                indices.push(p0, p3, p2, p0, p2, p1);
              } else {
                indices.push(p0, p1, p2, p0, p2, p3);
              }
            }
          }
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
