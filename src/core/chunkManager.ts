import type { PackedSplatsForGpu } from "../browser/webgpuVoxel.js";
import { voxelizeChunkWithDevice } from "../browser/webgpuVoxel.js";

export interface VoxelGrid {
  density: Uint32Array;
  boundsMin: [number, number, number];
  boundsMax: [number, number, number];
  voxelSize: number;
  dims: [number, number, number];
  supportHeight: HeightField;
}

export interface HeightField {
  height: Float32Array;
  state: Uint8Array;
  width: number;
  depth: number;
  fallbackHeight: number;
}

export type SupportFillMode = "nearby" | "fallback" | "min";
export type ChunkGenerationMode = "heightfield" | "voxels" | "both";

export interface SurfaceHeightQuery {
  height?: number;
  loading: boolean;
}

export interface ChunkVoxelManagerOptions {
  maxConcurrentJobs?: number;
  smoothIterations?: number;
  supportDensityThreshold?: number;
  supportFillIterations?: number;
  supportFillMaxHeightRange?: number;
  supportFillMode?: SupportFillMode;
  generationMode?: ChunkGenerationMode;
  activeHorizontalRadius?: number;
  activeVerticalBelow?: number;
  activeVerticalAbove?: number;
}

export interface ChunkVoxelManagerStatus {
  generationMode: ChunkGenerationMode;
  activeJobs: number;
  pendingVoxelChunks: number;
  pendingSupportTiles: number;
  loadingVoxelChunks: number;
  loadingSupportTiles: number;
  loadedVoxelChunks: number;
  loadedSupportTiles: number;
  activeVoxelChunks: number;
  activeSupportTiles: number;
  voxelRevision: number;
  supportRevision: number;
}

interface ChunkRequest {
  key: string;
  cx: number;
  cy: number;
  cz: number;
  priority: number;
  generation: number;
}

interface SupportTileRequest {
  key: string;
  cx: number;
  cz: number;
  priority: number;
  generation: number;
}

interface ColumnRange {
  minY: number;
  maxY: number;
}

interface HeightMipmapLevel {
  width: number;
  depth: number;
  sum: Float32Array;
  count: Uint32Array;
  min: Float32Array;
  max: Float32Array;
}

export class SplatSpatialBinning {
  private splats: PackedSplatsForGpu;
  private chunkSize: number;
  private padding: number;
  private chunkBuckets: Map<string, number[]>;
  private columnBuckets: Map<string, number[]>;
  private columnRanges: Map<string, ColumnRange>;

  constructor(splats: PackedSplatsForGpu, chunkSize: number = 16.0, padding: number = 2.0) {
    this.splats = splats;
    this.chunkSize = chunkSize;
    this.padding = padding;
    this.chunkBuckets = new Map();
    this.columnBuckets = new Map();
    this.columnRanges = new Map();
    this.binSplats();
  }

  private binSplats(): void {
    const centers = this.splats.centers;
    const count = this.splats.count;
    const size = this.chunkSize;
    const pad = this.padding;

    for (let i = 0; i < count; i += 1) {
      const base = i * 4;
      const x = centers[base];
      const y = centers[base + 1];
      const z = centers[base + 2];

      // Central chunk coordinates
      const minCx = Math.floor((x - pad) / size);
      const maxCx = Math.floor((x + pad) / size);
      const minCy = Math.floor((y - pad) / size);
      const maxCy = Math.floor((y + pad) / size);
      const minCz = Math.floor((z - pad) / size);
      const maxCz = Math.floor((z + pad) / size);

      for (let tx = minCx; tx <= maxCx; tx += 1) {
        for (let tz = minCz; tz <= maxCz; tz += 1) {
          const columnKey = `${tx},${tz}`;
          let columnBucket = this.columnBuckets.get(columnKey);
          if (!columnBucket) {
            columnBucket = [];
            this.columnBuckets.set(columnKey, columnBucket);
          }
          columnBucket.push(i);
          const columnRange = this.columnRanges.get(columnKey);
          if (columnRange) {
            columnRange.minY = Math.min(columnRange.minY, y);
            columnRange.maxY = Math.max(columnRange.maxY, y);
          } else {
            this.columnRanges.set(columnKey, { minY: y, maxY: y });
          }

          for (let ty = minCy; ty <= maxCy; ty += 1) {
            const key = `${tx},${ty},${tz}`;
            let bucket = this.chunkBuckets.get(key);
            if (!bucket) {
              bucket = [];
              this.chunkBuckets.set(key, bucket);
            }
            bucket.push(i);
          }
        }
      }
    }
  }

  public getSplatSubsetForChunk(cx: number, cy: number, cz: number): PackedSplatsForGpu | undefined {
    const key = `${cx},${cy},${cz}`;
    const indices = this.chunkBuckets.get(key);
    if (!indices || indices.length === 0) {
      return undefined;
    }

    return this.createSplatSubset(
      indices,
      [cx * this.chunkSize, cy * this.chunkSize, cz * this.chunkSize],
      [(cx + 1) * this.chunkSize, (cy + 1) * this.chunkSize, (cz + 1) * this.chunkSize],
    );
  }

  public getSplatSubsetForColumn(cx: number, cz: number): PackedSplatsForGpu | undefined {
    const key = `${cx},${cz}`;
    const indices = this.columnBuckets.get(key);
    const range = this.columnRanges.get(key);
    if (!indices || indices.length === 0 || !range) {
      return undefined;
    }

    const yMin = range.minY - this.padding;
    const yMax = range.maxY + this.padding;
    return this.createSplatSubset(
      indices,
      [cx * this.chunkSize, yMin, cz * this.chunkSize],
      [(cx + 1) * this.chunkSize, yMax, (cz + 1) * this.chunkSize],
    );
  }

  private createSplatSubset(
    indices: readonly number[],
    boundsMin: [number, number, number],
    boundsMax: [number, number, number],
  ): PackedSplatsForGpu {
    const subCount = indices.length;
    const centers = new Float32Array(subCount * 4);
    const scales = new Float32Array(subCount * 4);
    const quaternions = new Float32Array(subCount * 4);
    const colors = new Float32Array(subCount * 4);

    const origCenters = this.splats.centers;
    const origScales = this.splats.scales;
    const origQuats = this.splats.quaternions;
    const origColors = this.splats.colors;

    for (let i = 0; i < subCount; i += 1) {
      const idx = indices[i];
      const origBase = idx * 4;
      const subBase = i * 4;

      centers[subBase] = origCenters[origBase];
      centers[subBase + 1] = origCenters[origBase + 1];
      centers[subBase + 2] = origCenters[origBase + 2];
      centers[subBase + 3] = origCenters[origBase + 3];

      scales[subBase] = origScales[origBase];
      scales[subBase + 1] = origScales[origBase + 1];
      scales[subBase + 2] = origScales[origBase + 2];
      scales[subBase + 3] = origScales[origBase + 3];

      quaternions[subBase] = origQuats[origBase];
      quaternions[subBase + 1] = origQuats[origBase + 1];
      quaternions[subBase + 2] = origQuats[origBase + 2];
      quaternions[subBase + 3] = origQuats[origBase + 3];

      colors[subBase] = origColors[origBase];
      colors[subBase + 1] = origColors[origBase + 1];
      colors[subBase + 2] = origColors[origBase + 2];
      colors[subBase + 3] = origColors[origBase + 3];
    }

    return {
      centers,
      scales,
      quaternions,
      colors,
      boundsMin,
      boundsMax,
      count: subCount,
    };
  }

  public get chunkSizeVal(): number {
    return this.chunkSize;
  }

  public getDensestChunkCoordinates(): [number, number, number] {
    return this.getChunkCoordinatesByDensity(1)[0] ?? [0, 0, 0];
  }

  public getChunkCoordinatesByDensity(limit: number): [number, number, number][] {
    return [...this.chunkBuckets.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, Math.max(0, Math.floor(limit)))
      .map(([key]) => key.split(",").map(Number) as [number, number, number]);
  }

  public getColumnCoordinatesByDensity(limit: number): [number, number][] {
    return [...this.columnBuckets.entries()]
      .sort((a, b) => b[1].length - a[1].length)
      .slice(0, Math.max(0, Math.floor(limit)))
      .map(([key]) => key.split(",").map(Number) as [number, number]);
  }
}

export class ChunkVoxelManager {
  private device: GPUDevice;
  private binning: SplatSpatialBinning;
  private chunkResolution: number;
  private loadedChunks: Map<string, VoxelGrid>;
  private supportHeightTiles: Map<string, HeightField>;
  private loadingQueue: Set<string>;
  private pendingChunks: Map<string, ChunkRequest>;
  private loadingSupportTiles: Set<string>;
  private pendingSupportTiles: Map<string, SupportTileRequest>;
  private activeKeys: Set<string>;
  private activeSupportKeys: Set<string>;
  private activeJobs: number;
  private maxConcurrentJobs: number;
  private smoothIterations: number;
  private supportDensityThreshold: number;
  private supportFillIterations: number;
  private supportFillMaxHeightRange: number;
  private supportFillMode: SupportFillMode;
  private generationMode: ChunkGenerationMode;
  private activeHorizontalRadius: number;
  private activeVerticalBelow: number;
  private activeVerticalAbove: number;
  private activeCenterKey: string | undefined;
  private activeYMin: number;
  private activeYMax: number;
  private generation: number;
  private voxelRevision: number;
  private supportRevision: number;
  public enableHeightfieldSmoothing: boolean = true;

  constructor(
    device: GPUDevice,
    binning: SplatSpatialBinning,
    chunkResolution: number = 64,
    options: ChunkVoxelManagerOptions = {},
  ) {
    this.device = device;
    this.binning = binning;
    this.chunkResolution = chunkResolution;
    this.loadedChunks = new Map();
    this.supportHeightTiles = new Map();
    this.loadingQueue = new Set();
    this.pendingChunks = new Map();
    this.loadingSupportTiles = new Set();
    this.pendingSupportTiles = new Map();
    this.activeKeys = new Set();
    this.activeSupportKeys = new Set();
    this.activeJobs = 0;
    this.maxConcurrentJobs = options.maxConcurrentJobs ?? 2;
    this.smoothIterations = options.smoothIterations ?? 1;
    this.supportDensityThreshold = options.supportDensityThreshold ?? 120;
    this.supportFillIterations = options.supportFillIterations ?? 18;
    this.supportFillMaxHeightRange = options.supportFillMaxHeightRange ?? 1.2;
    this.supportFillMode = options.supportFillMode ?? "fallback";
    this.generationMode = options.generationMode ?? "both";
    this.activeHorizontalRadius = Math.max(0, Math.floor(options.activeHorizontalRadius ?? 1));
    this.activeVerticalBelow = Math.max(0, Math.floor(options.activeVerticalBelow ?? 1));
    this.activeVerticalAbove = Math.max(0, Math.floor(options.activeVerticalAbove ?? 1));
    this.activeCenterKey = undefined;
    this.activeYMin = 0;
    this.activeYMax = 0;
    this.generation = 0;
    this.voxelRevision = 0;
    this.supportRevision = 0;
  }

  public setSmoothIterations(smoothIterations: number): void {
    const normalized = Math.max(0, Math.floor(smoothIterations));
    if (normalized === this.smoothIterations) {
      return;
    }

    this.smoothIterations = normalized;
    this.generation += 1;
    this.loadedChunks.clear();
    this.supportHeightTiles.clear();
    this.pendingChunks.clear();
    this.loadingSupportTiles.clear();
    this.pendingSupportTiles.clear();
    this.activeSupportKeys.clear();
    this.bumpVoxelRevision();
    this.bumpSupportRevision();
    this.activeCenterKey = undefined;
  }

  public setSupportFillMode(supportFillMode: SupportFillMode): void {
    this.supportFillMode = supportFillMode;
  }

  public setGenerationMode(generationMode: ChunkGenerationMode): void {
    if (generationMode === this.generationMode) {
      return;
    }

    this.generationMode = generationMode;
    this.generation += 1;
    this.pendingChunks.clear();
    this.pendingSupportTiles.clear();
    this.loadingQueue.clear();
    this.loadingSupportTiles.clear();
    if (!this.shouldGenerateVoxelChunks()) {
      this.loadedChunks.clear();
      this.bumpVoxelRevision();
    }
    if (!this.shouldGenerateSupportTiles()) {
      this.supportHeightTiles.clear();
      this.bumpSupportRevision();
    }
    this.activeKeys.clear();
    this.activeSupportKeys.clear();
    this.activeCenterKey = undefined;
  }

  public async loadChunkNow(cx: number, cy: number, cz: number): Promise<VoxelGrid> {
    const key = `${cx},${cy},${cz}`;
    const loaded = this.loadedChunks.get(key);
    if (loaded) {
      return loaded;
    }

    const generation = this.generation;
    const chunk = await this.createVoxelGridForChunk(cx, cy, cz);
    if (generation === this.generation) {
      this.loadedChunks.set(key, chunk);
      this.bumpVoxelRevision();
    }
    return chunk;
  }

  public async loadSupportHeightTileNow(cx: number, cz: number): Promise<HeightField> {
    const key = getSupportTileKey(cx, cz);
    const loaded = this.supportHeightTiles.get(key);
    if (loaded) {
      return loaded;
    }

    const generation = this.generation;
    const tile = await this.createSupportHeightTile(cx, cz);
    if (generation === this.generation) {
      this.supportHeightTiles.set(key, tile);
      this.bumpSupportRevision();
    }
    return tile;
  }

  private shouldGenerateVoxelChunks(): boolean {
    return this.generationMode === "voxels" || this.generationMode === "both";
  }

  private shouldGenerateSupportTiles(): boolean {
    return this.generationMode === "heightfield" || this.generationMode === "both";
  }

  public updateActiveRegion(playerX: number, playerY: number, playerZ: number): { loaded: number; total: number } {
    const size = this.binning.chunkSizeVal;
    const pcx = Math.floor(playerX / size);
    const pcy = Math.floor(playerY / size);
    const pcz = Math.floor(playerZ / size);
    const centerKey = `${pcx},${pcy},${pcz}`;

    if (centerKey === this.activeCenterKey) {
      this.processQueue();
      const wantsVoxels = this.shouldGenerateVoxelChunks();
      return {
        loaded: wantsVoxels ? this.loadedChunks.size : this.supportHeightTiles.size,
        total: wantsVoxels ? this.activeKeys.size : this.activeSupportKeys.size,
      };
    }

    const activeKeys = new Set<string>();
    const activeSupportKeys = new Set<string>();
    const wantsSupport = this.shouldGenerateSupportTiles();
    const wantsVoxels = this.shouldGenerateVoxelChunks();
    const hr = this.activeHorizontalRadius;
    const yMinOffset = -this.activeVerticalBelow;
    const yMaxOffset = this.activeVerticalAbove;
    this.activeYMin = pcy + yMinOffset;
    this.activeYMax = pcy + yMaxOffset;

    for (let dx = -hr; dx <= hr; dx += 1) {
      for (let dz = -hr; dz <= hr; dz += 1) {
        const supportCx = pcx + dx;
        const supportCz = pcz + dz;
        const supportKey = getSupportTileKey(supportCx, supportCz);
        if (wantsSupport) {
          activeSupportKeys.add(supportKey);
        }
        if (
          wantsSupport &&
          !this.supportHeightTiles.has(supportKey) &&
          !this.loadingSupportTiles.has(supportKey) &&
          !this.pendingSupportTiles.has(supportKey)
        ) {
          this.pendingSupportTiles.set(supportKey, {
            key: supportKey,
            cx: supportCx,
            cz: supportCz,
            priority: dx * dx + dz * dz,
            generation: this.generation,
          });
        }

        for (let dy = yMinOffset; dy <= yMaxOffset; dy += 1) {
          const cx = supportCx;
          const cy = pcy + dy;
          const cz = supportCz;
          const key = `${cx},${cy},${cz}`;
          if (wantsVoxels) {
            activeKeys.add(key);
          }

          // If not loaded and not already scheduled, enqueue. Jobs are drained
          // progressively so one animation frame does not spawn all 27 chunks.
          if (
            wantsVoxels &&
            !this.loadedChunks.has(key) &&
            !this.loadingQueue.has(key) &&
            !this.pendingChunks.has(key)
          ) {
            this.pendingChunks.set(key, {
              key,
              cx,
              cy,
              cz,
              priority: dx * dx + dy * dy + dz * dz,
              generation: this.generation,
            });
          }
        }
      }
    }

    this.activeCenterKey = centerKey;
    this.activeKeys = activeKeys;
    this.activeSupportKeys = activeSupportKeys;

    for (const key of this.pendingChunks.keys()) {
      if (!activeKeys.has(key)) {
        this.pendingChunks.delete(key);
      }
    }
    for (const key of this.pendingSupportTiles.keys()) {
      if (!activeSupportKeys.has(key)) {
        this.pendingSupportTiles.delete(key);
      }
    }

    // Purge chunks that fall out of the active window
    let purgedVoxelChunks = false;
    for (const key of this.loadedChunks.keys()) {
      if (!activeKeys.has(key)) {
        this.loadedChunks.delete(key);
        purgedVoxelChunks = true;
      }
    }
    if (purgedVoxelChunks) {
      this.bumpVoxelRevision();
    }
    let purgedSupportTiles = false;
    for (const key of this.supportHeightTiles.keys()) {
      if (!activeSupportKeys.has(key)) {
        this.supportHeightTiles.delete(key);
        purgedSupportTiles = true;
      }
    }
    if (purgedSupportTiles) {
      this.bumpSupportRevision();
    }

    this.processQueue();

    return {
      loaded: wantsVoxels ? this.loadedChunks.size : this.supportHeightTiles.size,
      total: wantsVoxels ? activeKeys.size : activeSupportKeys.size,
    };
  }

  private processQueue(): void {
    let openSlots = this.maxConcurrentJobs - this.activeJobs;
    const wantsSupport = this.shouldGenerateSupportTiles();
    const wantsVoxels = this.shouldGenerateVoxelChunks();
    if (
      openSlots <= 0 ||
      ((!wantsSupport || this.pendingSupportTiles.size === 0) && (!wantsVoxels || this.pendingChunks.size === 0))
    ) {
      return;
    }

    if (wantsSupport) {
      const supportRequests = [...this.pendingSupportTiles.values()].sort((a, b) => a.priority - b.priority);
      for (let i = 0; openSlots > 0 && i < supportRequests.length; i += 1) {
        const request = supportRequests[i];
        this.pendingSupportTiles.delete(request.key);
        if (
          !this.activeSupportKeys.has(request.key) ||
          this.supportHeightTiles.has(request.key) ||
          this.loadingSupportTiles.has(request.key)
        ) {
          continue;
        }

        this.loadingSupportTiles.add(request.key);
        this.activeJobs += 1;
        openSlots -= 1;
        void this.buildSupportHeightTileAsync(request)
          .catch((err) => {
            console.error(`Failed to build support height tile ${request.key}:`, err);
          })
          .finally(() => {
            this.loadingSupportTiles.delete(request.key);
            this.activeJobs = Math.max(0, this.activeJobs - 1);
            this.processQueue();
          });
      }
    }

    if (!wantsVoxels || openSlots <= 0 || this.pendingChunks.size === 0) {
      return;
    }

    const requests = [...this.pendingChunks.values()].sort((a, b) => a.priority - b.priority);
    for (let i = 0; i < openSlots && i < requests.length; i += 1) {
      const request = requests[i];
      this.pendingChunks.delete(request.key);
      if (
        !this.activeKeys.has(request.key) ||
        this.loadedChunks.has(request.key) ||
        this.loadingQueue.has(request.key)
      ) {
        continue;
      }

      this.loadingQueue.add(request.key);
      this.activeJobs += 1;
      void this.voxelizeChunkAsync(request)
        .catch((err) => {
          console.error(`Failed to voxelize chunk ${request.key}:`, err);
        })
        .finally(() => {
          this.loadingQueue.delete(request.key);
          this.activeJobs = Math.max(0, this.activeJobs - 1);
          this.processQueue();
        });
    }
  }

  private async voxelizeChunkAsync(request: ChunkRequest): Promise<void> {
    const { cx, cy, cz, key, generation } = request;
    await yieldToMainThread();

    if (generation !== this.generation || !this.activeKeys.has(key)) {
      return;
    }

    const chunk = await this.createVoxelGridForChunk(cx, cy, cz);
    if (generation !== this.generation || !this.activeKeys.has(key)) {
      return;
    }

    this.loadedChunks.set(key, chunk);
    this.bumpVoxelRevision();
  }

  private async buildSupportHeightTileAsync(request: SupportTileRequest): Promise<void> {
    const { cx, cz, key, generation } = request;
    await yieldToMainThread();

    if (generation !== this.generation || !this.activeSupportKeys.has(key)) {
      return;
    }

    const tile = await this.createSupportHeightTile(cx, cz);
    if (generation !== this.generation || !this.activeSupportKeys.has(key)) {
      return;
    }

    this.supportHeightTiles.set(key, tile);
    this.bumpSupportRevision();
  }

  private async createVoxelGridForChunk(cx: number, cy: number, cz: number): Promise<VoxelGrid> {
    const subset = this.binning.getSplatSubsetForChunk(cx, cy, cz);
    if (!subset) {
      // No splats in this chunk, store an empty voxel grid representing empty space.
      return {
        density: new Uint32Array(this.chunkResolution ** 3),
        boundsMin: [cx * this.binning.chunkSizeVal, cy * this.binning.chunkSizeVal, cz * this.binning.chunkSizeVal],
        boundsMax: [(cx + 1) * this.binning.chunkSizeVal, (cy + 1) * this.binning.chunkSizeVal, (cz + 1) * this.binning.chunkSizeVal],
        voxelSize: this.binning.chunkSizeVal / this.chunkResolution,
        dims: [this.chunkResolution, this.chunkResolution, this.chunkResolution],
        supportHeight: createEmptyHeightField(this.chunkResolution, this.chunkResolution),
      };
    }

    const boundsMin = subset.boundsMin;
    const boundsMax = subset.boundsMax;
    const voxelized = await voxelizeChunkWithDevice(
      this.device,
      subset,
      boundsMin,
      boundsMax,
      this.chunkResolution,
      {
        smoothIterations: this.smoothIterations,
        minHeightAtomicThreshold: this.supportDensityThreshold,
      },
    );
    const density = voxelized.density;

    return {
      density,
      boundsMin,
      boundsMax,
      voxelSize: voxelized.voxelSize,
      dims: voxelized.dims,
      supportHeight: buildSupportHeightField(
        density,
        voxelized.dims,
        boundsMin,
        voxelized.voxelSize,
        this.supportDensityThreshold,
        this.supportFillIterations,
        this.supportFillMaxHeightRange,
      ),
    };
  }

  private async createSupportHeightTile(cx: number, cz: number): Promise<HeightField> {
    const subset = this.binning.getSplatSubsetForColumn(cx, cz);
    if (!subset) {
      return createEmptyHeightField(this.chunkResolution, this.chunkResolution);
    }

    const voxelSize = this.binning.chunkSizeVal / this.chunkResolution;
    const voxelized = await voxelizeChunkWithDevice(
      this.device,
      subset,
      subset.boundsMin,
      subset.boundsMax,
      this.chunkResolution,
      {
        smoothIterations: this.smoothIterations,
        minHeightAtomicThreshold: this.supportDensityThreshold,
        voxelSize,
      },
    );

    return buildSupportHeightField(
      voxelized.density,
      voxelized.dims,
      subset.boundsMin,
      voxelized.voxelSize,
      this.supportDensityThreshold,
      this.supportFillIterations,
      this.supportFillMaxHeightRange,
    );
  }

  public getVoxelAtWorldPosition(x: number, y: number, z: number): { density: number; voxelSize: number } | undefined {
    const size = this.binning.chunkSizeVal;
    const cx = Math.floor(x / size);
    const cy = Math.floor(y / size);
    const cz = Math.floor(z / size);

    const key = `${cx},${cy},${cz}`;
    const chunk = this.loadedChunks.get(key);
    if (!chunk) {
      return undefined;
    }

    const rx = x - cx * size;
    const ry = y - cy * size;
    const rz = z - cz * size;

    const [dimX, dimY, dimZ] = chunk.dims;
    const vx = Math.max(0, Math.min(dimX - 1, Math.floor(rx / chunk.voxelSize)));
    const vy = Math.max(0, Math.min(dimY - 1, Math.floor(ry / chunk.voxelSize)));
    const vz = Math.max(0, Math.min(dimZ - 1, Math.floor(rz / chunk.voxelSize)));

    const index = vx + vy * dimX + vz * dimX * dimY;
    return {
      density: chunk.density[index],
      voxelSize: chunk.voxelSize,
    };
  }

  public getSurfaceHeightAtWorldXZ(
    x: number,
    z: number,
    densityThreshold: number,
    maxY: number = Number.POSITIVE_INFINITY,
  ): number | undefined {
    return this.querySurfaceHeightAtWorldXZ(x, z, densityThreshold, maxY).height;
  }

  public querySurfaceHeightAtWorldXZ(
    x: number,
    z: number,
    densityThreshold: number,
    maxY: number = Number.POSITIVE_INFINITY,
  ): SurfaceHeightQuery {
    let bestHeight: number | undefined;
    let loading = false;
    const size = this.binning.chunkSizeVal;
    const cx = Math.floor(x / size);
    const cz = Math.floor(z / size);

    if (this.activeKeys.size > 0) {
      const maxCy = Number.isFinite(maxY)
        ? Math.min(this.activeYMax, Math.floor(maxY / size))
        : this.activeYMax;
      for (let cy = maxCy; cy >= this.activeYMin; cy -= 1) {
        const key = `${cx},${cy},${cz}`;
        const chunk = this.loadedChunks.get(key);
        if (!chunk) {
          if (this.activeKeys.has(key) || this.pendingChunks.has(key) || this.loadingQueue.has(key)) {
            loading = true;
          }
          continue;
        }

        const height = this.getSurfaceHeightInChunk(chunk, x, z, maxY);
        if (height !== undefined && (bestHeight === undefined || height > bestHeight)) {
          bestHeight = height;
        }
      }
      return { height: bestHeight, loading: bestHeight === undefined && loading };
    }

    for (const chunk of this.loadedChunks.values()) {
      const height = this.getSurfaceHeightInChunk(chunk, x, z, maxY);
      if (height === undefined) {
        continue;
      }
      if (bestHeight === undefined || height > bestHeight) {
        bestHeight = height;
      }
    }

    return { height: bestHeight, loading: false };
  }

  public querySupportHeightAtWorldXZ(
    x: number,
    z: number,
    densityThreshold: number,
    targetY: number,
  ): SurfaceHeightQuery {
    const size = this.binning.chunkSizeVal;
    const cx = Math.floor(x / size);
    const cz = Math.floor(z / size);
    const tile = this.supportHeightTiles.get(getSupportTileKey(cx, cz));
    if (!tile) {
      return { loading: this.isSupportColumnLoading(cx, cz) };
    }

    if (this.enableHeightfieldSmoothing) {
      const R = tile.width;
      const voxelSize = size / R;
      const gx = (x / voxelSize) - 0.5;
      const gz = (z / voxelSize) - 0.5;
      const g_x0 = Math.floor(gx);
      const g_z0 = Math.floor(gz);
      const tx = gx - g_x0;
      const tz = gz - g_z0;

      let sumHeight = 0;
      let sumWeight = 0;
      let anyLoading = false;

      for (let dz = 0; dz <= 1; dz += 1) {
        for (let dx = 0; dx <= 1; dx += 1) {
          const g_x = g_x0 + dx;
          const g_z = g_z0 + dz;
          const tileCx = Math.floor(g_x / R);
          const tileCz = Math.floor(g_z / R);
          const tileKey = getSupportTileKey(tileCx, tileCz);
          const sampleTile = this.supportHeightTiles.get(tileKey);

          if (!sampleTile) {
            if (this.isSupportColumnLoading(tileCx, tileCz)) {
              anyLoading = true;
            }
            continue;
          }

          const vx = g_x - tileCx * R;
          const vz = g_z - tileCz * R;
          const index = vx + vz * R;
          const h = this.resolveSupportHeight(sampleTile, index);

          if (h !== undefined && Number.isFinite(h)) {
            const wx = dx === 0 ? (1 - tx) : tx;
            const wz = dz === 0 ? (1 - tz) : tz;
            const w = wx * wz;
            sumHeight += h * w;
            sumWeight += w;
          }
        }
      }

      if (sumWeight > 0.0001) {
        return { height: sumHeight / sumWeight, loading: false };
      }
      if (anyLoading) {
        return { loading: true };
      }
    }

    const vx = Math.max(0, Math.min(tile.width - 1, Math.floor((x - cx * size) / (size / tile.width))));
    const vz = Math.max(0, Math.min(tile.depth - 1, Math.floor((z - cz * size) / (size / tile.depth))));
    const index = vx + vz * tile.width;
    const height = this.resolveSupportHeight(tile, index);

    if (height === undefined || !Number.isFinite(height)) {
      return { loading: this.isSupportColumnLoading(cx, cz) };
    }
    return { height, loading: false };
  }

  private isSupportColumnLoading(cx: number, cz: number): boolean {
    if (!this.shouldGenerateSupportTiles()) {
      return false;
    }
    const key = getSupportTileKey(cx, cz);
    return this.pendingSupportTiles.has(key) || this.loadingSupportTiles.has(key);
  }

  private resolveSupportHeight(field: HeightField, index: number): number | undefined {
    const fallbackHeight = Number.isFinite(field.fallbackHeight) ? field.fallbackHeight : undefined;
    const canUseFallback =
      fallbackHeight !== undefined && (this.supportFillMode === "fallback" || this.supportFillMode === "min");

    if (field.state[index] === 0) {
      return canUseFallback ? fallbackHeight : undefined;
    }

    const realHeight = field.height[index];
    if (!Number.isFinite(realHeight)) {
      return canUseFallback ? fallbackHeight : undefined;
    }

    return realHeight;
  }

  private getSurfaceHeightInChunk(chunk: VoxelGrid, x: number, z: number, maxY: number): number | undefined {
    if (
      x < chunk.boundsMin[0] ||
      x >= chunk.boundsMax[0] ||
      z < chunk.boundsMin[2] ||
      z >= chunk.boundsMax[2] ||
      chunk.boundsMin[1] > maxY
    ) {
      return undefined;
    }

    const field = chunk.supportHeight;
    const vx = Math.max(0, Math.min(field.width - 1, Math.floor((x - chunk.boundsMin[0]) / chunk.voxelSize)));
    const vz = Math.max(0, Math.min(field.depth - 1, Math.floor((z - chunk.boundsMin[2]) / chunk.voxelSize)));
    const fieldIndex = vx + vz * field.width;
    const height = this.resolveSupportHeight(field, fieldIndex);

    if (height === undefined || !Number.isFinite(height) || height > maxY) {
      return undefined;
    }
    return height;
  }

  public get loadedChunksCount(): number {
    return this.loadedChunks.size;
  }

  public getVoxelRevision(): number {
    return this.voxelRevision;
  }

  public getSupportRevision(): number {
    return this.supportRevision;
  }

  public getStatus(): ChunkVoxelManagerStatus {
    return {
      generationMode: this.generationMode,
      activeJobs: this.activeJobs,
      pendingVoxelChunks: this.pendingChunks.size,
      pendingSupportTiles: this.pendingSupportTiles.size,
      loadingVoxelChunks: this.loadingQueue.size,
      loadingSupportTiles: this.loadingSupportTiles.size,
      loadedVoxelChunks: this.loadedChunks.size,
      loadedSupportTiles: this.supportHeightTiles.size,
      activeVoxelChunks: this.activeKeys.size,
      activeSupportTiles: this.activeSupportKeys.size,
      voxelRevision: this.voxelRevision,
      supportRevision: this.supportRevision,
    };
  }

  public getLoadedChunks(): readonly [string, VoxelGrid][] {
    return [...this.loadedChunks.entries()];
  }

  public getLoadedSupportHeightTiles(): readonly [string, HeightField][] {
    return [...this.supportHeightTiles.entries()];
  }

  private bumpVoxelRevision(): void {
    this.voxelRevision += 1;
  }

  private bumpSupportRevision(): void {
    this.supportRevision += 1;
  }
}

function getSupportTileKey(cx: number, cz: number): string {
  return `${cx},${cz}`;
}

async function yieldToMainThread(): Promise<void> {
  await new Promise<void>((resolve) => {
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(() => resolve());
      return;
    }
    setTimeout(resolve, 0);
  });
}

function createEmptyHeightField(width: number, depth: number): HeightField {
  const height = new Float32Array(width * depth);
  height.fill(Number.NaN);
  return {
    height,
    state: new Uint8Array(width * depth),
    width,
    depth,
    fallbackHeight: Number.NaN,
  };
}

function buildSupportHeightField(
  density: Uint32Array,
  dims: [number, number, number],
  boundsMin: [number, number, number],
  voxelSize: number,
  densityThreshold: number,
  fillIterations: number,
  fillMaxHeightRange: number,
): HeightField {
  const [dimX, dimY, dimZ] = dims;
  const field = createEmptyHeightField(dimX, dimZ);

  for (let z = 0; z < dimZ; z += 1) {
    for (let x = 0; x < dimX; x += 1) {
      const columnIndex = x + z * dimX;
      for (let y = dimY - 1; y >= 0; y -= 1) {
        const densityIndex = x + y * dimX + z * dimX * dimY;
        if (density[densityIndex] >= densityThreshold) {
          field.height[columnIndex] = boundsMin[1] + (y + 1) * voxelSize;
          field.state[columnIndex] = 1;
          break;
        }
      }
    }
  }

  fillHeightFieldHoles(field, Math.max(0, Math.floor(fillIterations)), fillMaxHeightRange);
  field.fallbackHeight = getHeightFieldFallback(field);
  return field;
}

function getHeightFieldFallback(field: HeightField): number {
  const levels = buildHeightMipmap(field, getFullMipmapLevelCount(field.width, field.depth));
  const finalLevel = levels[levels.length - 1];
  if (!finalLevel) {
    return Number.NaN;
  }

  let sum = 0;
  let count = 0;
  for (let i = 0; i < finalLevel.count.length; i += 1) {
    if (finalLevel.count[i] === 0) {
      continue;
    }
    sum += finalLevel.sum[i];
    count += finalLevel.count[i];
  }
  return count > 0 ? sum / count : Number.NaN;
}

function fillHeightFieldHoles(field: HeightField, iterations: number, maxHeightRange: number): void {
  const maxLevels = Math.max(0, Math.floor(iterations));
  if (maxLevels === 0) {
    return;
  }

  fillHeightFieldHolesFromFourNeighbors(field, maxHeightRange);
  if (maxLevels === 1) {
    return;
  }

  const mipmap = buildHeightMipmap(field, maxLevels);
  if (mipmap.length <= 1) {
    return;
  }

  const { width, depth } = field;
  for (let z = 0; z < depth; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + z * width;
      if (field.state[index] !== 0) {
        continue;
      }

      const height = sampleHeightMipmap(mipmap, x, z, maxHeightRange);
      if (height === undefined) {
        continue;
      }

      field.height[index] = height;
      field.state[index] = 2;
    }
  }
}

function fillHeightFieldHolesFromFourNeighbors(field: HeightField, maxHeightRange: number): void {
  const { width, depth } = field;
  const nextHeight = new Float32Array(field.height.length);
  const nextState = new Uint8Array(field.state.length);
  nextHeight.set(field.height);
  nextState.set(field.state);

  for (let z = 0; z < depth; z += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = x + z * width;
      if (field.state[index] !== 0) {
        continue;
      }

      const height = sampleFourNeighborHeight(field, x, z, maxHeightRange);
      if (height === undefined) {
        continue;
      }

      nextHeight[index] = height;
      nextState[index] = 2;
    }
  }

  field.height.set(nextHeight);
  field.state.set(nextState);
}

function sampleFourNeighborHeight(field: HeightField, x: number, z: number, maxHeightRange: number): number | undefined {
  const offsets: readonly [number, number][] = [
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ];
  let sum = 0;
  let count = 0;
  let minHeight = Number.POSITIVE_INFINITY;
  let maxHeight = Number.NEGATIVE_INFINITY;

  for (const [dx, dz] of offsets) {
    const nx = x + dx;
    const nz = z + dz;
    if (nx < 0 || nx >= field.width || nz < 0 || nz >= field.depth) {
      continue;
    }

    const index = nx + nz * field.width;
    if (field.state[index] === 0) {
      continue;
    }

    const height = field.height[index];
    if (!Number.isFinite(height)) {
      continue;
    }

    sum += height;
    count += 1;
    minHeight = Math.min(minHeight, height);
    maxHeight = Math.max(maxHeight, height);
  }

  if (count === 0 || maxHeight - minHeight > maxHeightRange) {
    return undefined;
  }
  return sum / count;
}

function buildHeightMipmap(field: HeightField, maxLevels: number): HeightMipmapLevel[] {
  const levels: HeightMipmapLevel[] = [createBaseHeightMipmapLevel(field)];

  for (let levelIndex = 1; levelIndex < maxLevels; levelIndex += 1) {
    const previous = levels[levelIndex - 1];
    if (previous.width === 1 && previous.depth === 1) {
      break;
    }

    const width = Math.max(1, Math.ceil(previous.width / 2));
    const depth = Math.max(1, Math.ceil(previous.depth / 2));
    const next = createEmptyHeightMipmapLevel(width, depth);

    for (let z = 0; z < depth; z += 1) {
      for (let x = 0; x < width; x += 1) {
        const targetIndex = x + z * width;
        for (let dz = 0; dz < 2; dz += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const px = x * 2 + dx;
            const pz = z * 2 + dz;
            if (px >= previous.width || pz >= previous.depth) {
              continue;
            }

            const sourceIndex = px + pz * previous.width;
            const count = previous.count[sourceIndex];
            if (count === 0) {
              continue;
            }

            next.sum[targetIndex] += previous.sum[sourceIndex];
            next.count[targetIndex] += count;
            next.min[targetIndex] = Math.min(next.min[targetIndex], previous.min[sourceIndex]);
            next.max[targetIndex] = Math.max(next.max[targetIndex], previous.max[sourceIndex]);
          }
        }
      }
    }

    levels.push(next);
  }

  return levels;
}

function getFullMipmapLevelCount(width: number, depth: number): number {
  return Math.ceil(Math.log2(Math.max(1, width, depth))) + 1;
}

function createBaseHeightMipmapLevel(field: HeightField): HeightMipmapLevel {
  const level = createEmptyHeightMipmapLevel(field.width, field.depth);
  for (let i = 0; i < field.height.length; i += 1) {
    if (field.state[i] === 0) {
      continue;
    }
    const height = field.height[i];
    if (!Number.isFinite(height)) {
      continue;
    }
    level.sum[i] = height;
    level.count[i] = 1;
    level.min[i] = height;
    level.max[i] = height;
  }
  return level;
}

function createEmptyHeightMipmapLevel(width: number, depth: number): HeightMipmapLevel {
  const length = width * depth;
  const min = new Float32Array(length);
  const max = new Float32Array(length);
  min.fill(Number.POSITIVE_INFINITY);
  max.fill(Number.NEGATIVE_INFINITY);
  return {
    width,
    depth,
    sum: new Float32Array(length),
    count: new Uint32Array(length),
    min,
    max,
  };
}

function sampleHeightMipmap(
  levels: readonly HeightMipmapLevel[],
  x: number,
  z: number,
  maxHeightRange: number,
): number | undefined {
  for (let levelIndex = 1; levelIndex < levels.length; levelIndex += 1) {
    const level = levels[levelIndex];
    const scale = 2 ** levelIndex;
    const coarseX = Math.floor(x / scale);
    const coarseZ = Math.floor(z / scale);
    const localX = x - coarseX * scale;
    const localZ = z - coarseZ * scale;
    const startX = getMipmapWindowStart(coarseX, localX, scale, level.width);
    const startZ = getMipmapWindowStart(coarseZ, localZ, scale, level.depth);

    let sum = 0;
    let count = 0;
    let minHeight = Number.POSITIVE_INFINITY;
    let maxHeight = Number.NEGATIVE_INFINITY;

    for (let dz = 0; dz < Math.min(2, level.depth); dz += 1) {
      for (let dx = 0; dx < Math.min(2, level.width); dx += 1) {
        const sampleIndex = startX + dx + (startZ + dz) * level.width;
        const sampleCount = level.count[sampleIndex];
        if (sampleCount === 0) {
          continue;
        }

        sum += level.sum[sampleIndex];
        count += sampleCount;
        minHeight = Math.min(minHeight, level.min[sampleIndex]);
        maxHeight = Math.max(maxHeight, level.max[sampleIndex]);
      }
    }

    if (count === 0 || maxHeight - minHeight > maxHeightRange) {
      continue;
    }

    return sum / count;
  }

  return undefined;
}

function getMipmapWindowStart(coarse: number, local: number, scale: number, size: number): number {
  if (size <= 1) {
    return 0;
  }
  const start = local < scale / 2 ? coarse - 1 : coarse;
  return Math.max(0, Math.min(size - 2, start));
}
