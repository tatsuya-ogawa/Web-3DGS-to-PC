import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import {
  packSplats,
  SplatSpatialBinning,
  ChunkVoxelManager,
  type ChunkGenerationMode,
  type ChunkVoxelManagerStatus,
  type ChunkVoxelGrid as VoxelGrid,
  type HeightField,
} from "web-3dgs-to-pc/browser";
import { TpsController, TPS_CHARACTER_SCALE } from "./tpsController.js";

const sceneSelect = document.getElementById("scene-select") as HTMLSelectElement;
const sceneFileInput = document.getElementById("scene-file-input") as HTMLInputElement;
const splatCheckbox = document.getElementById("splat-checkbox") as HTMLInputElement;
const voxelCheckbox = document.getElementById("voxel-checkbox") as HTMLInputElement;
const hudChunk = document.getElementById("hud-chunk") as HTMLElement;
const hudXyz = document.getElementById("hud-xyz") as HTMLElement;
const hudActiveChunks = document.getElementById("hud-active-chunks") as HTMLElement;
const hudFps = document.getElementById("hud-fps") as HTMLElement;
const resetPlayerBtn = document.getElementById("reset-player-btn") as HTMLButtonElement;
const overlay = document.getElementById("overlay") as HTMLDivElement;
const canvasHost = document.getElementById("canvas-host") as HTMLDivElement;
const heightfieldCheckbox = document.getElementById("heightfield-checkbox") as HTMLInputElement;
const terrainModeSelect = document.getElementById("terrain-mode-select") as HTMLSelectElement;
const smoothingCheckbox = document.getElementById("smoothing-checkbox") as HTMLInputElement;
const heightfieldStyleSelect = document.getElementById("heightfield-style-select") as HTMLSelectElement;

let splatMeshObject: SplatMesh | undefined;
let sparkRenderer: SparkRenderer | undefined;
let chunkManager: ChunkVoxelManager | undefined;
let controller: TpsController | undefined;
let playerMesh: THREE.Group | undefined;
let currentPackedSplats: any | undefined;
let binning: SplatSpatialBinning | undefined;
let showSplat = true;
let showVoxel = false;
let chunkGenerationMode: ChunkGenerationMode = parseChunkGenerationMode(terrainModeSelect.value);
let lastTerrainSyncTime = 0;
let lastHudUpdateTime = 0;
let lastFrameTime = 0;
let lastVoxelMeshRevision = -1;
let lastHeightfieldMeshRevision = -1;
let voxelMeshBuildPending = false;

let showHeightfield = false;
const heightfieldGroup = new THREE.Group();
const heightfieldMeshes = new Map<string, THREE.Mesh>();

const voxelGroup = new THREE.Group();
const voxelMeshes = new Map<string, THREE.InstancedMesh>();
const voxelMeshEmptyKeys = new Set<string>();
const BASE_CHUNK_SIZE = 16.0;
const BASE_CHUNK_RESOLUTION = 64;
const CHUNK_SIZE = BASE_CHUNK_SIZE * TPS_CHARACTER_SCALE;
const TPS_CHUNK_RESOLUTION = BASE_CHUNK_RESOLUTION;
const VOXEL_DISPLAY_THRESHOLD = 120;
const MAX_VOXELS_PER_CHUNK = 12000;
const MAX_VOXEL_MESH_BUILDS_PER_SYNC = 1;
const MAX_VOXEL_MESH_ATTEMPTS_PER_SYNC = 2;
const ACTIVE_FRAME_INTERVAL_MS = 1000 / 60;
const IDLE_FRAME_INTERVAL_MS = 1000 / 24;
const TERRAIN_SYNC_INTERVAL_MS = 500;
const BUSY_TERRAIN_SYNC_INTERVAL_MS = 180;
const HUD_UPDATE_INTERVAL_MS = 160;
const SPAWN_SEARCH_CHUNK_LIMIT = 8;
const SPAWN_PLAYER_RADIUS = 0.45 * TPS_CHARACTER_SCALE;
const SPAWN_PLAYER_HEIGHT = 1.8 * TPS_CHARACTER_SCALE;
const SPAWN_MIN_SUPPORT_RATIO = 0.82;
const SPAWN_MAX_HEIGHT_RANGE = 0.55 * TPS_CHARACTER_SCALE;
const SPAWN_SURFACE_OFFSET = 0.08 * TPS_CHARACTER_SCALE;

interface SpawnCandidate {
  position: THREE.Vector3;
  score: number;
  supportRatio: number;
  heightRange: number;
  tile: [number, number];
}

interface SceneSource {
  fileBytes: Uint8Array;
  fileName: string;
}

interface BundledSource {
  fileName: string;
  label: string;
  url: string;
}

const bundledSourceUrls = import.meta.glob<string>("../splats/*.{sog,spz,splat,ksplat,ply}", {
  eager: true,
  import: "default",
  query: "?url",
});
const bundledSources: BundledSource[] = Object.entries(bundledSourceUrls)
  .map(([path, url]) => {
    const fileName = path.split("/").pop() ?? "scene.sog";
    return {
      fileName,
      label: formatSceneLabel(fileName),
      url,
    };
  })
  .sort((a, b) => a.label.localeCompare(b.label));

// Three.js setup
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
canvasHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.add(voxelGroup);
scene.add(heightfieldGroup);
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);

// Sky hemispheric light and directional light
scene.add(new THREE.HemisphereLight(0xffffff, 0x66736f, 1.35));
const dirLight = new THREE.DirectionalLight(0xffffff, 2.4);
dirLight.position.set(3, 10, 4);
scene.add(dirLight);

// Setup Spark splat renderer
sparkRenderer = new SparkRenderer({ renderer });
scene.add(sparkRenderer);

// Resize handling
function resize(): void {
  const rect = canvasHost.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, true);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
resize();
window.addEventListener("resize", resize);

// PointerLock controls
overlay.addEventListener("click", () => {
  overlay.classList.add("hidden");
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    overlay.classList.remove("hidden");
  }
});

// Setup persistent WebGPU device
let gpuDevice: GPUDevice | undefined;
async function initWebGpu(): Promise<GPUDevice> {
  if (gpuDevice) return gpuDevice;
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported by your browser");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No WebGPU adapter found");
  gpuDevice = await adapter.requestDevice();
  return gpuDevice;
}

function parseChunkGenerationMode(value: string): ChunkGenerationMode {
  if (value === "voxels" || value === "both") {
    return value;
  }
  return "heightfield";
}

function getChunkGenerationMode(): ChunkGenerationMode {
  return chunkGenerationMode;
}

function usesVoxelChunks(mode: ChunkGenerationMode): boolean {
  return mode === "voxels" || mode === "both";
}

function usesHeightfieldTiles(mode: ChunkGenerationMode): boolean {
  return mode === "heightfield" || mode === "both";
}

function applyTerrainModeControls(): void {
  const mode = getChunkGenerationMode();
  const canShowVoxels = usesVoxelChunks(mode);
  const canShowHeightfield = usesHeightfieldTiles(mode);

  voxelCheckbox.disabled = !canShowVoxels;
  heightfieldCheckbox.disabled = !canShowHeightfield;

  if (!canShowVoxels) {
    showVoxel = false;
    voxelCheckbox.checked = false;
    clearVoxelMeshes();
  } else {
    showVoxel = voxelCheckbox.checked;
  }

  if (!canShowHeightfield) {
    showHeightfield = false;
    heightfieldCheckbox.checked = false;
    clearHeightfieldMeshes();
  } else {
    showHeightfield = heightfieldCheckbox.checked;
  }

  applyRenderMode();
}

// Scene loading & Initialization
async function loadScene(source: SceneSource): Promise<void> {
  console.log(`Loading scene: ${source.fileName}`);
  
  // 1. Reset existing states
  if (splatMeshObject) {
    scene.remove(splatMeshObject);
    splatMeshObject.dispose();
    splatMeshObject = undefined;
  }
  if (playerMesh) {
    scene.remove(playerMesh);
    playerMesh = undefined;
  }
  controller = undefined;
  chunkManager = undefined;
  clearVoxelMeshes();
  clearHeightfieldMeshes();
  lastVoxelMeshRevision = -1;
  lastHeightfieldMeshRevision = -1;

  // 2. Decode local splat bytes
  splatMeshObject = new SplatMesh({
    fileBytes: source.fileBytes,
    fileName: source.fileName,
    extSplats: true,
  });
  await splatMeshObject.initialized;

  // Rotate to standard world alignment (flip Y to make it floor-up)
  splatMeshObject.rotation.x = Math.PI;
  scene.add(splatMeshObject);

  // 3. Initialize dynamic voxel chunk manager
  const device = await initWebGpu();
  const packed = packSplats(splatMeshObject, splatMeshObject.numSplats, 0.01);
  
  // Align packed splats matching the world-space rotation (rotate centers y-z)
  const centers = packed.centers;
  for (let i = 0; i < packed.count; i += 1) {
    const base = i * 4;
    // Rotate centers by Math.PI (180 deg) around X axis: y -> -y, z -> -z
    centers[base + 1] = -centers[base + 1];
    centers[base + 2] = -centers[base + 2];
  }

  binning = new SplatSpatialBinning(packed, CHUNK_SIZE, 2.0);
  chunkManager = new ChunkVoxelManager(device, binning, TPS_CHUNK_RESOLUTION, {
    maxConcurrentJobs: 2,
    smoothIterations: 0,
    generationMode: getChunkGenerationMode(),
    supportFillMode: "fallback",
    activeHorizontalRadius: 2,
    activeVerticalBelow: 1,
    activeVerticalAbove: 1,
  });
  if (chunkManager) {
    chunkManager.enableHeightfieldSmoothing = smoothingCheckbox ? smoothingCheckbox.checked : true;
  }
  currentPackedSplats = packed;

  // 4. Find a supported voxel floor in dense chunks before spawning
  const spawnPos = await findSupportedSpawnPosition(chunkManager, binning);

  // 5. Initialize player capsule controller
  controller = new TpsController(camera, canvasHost, chunkManager, spawnPos);

  // 6. Create visual wireframe player capsule
  playerMesh = new THREE.Group();
  
  // Capsule body wireframe
  const geom = new THREE.CapsuleGeometry(controller.radius, controller.height - controller.radius * 2, 8, 12);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x2f6f66,
    wireframe: true,
    transparent: true,
    opacity: 0.8,
  });
  const body = new THREE.Mesh(geom, mat);
  body.position.y = controller.height / 2; // Offset geometry to center
  playerMesh.add(body);

  // Forward pointer direction helper
  const pointerGeom = new THREE.ConeGeometry(0.12 * TPS_CHARACTER_SCALE, 0.35 * TPS_CHARACTER_SCALE, 8);
  const pointerMat = new THREE.MeshBasicMaterial({ color: 0xff3b30 });
  const pointer = new THREE.Mesh(pointerGeom, pointerMat);
  pointer.rotation.x = Math.PI / 2; // Point forward
  pointer.position.set(
    0,
    controller.height - 0.3 * TPS_CHARACTER_SCALE,
    -controller.radius - 0.1 * TPS_CHARACTER_SCALE,
  );
  playerMesh.add(pointer);

  scene.add(playerMesh);
  applyTerrainModeControls();
  applyRenderMode();

  console.log("Scene loaded successfully!");
}

populateBundledSourceSelect();

sceneSelect.addEventListener("change", () => {
  const source = bundledSources.find((item) => item.url === sceneSelect.value);
  if (!source) {
    return;
  }

  void fetch(source.url)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load ${source.fileName}: ${response.status}`);
      }
      return response.arrayBuffer();
    })
    .then((buffer) => loadScene({
      fileBytes: new Uint8Array(buffer),
      fileName: source.fileName,
    }))
    .catch(console.error);
});

sceneFileInput.addEventListener("change", () => {
  const file = sceneFileInput.files?.[0];
  if (!file) {
    return;
  }

  void file.arrayBuffer()
    .then((buffer) => loadScene({
      fileBytes: new Uint8Array(buffer),
      fileName: file.name,
    }))
    .catch(console.error);
});

splatCheckbox.addEventListener("change", () => {
  showSplat = splatCheckbox.checked;
  applyRenderMode();
});

voxelCheckbox.addEventListener("change", () => {
  showVoxel = voxelCheckbox.checked;
  applyRenderMode();
  if (showVoxel) {
    syncVoxelMeshes();
  } else {
    clearVoxelMeshes();
    lastVoxelMeshRevision = -1;
  }
});

heightfieldCheckbox.addEventListener("change", () => {
  showHeightfield = heightfieldCheckbox.checked;
  syncHeightfieldMeshes(true);
});

smoothingCheckbox.addEventListener("change", () => {
  if (chunkManager) {
    chunkManager.enableHeightfieldSmoothing = smoothingCheckbox.checked;
  }
});

heightfieldStyleSelect.addEventListener("change", () => {
  if (showHeightfield) {
    syncHeightfieldMeshes(true);
  }
});

terrainModeSelect.addEventListener("change", () => {
  chunkGenerationMode = parseChunkGenerationMode(terrainModeSelect.value);
  chunkManager?.setGenerationMode(chunkGenerationMode);
  applyTerrainModeControls();
  if (usesVoxelChunks(chunkGenerationMode) && showVoxel) {
    syncVoxelMeshes(true);
  }
  if (usesHeightfieldTiles(chunkGenerationMode) && showHeightfield) {
    syncHeightfieldMeshes(true);
  }
});

// Reset player button
resetPlayerBtn.addEventListener("click", () => {
  if (!controller || !chunkManager || !binning) return;

  resetPlayerBtn.disabled = true;
  void findSupportedSpawnPosition(chunkManager, binning)
    .then((spawnPos) => {
      controller?.respawn(spawnPos);
      overlay.classList.add("hidden");
    })
    .finally(() => {
      resetPlayerBtn.disabled = false;
    });
});

// Loop variables
let lastTime = performance.now();
let frameCount = 0;
let fpsLastTime = performance.now();

// Render Loop
function animate(): void {
  requestAnimationFrame(animate);

  const time = performance.now();
  const terrainStatus = chunkManager?.getStatus();
  const terrainBusy = terrainStatus ? isTerrainBusy(terrainStatus) : false;
  const userActive = controller?.isActive() ?? true;
  const targetFrameInterval = userActive || terrainBusy ? ACTIVE_FRAME_INTERVAL_MS : IDLE_FRAME_INTERVAL_MS;
  if (lastFrameTime > 0 && time - lastFrameTime < targetFrameInterval) {
    return;
  }
  lastFrameTime = time;

  const dt = (time - lastTime) / 1000.0;
  lastTime = time;

  // FPS calculation
  frameCount += 1;
  if (time - fpsLastTime >= 1000.0) {
    hudFps.textContent = frameCount.toString();
    frameCount = 0;
    fpsLastTime = time;
  }

  // Update physics and camera
  if (controller && chunkManager && playerMesh) {
    const p = controller.position;
    const stat = chunkManager.updateActiveRegion(p.x, p.y, p.z);
    const latestTerrainStatus = chunkManager.getStatus();
    const terrainSyncInterval = isTerrainBusy(latestTerrainStatus)
      ? BUSY_TERRAIN_SYNC_INTERVAL_MS
      : TERRAIN_SYNC_INTERVAL_MS;
    if (time - lastTerrainSyncTime > terrainSyncInterval) {
      if (showVoxel) {
        syncVoxelMeshes();
      }
      if (showHeightfield) {
        syncHeightfieldMeshes();
      }
      lastTerrainSyncTime = time;
    }
    
    controller.update(dt);

    playerMesh.position.copy(p);
    playerMesh.rotation.y = controller.yaw;

    if (time - lastHudUpdateTime > HUD_UPDATE_INTERVAL_MS) {
      const size = CHUNK_SIZE;
      const cx = Math.floor(p.x / size);
      const cy = Math.floor(p.y / size);
      const cz = Math.floor(p.z / size);
      hudChunk.textContent = `[${cx}, ${cy}, ${cz}]`;
      hudActiveChunks.textContent = `${stat.loaded} / ${stat.total}`;
      hudXyz.textContent = `[${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)}]`;
      lastHudUpdateTime = time;
    }
  }

  renderer.render(scene, camera);
}

function isTerrainBusy(status: ChunkVoxelManagerStatus): boolean {
  return (
    status.activeJobs > 0 ||
    status.pendingVoxelChunks > 0 ||
    status.pendingSupportTiles > 0 ||
    status.loadingVoxelChunks > 0 ||
    status.loadingSupportTiles > 0
  );
}

function applyRenderMode(): void {
  if (splatMeshObject) {
    splatMeshObject.visible = showSplat;
  }
  voxelGroup.visible = showVoxel;
}

function syncVoxelMeshes(force: boolean = false): void {
  if (!chunkManager) return;
  const revision = chunkManager.getVoxelRevision();
  if (!force && revision === lastVoxelMeshRevision && !voxelMeshBuildPending) {
    return;
  }

  const loadedEntries = chunkManager.getLoadedChunks();
  const loadedKeys = new Set(loadedEntries.map(([key]) => key));
  for (const key of [...voxelMeshes.keys()]) {
    if (!loadedKeys.has(key)) {
      const mesh = voxelMeshes.get(key)!;
      voxelGroup.remove(mesh);
      disposeInstancedMesh(mesh);
      voxelMeshes.delete(key);
    }
  }
  for (const key of [...voxelMeshEmptyKeys]) {
    if (!loadedKeys.has(key)) {
      voxelMeshEmptyKeys.delete(key);
    }
  }

  let meshesBuilt = 0;
  let attempts = 0;
  let hasMoreMeshWork = false;
  for (const [key, chunk] of loadedEntries) {
    if (voxelMeshes.has(key) || voxelMeshEmptyKeys.has(key)) continue;
    if (meshesBuilt >= MAX_VOXEL_MESH_BUILDS_PER_SYNC || attempts >= MAX_VOXEL_MESH_ATTEMPTS_PER_SYNC) {
      hasMoreMeshWork = true;
      break;
    }
    attempts += 1;
    const mesh = createVoxelMeshForChunk(chunk);
    if (!mesh) {
      voxelMeshEmptyKeys.add(key);
      continue;
    }
    voxelMeshes.set(key, mesh);
    voxelGroup.add(mesh);
    meshesBuilt += 1;
  }
  lastVoxelMeshRevision = revision;
  voxelMeshBuildPending = hasMoreMeshWork;
}

function createVoxelMeshForChunk(chunk: any): THREE.InstancedMesh | undefined {
  const occupiedIndices: number[] = [];
  for (let i = 0; i < chunk.density.length; i += 1) {
    if (chunk.density[i] >= VOXEL_DISPLAY_THRESHOLD) {
      occupiedIndices.push(i);
    }
  }
  if (occupiedIndices.length === 0) {
    return undefined;
  }

  const stride = Math.max(1, Math.ceil(occupiedIndices.length / MAX_VOXELS_PER_CHUNK));
  const instanceCount = Math.ceil(occupiedIndices.length / stride);
  const geometry = new THREE.BoxGeometry(
    chunk.voxelSize * 0.92,
    chunk.voxelSize * 0.92,
    chunk.voxelSize * 0.92,
  );
  const material = new THREE.MeshBasicMaterial({
    color: 0x00b8d4,
    transparent: true,
    opacity: 0.46,
    depthWrite: false,
  });
  const mesh = new THREE.InstancedMesh(geometry, material, instanceCount);
  mesh.frustumCulled = false;

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();
  const [dimX, dimY] = chunk.dims;
  let out = 0;
  for (let i = 0; i < occupiedIndices.length; i += stride) {
    const index = occupiedIndices[i];
    const z = Math.floor(index / (dimX * dimY));
    const rest = index - z * dimX * dimY;
    const y = Math.floor(rest / dimX);
    const x = rest - y * dimX;
    dummy.position.set(
      chunk.boundsMin[0] + (x + 0.5) * chunk.voxelSize,
      chunk.boundsMin[1] + (y + 0.5) * chunk.voxelSize,
      chunk.boundsMin[2] + (z + 0.5) * chunk.voxelSize,
    );
    dummy.updateMatrix();
    mesh.setMatrixAt(out, dummy.matrix);
    const intensity = Math.min(1, Math.log2(chunk.density[index] + 1) / 10);
    color.setRGB(0.05 + intensity * 0.25, 0.55 + intensity * 0.35, 0.75 + intensity * 0.2);
    mesh.setColorAt(out, color);
    out += 1;
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) {
    mesh.instanceColor.needsUpdate = true;
  }
  return mesh;
}

function clearVoxelMeshes(): void {
  for (const mesh of voxelMeshes.values()) {
    voxelGroup.remove(mesh);
    disposeInstancedMesh(mesh);
  }
  voxelMeshes.clear();
  voxelMeshEmptyKeys.clear();
  voxelMeshBuildPending = false;
  lastVoxelMeshRevision = -1;
}

function clearHeightfieldMeshes(): void {
  for (const mesh of heightfieldMeshes.values()) {
    heightfieldGroup.remove(mesh);
    disposeMesh(mesh);
  }
  heightfieldMeshes.clear();
  lastHeightfieldMeshRevision = -1;
}

function syncHeightfieldMeshes(force: boolean = false): void {
  if (!chunkManager) return;
  heightfieldGroup.visible = showHeightfield;
  if (!showHeightfield) {
    return;
  }

  const revision = chunkManager.getSupportRevision();
  if (!force && revision === lastHeightfieldMeshRevision) {
    return;
  }

  const loadedEntries = chunkManager.getLoadedSupportHeightTiles();
  const loadedKeys = new Set(loadedEntries.map(([key]) => key));

  // 1. Remove stale meshes
  for (const key of [...heightfieldMeshes.keys()]) {
    if (!loadedKeys.has(key)) {
      const mesh = heightfieldMeshes.get(key)!;
      heightfieldGroup.remove(mesh);
      disposeMesh(mesh);
      heightfieldMeshes.delete(key);
    }
  }

  // 2. Build missing meshes
  for (const [key, field] of loadedEntries) {
    if (heightfieldMeshes.has(key)) continue;
    const mesh = createHeightfieldMeshForTile(key, field);
    if (!mesh) continue;
    heightfieldMeshes.set(key, mesh);
    heightfieldGroup.add(mesh);
  }
  lastHeightfieldMeshRevision = revision;
}

function createHeightfieldMeshForTile(key: string, field: HeightField): THREE.Mesh | undefined {
  if (!field || !field.height) return undefined;

  const [cx, cz] = key.split(",").map(Number);
  const dimX = field.width;
  const dimZ = field.depth;
  const totalCells = dimX * dimZ;

  const validIndices: number[] = [];
  for (let i = 0; i < totalCells; i += 1) {
    if (field.state[i] !== 0 && Number.isFinite(field.height[i])) {
      validIndices.push(i);
    }
  }

  if (validIndices.length === 0) return undefined;

  const voxelSize = CHUNK_SIZE / dimX;
  const style = heightfieldStyleSelect ? heightfieldStyleSelect.value : "smooth-grid";

  if (style === "holo-blocks") {
    // Each chunk column represented as a beautiful thin translucent glowing tile
    const geometry = new THREE.BoxGeometry(
      voxelSize * 0.95,
      0.02 * TPS_CHARACTER_SCALE,
      voxelSize * 0.95
    );

    // MeshBasicMaterial with additive transparency for a glowing holo-grid look
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
    });

    const mesh = new THREE.InstancedMesh(geometry, material, validIndices.length);
    mesh.frustumCulled = false;

    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    for (let idx = 0; idx < validIndices.length; idx += 1) {
      const cellIdx = validIndices[idx];
      const z = Math.floor(cellIdx / dimX);
      const x = cellIdx - z * dimX;

      const height = field.height[cellIdx];
      const state = field.state[cellIdx];

      const worldX = cx * CHUNK_SIZE + (x + 0.5) * voxelSize;
      const worldZ = cz * CHUNK_SIZE + (z + 0.5) * voxelSize;

      dummy.position.set(worldX, height, worldZ);
      dummy.updateMatrix();
      mesh.setMatrixAt(idx, dummy.matrix);

      // Color code: state 1 (direct voxel check) -> bright cyan-green, state 2 (interpolated) -> purple/pink
      if (state === 1) {
        color.setHSL(0.48, 1.0, 0.5); // Cyan-green
      } else {
        color.setHSL(0.78, 1.0, 0.6); // Indigo-purple
      }
      mesh.setColorAt(idx, color);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }

    return mesh;
  } else {
    // smooth-grid style: continuous glowing wireframe terrain mesh
    const vertices: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];

    const gridIndices = new Int32Array(dimX * dimZ);
    gridIndices.fill(-1);

    let vertexCount = 0;
    for (let z = 0; z < dimZ; z += 1) {
      for (let x = 0; x < dimX; x += 1) {
        const idx = x + z * dimX;
        const height = field.height[idx];
        const state = field.state[idx];
        if (state !== 0 && Number.isFinite(height)) {
          const worldX = cx * CHUNK_SIZE + (x + 0.5) * voxelSize;
          const worldZ = cz * CHUNK_SIZE + (z + 0.5) * voxelSize;
          
          vertices.push(worldX, height, worldZ);
          
          const color = new THREE.Color();
          if (state === 1) {
            color.setHSL(0.48, 1.0, 0.5); // Cyan-green
          } else {
            color.setHSL(0.78, 1.0, 0.6); // Indigo-purple
          }
          colors.push(color.r, color.g, color.b);
          
          gridIndices[idx] = vertexCount;
          vertexCount += 1;
        }
      }
    }

    // Generate triangles
    for (let z = 0; z < dimZ - 1; z += 1) {
      for (let x = 0; x < dimX - 1; x += 1) {
        const i00 = gridIndices[x + z * dimX];
        const i10 = gridIndices[(x + 1) + z * dimX];
        const i01 = gridIndices[x + (z + 1) * dimX];
        const i11 = gridIndices[(x + 1) + (z + 1) * dimX];

        if (i00 !== -1 && i10 !== -1 && i01 !== -1) {
          indices.push(i00, i01, i10);
        }
        if (i10 !== -1 && i11 !== -1 && i01 !== -1) {
          indices.push(i10, i01, i11);
        }
      }
    }

    if (vertices.length === 0 || indices.length === 0) return undefined;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
      wireframe: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.frustumCulled = false;
    return mesh;
  }
}

function disposeMesh(mesh: THREE.Mesh): void {
  mesh.geometry.dispose();
  if (Array.isArray(mesh.material)) {
    mesh.material.forEach((material) => material.dispose());
  } else {
    mesh.material.dispose();
  }
}

async function findSupportedSpawnPosition(
  chunkManager: ChunkVoxelManager,
  binning: SplatSpatialBinning,
): Promise<THREE.Vector3> {
  const tileCoords = binning.getColumnCoordinatesByDensity(SPAWN_SEARCH_CHUNK_LIMIT);
  let best: SpawnCandidate | undefined;

  for (const [cx, cz] of tileCoords) {
    const field = await chunkManager.loadSupportHeightTileNow(cx, cz);
    const candidate = findBestSpawnInTile(field, [cx, cz]);
    if (!candidate) continue;
    if (!best || candidate.score > best.score) {
      best = candidate;
    }
  }

  if (best) {
    console.log(
      `Spawn selected heightfield=${best.tile.join(",")} support=${best.supportRatio.toFixed(2)} range=${best.heightRange.toFixed(2)} score=${best.score.toFixed(1)}`,
    );
    return best.position;
  }

  console.warn("No supported voxel spawn found; falling back to densest splat position");
  return findDensestSplatSpawnPosition(binning);
}

function findBestSpawnInTile(
  field: HeightField,
  tileCoords: [number, number],
): SpawnCandidate | undefined {
  const dimX = field.width;
  const dimZ = field.depth;
  const voxelSize = CHUNK_SIZE / dimX;
  const radiusVoxels = Math.max(1, Math.ceil(SPAWN_PLAYER_RADIUS / voxelSize));
  const radiusSq = (SPAWN_PLAYER_RADIUS / voxelSize) ** 2;
  let best: SpawnCandidate | undefined;

  for (let z = radiusVoxels; z < dimZ - radiusVoxels; z += 1) {
    for (let x = radiusVoxels; x < dimX - radiusVoxels; x += 1) {
      const centerColumn = x + z * dimX;
      const centerHeight = field.height[centerColumn];
      if (field.state[centerColumn] === 0 || !Number.isFinite(centerHeight)) continue;

      let totalSamples = 0;
      let supportSamples = 0;
      let originalSamples = 0;
      let minSupportHeight = Number.POSITIVE_INFINITY;
      let maxSupportHeight = Number.NEGATIVE_INFINITY;

      for (let dz = -radiusVoxels; dz <= radiusVoxels; dz += 1) {
        for (let dx = -radiusVoxels; dx <= radiusVoxels; dx += 1) {
          if (dx * dx + dz * dz > radiusSq) continue;
          totalSamples += 1;

          const sampleColumn = x + dx + (z + dz) * dimX;
          const sampleHeight = field.height[sampleColumn];
          if (
            field.state[sampleColumn] === 0 ||
            !Number.isFinite(sampleHeight) ||
            Math.abs(sampleHeight - centerHeight) > SPAWN_MAX_HEIGHT_RANGE
          ) {
            continue;
          }

          supportSamples += 1;
          if (field.state[sampleColumn] === 1) {
            originalSamples += 1;
          }
          minSupportHeight = Math.min(minSupportHeight, sampleHeight);
          maxSupportHeight = Math.max(maxSupportHeight, sampleHeight);
        }
      }

      const supportRatio = supportSamples / totalSamples;
      if (supportRatio < SPAWN_MIN_SUPPORT_RATIO) continue;

      const heightRange = maxSupportHeight - minSupportHeight;
      const originalRatio = originalSamples / Math.max(1, supportSamples);
      const edgeDistance = Math.min(x, z, dimX - 1 - x, dimZ - 1 - z);
      const score =
        supportRatio * 1000 +
        originalRatio * 180 -
        heightRange * 300 +
        edgeDistance * 0.5;

      if (best && score <= best.score) continue;

      best = {
        position: new THREE.Vector3(
          tileCoords[0] * CHUNK_SIZE + (x + 0.5) * voxelSize,
          centerHeight + SPAWN_SURFACE_OFFSET,
          tileCoords[1] * CHUNK_SIZE + (z + 0.5) * voxelSize,
        ),
        score,
        supportRatio,
        heightRange,
        tile: tileCoords,
      };
    }
  }

  return best;
}

function findDensestSplatSpawnPosition(binning: SplatSpatialBinning): THREE.Vector3 {
  const [dcx, dcy, dcz] = binning.getDensestChunkCoordinates();
  const subset = binning.getSplatSubsetForChunk(dcx, dcy, dcz)!;
  let highestSplatY = Number.NEGATIVE_INFINITY;
  let spawnX = (dcx + 0.5) * binning.chunkSizeVal;
  let spawnZ = (dcz + 0.5) * binning.chunkSizeVal;
  
  const subsetCenters = subset.centers;
  const subsetCount = subset.count;
  for (let i = 0; i < subsetCount; i += 1) {
    const base = i * 4;
    const sx = subsetCenters[base];
    const sy = subsetCenters[base + 1];
    const sz = subsetCenters[base + 2];
    if (sy > highestSplatY) {
      highestSplatY = sy;
      spawnX = sx;
      spawnZ = sz;
    }
  }

  // Spawn player slightly above the highest floor in the densest chunk
  return new THREE.Vector3(spawnX, highestSplatY + 0.8 * TPS_CHARACTER_SCALE, spawnZ);
}

// Startup
if (sceneSelect.value) {
  sceneSelect.dispatchEvent(new Event("change"));
}
animate();

function populateBundledSourceSelect(): void {
  sceneSelect.innerHTML = "";
  if (bundledSources.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No bundled scenes";
    sceneSelect.appendChild(option);
    sceneSelect.disabled = true;
    return;
  }

  for (const source of bundledSources) {
    const option = document.createElement("option");
    option.value = source.url;
    option.textContent = source.label;
    sceneSelect.appendChild(option);
  }
}

function formatSceneLabel(fileName: string): string {
  const name = fileName.replace(/\.[^.]+$/, "");
  return name
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[-_]+/g, " ")
    .trim() || fileName;
}
