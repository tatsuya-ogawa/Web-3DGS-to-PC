import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import {
  writePointCloudPlyBytes,
  writeTriangleMeshPlyBytes,
  packSplats,
  voxelizeWithWebGpu,
  type WebGpuVoxelResult,
} from "web-3dgs-to-pc/browser";
import "./styles.css";

interface StatusPatch {
  stage?: string;
  splats?: string;
  grid?: string;
  output?: string;
  adapter?: string;
  progress?: number;
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

const root = document.querySelector<HTMLDivElement>("#app");
if (!root) {
  throw new Error("Missing #app root");
}

root.innerHTML = `
  <main class="app-shell">
    <section class="panel">
      <header class="panel-header">
        <h1 class="title">Web 3DGS to PC</h1>
        <p class="subtitle">Spark SOG decode, WebGPU voxelization, PLY export.</p>
      </header>

      <div class="controls">
        <label>
          Bundled scene
          <select id="source-select"></select>
        </label>

        <label>
          Local 3DGS file
          <input id="file-input" type="file" accept=".sog,.spz,.splat,.ksplat,.ply" />
        </label>

        <div class="field-grid">
          <label>
            Resolution
            <input id="resolution-input" type="number" min="16" max="256" step="1" value="96" />
          </label>
          <label>
            Max points
            <input id="max-points-input" type="number" min="1" step="1000" value="50000" />
          </label>
          <label>
            Iso
            <input id="iso-input" type="number" min="0" step="0.0005" value="0.001" />
          </label>
          <label>
            Iso percentile
            <input id="iso-percentile-input" type="number" min="0" max="1" step="0.01" value="0.85" />
          </label>
          <label>
            Sigma radius
            <input id="sigma-radius-input" type="number" min="0.25" step="0.25" value="3" />
          </label>
          <label>
            Bounds trim
            <input id="bounds-quantile-input" type="number" min="0" max="0.2" step="0.005" value="0.01" />
          </label>
          <label>
            Max splats
            <input id="max-splats-input" type="number" min="1" step="10000" placeholder="all" />
          </label>
          <label>
            Density scale
            <input id="density-scale-input" type="number" min="0.01" step="0.1" value="1" />
          </label>
          <label>
            Smooth iterations
            <input id="smooth-iterations-input" type="number" min="0" max="10" step="1" value="0" />
          </label>
        </div>

        <label>
          Output Mode
          <select id="output-mode-select">
            <option value="both-dual">Points & Mesh (Dual Contouring)</option>
            <option value="both-tetra">Points & Mesh (Marching Tetra)</option>
            <option value="voxels" selected>Points & Raw Voxels</option>
            <option value="points">Points Only</option>
          </select>
        </label>

        <div class="toggle-row" style="margin-top: -8px; margin-bottom: 8px;">
          <span>Color output</span>
          <input id="color-output-input" type="checkbox" checked />
        </div>

        <div class="toggle-row" style="margin-top: -8px; margin-bottom: 8px;">
          <span>Flip Y-Axis (COLMAP)</span>
          <input id="flip-y-input" type="checkbox" checked />
        </div>

        <div id="overlay-section" style="display: none; border-top: 1px solid #e1e3de; padding-top: 12px; margin-top: 4px; margin-bottom: 8px;">
          <h3 style="margin: 0 0 10px; font-size: 12px; font-weight: 700; color: #263836; text-transform: uppercase; letter-spacing: 0.5px;">Overlay Controls</h3>
          
          <div class="toggle-row" style="min-height: 30px; margin-bottom: 8px;">
            <span style="font-size: 12px; font-weight: 500;">Original 3DGS Splats</span>
            <input id="splat-show-input" type="checkbox" checked style="width: 16px; min-height: 16px;" />
          </div>

          <label style="margin-bottom: 10px; font-size: 11px; gap: 4px;">
            Splat Opacity
            <input id="splat-opacity-input" type="range" min="0" max="1" step="0.05" value="1.00" style="height: 6px; padding: 0; cursor: pointer;" />
          </label>

          <div class="toggle-row" style="min-height: 30px; margin-bottom: 8px; border-top: 1px dashed #e1e3de; padding-top: 8px;">
            <span style="font-size: 12px; font-weight: 500;">Mesh Wireframe</span>
            <input id="mesh-wireframe-input" type="checkbox" style="width: 16px; min-height: 16px;" />
          </div>

          <label style="margin-bottom: 10px; font-size: 11px; gap: 4px;">
            Mesh Opacity
            <input id="mesh-opacity-input" type="range" min="0" max="1" step="0.05" value="0.70" style="height: 6px; padding: 0; cursor: pointer;" />
          </label>

          <label style="margin-bottom: 10px; font-size: 11px; gap: 4px; border-top: 1px dashed #e1e3de; padding-top: 8px;">
            Point Size
            <input id="point-size-input" type="range" min="0.1" max="5" step="0.1" value="1.0" style="height: 6px; padding: 0; cursor: pointer;" />
          </label>

          <label style="margin-bottom: 4px; font-size: 11px; gap: 4px;">
            Points Opacity
            <input id="points-opacity-input" type="range" min="0" max="1" step="0.05" value="1.00" style="height: 6px; padding: 0; cursor: pointer;" />
          </label>
        </div>

        <div class="button-row">
          <button id="run-button">Run</button>
          <button id="reset-button" class="secondary">Reset camera</button>
          <button id="download-points-button" class="secondary" disabled>Point PLY</button>
          <button id="download-mesh-button" class="secondary" disabled>Mesh PLY</button>
        </div>
      </div>

      <footer class="status">
        <div class="progress"><div id="progress-bar" class="progress-bar"></div></div>
        <div class="status-line"><span>Stage</span><strong id="stage-value">Idle</strong></div>
        <div class="status-line"><span>Splats</span><strong id="splats-value">-</strong></div>
        <div class="status-line"><span>Grid</span><strong id="grid-value">-</strong></div>
        <div class="status-line"><span>Output</span><strong id="output-value">-</strong></div>
        <div class="status-line"><span>Adapter</span><strong id="adapter-value">-</strong></div>
      </footer>
    </section>

    <section class="viewport">
      <div class="toolbar" style="gap: 10px;">
        <span style="font-size: 11px; font-weight: 700; color: #5e6966; text-transform: uppercase; margin-right: 4px;">Layers</span>
        <button id="toggle-layer-splat">Splats</button>
        <button id="toggle-layer-points">Points</button>
        <button id="toggle-layer-mesh">Mesh/Voxels</button>
      </div>
      <div id="canvas-host" class="canvas-host"></div>
      <div id="empty-state" class="empty-state"><div><strong>No output yet</strong><span>Run a scene to inspect the result.</span></div></div>
    </section>
  </main>
`;

const sourceSelect = getElement<HTMLSelectElement>("source-select");
const fileInput = getElement<HTMLInputElement>("file-input");
const resolutionInput = getElement<HTMLInputElement>("resolution-input");
const maxPointsInput = getElement<HTMLInputElement>("max-points-input");
const isoInput = getElement<HTMLInputElement>("iso-input");
const isoPercentileInput = getElement<HTMLInputElement>("iso-percentile-input");
const sigmaRadiusInput = getElement<HTMLInputElement>("sigma-radius-input");
const boundsQuantileInput = getElement<HTMLInputElement>("bounds-quantile-input");
const maxSplatsInput = getElement<HTMLInputElement>("max-splats-input");
const densityScaleInput = getElement<HTMLInputElement>("density-scale-input");
const smoothIterationsInput = getElement<HTMLInputElement>("smooth-iterations-input");
const outputModeSelect = getElement<HTMLSelectElement>("output-mode-select");
const colorOutputInput = getElement<HTMLInputElement>("color-output-input");
const flipYInput = getElement<HTMLInputElement>("flip-y-input");
const runButton = getElement<HTMLButtonElement>("run-button");
const resetButton = getElement<HTMLButtonElement>("reset-button");
const downloadPointsButton = getElement<HTMLButtonElement>("download-points-button");
const downloadMeshButton = getElement<HTMLButtonElement>("download-mesh-button");
const splatShowInput = getElement<HTMLInputElement>("splat-show-input");
const splatOpacityInput = getElement<HTMLInputElement>("splat-opacity-input");
const meshWireframeInput = getElement<HTMLInputElement>("mesh-wireframe-input");
const meshOpacityInput = getElement<HTMLInputElement>("mesh-opacity-input");
const pointSizeInput = getElement<HTMLInputElement>("point-size-input");
const pointsOpacityInput = getElement<HTMLInputElement>("points-opacity-input");
const overlaySection = getElement<HTMLDivElement>("overlay-section");
const toggleLayerSplat = getElement<HTMLButtonElement>("toggle-layer-splat");
const toggleLayerPoints = getElement<HTMLButtonElement>("toggle-layer-points");
const toggleLayerMesh = getElement<HTMLButtonElement>("toggle-layer-mesh");
const progressBar = getElement<HTMLDivElement>("progress-bar");
const canvasHost = getElement<HTMLDivElement>("canvas-host");
const emptyState = getElement<HTMLDivElement>("empty-state");

const stageValue = getElement<HTMLElement>("stage-value");
const splatsValue = getElement<HTMLElement>("splats-value");
const gridValue = getElement<HTMLElement>("grid-value");
const outputValue = getElement<HTMLElement>("output-value");
const adapterValue = getElement<HTMLElement>("adapter-value");

let lastSourceName = "scene";
let lastPointBytes: Uint8Array | undefined;
let lastMeshBytes: Uint8Array | undefined;
let pointsObject: THREE.Points | undefined;
let meshObject: THREE.Mesh | undefined;
let voxelsObject: THREE.InstancedMesh | undefined;
let gridHelper: THREE.GridHelper | undefined;
let lastCameraSphere: THREE.Sphere | undefined;
let defaultPointSize = 0.01;
let splatMeshObject: SplatMesh | undefined;
let sparkRenderer: SparkRenderer | undefined;

let splatVisible = true;
let pointsVisible = true;
let meshVisible = true;

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
canvasHost.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 1000);
const controls = new OrbitControls(camera, renderer.domElement);

sparkRenderer = new SparkRenderer({ renderer });
scene.add(sparkRenderer);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

const group = new THREE.Group();
scene.add(group);
scene.add(new THREE.HemisphereLight(0xffffff, 0x66736f, 1.35));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
keyLight.position.set(3, 5, 4);
scene.add(keyLight);

resizeRenderer();
window.addEventListener("resize", resizeRenderer);
runButton.addEventListener("click", () => void runConversion());
resetButton.addEventListener("click", resetCamera);
downloadPointsButton.addEventListener("click", () => {
  if (lastPointBytes) {
    downloadBytes(lastPointBytes, `${safeBaseName(lastSourceName)}_points.ply`);
  }
});
downloadMeshButton.addEventListener("click", () => {
  if (lastMeshBytes) {
    downloadBytes(lastMeshBytes, `${safeBaseName(lastSourceName)}_mesh.ply`);
  }
});
toggleLayerSplat.addEventListener("click", () => {
  splatVisible = !splatVisible;
  updateLayerVisibilities();
});
toggleLayerPoints.addEventListener("click", () => {
  pointsVisible = !pointsVisible;
  updateLayerVisibilities();
});
toggleLayerMesh.addEventListener("click", () => {
  meshVisible = !meshVisible;
  updateLayerVisibilities();
});

populateBundledSourceSelect();
sourceSelect.addEventListener("change", () => void loadAndPreviewSplat());
fileInput.addEventListener("change", () => void loadAndPreviewSplat());

flipYInput.addEventListener("change", () => {
  const rotX = flipYInput.checked ? Math.PI : 0;
  group.rotation.x = rotX;
  if (splatMeshObject) {
    splatMeshObject.rotation.x = rotX;
    
    // Reset camera target and orbit based on new world rotation
    const box = splatMeshObject.getBoundingBox().clone();
    splatMeshObject.updateMatrixWorld(true);
    box.applyMatrix4(splatMeshObject.matrixWorld);
    
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
      lastCameraSphere = sphere;
      if (gridHelper) {
        gridHelper.position.copy(sphere.center);
        gridHelper.position.y = box.min.y;
      }
      resetCamera();
    }
  }
});

splatShowInput.addEventListener("change", () => {
  splatVisible = splatShowInput.checked;
  updateLayerVisibilities();
});

splatOpacityInput.addEventListener("input", () => {
  if (splatMeshObject) {
    splatMeshObject.opacity = Number(splatOpacityInput.value);
  }
});

meshWireframeInput.addEventListener("change", () => {
  if (meshObject) {
    const material = meshObject.material as THREE.MeshStandardMaterial;
    material.wireframe = meshWireframeInput.checked;
  }
});

meshOpacityInput.addEventListener("input", () => {
  if (meshObject) {
    const material = meshObject.material as THREE.MeshStandardMaterial;
    material.opacity = Number(meshOpacityInput.value);
  }
  if (voxelsObject) {
    const material = voxelsObject.material as THREE.MeshStandardMaterial;
    material.opacity = Number(meshOpacityInput.value);
  }
});

pointSizeInput.addEventListener("input", () => {
  if (pointsObject) {
    const material = pointsObject.material as THREE.PointsMaterial;
    material.size = defaultPointSize * Number(pointSizeInput.value);
  }
});

pointsOpacityInput.addEventListener("input", () => {
  if (pointsObject) {
    const material = pointsObject.material as THREE.PointsMaterial;
    material.opacity = Number(pointsOpacityInput.value);
  }
});

async function loadAndPreviewSplat(): Promise<void> {
  clearSceneObjects();
  if (splatMeshObject) {
    scene.remove(splatMeshObject);
    splatMeshObject.dispose();
    splatMeshObject = undefined;
  }
  
  setStatus({ stage: "Loading Splat", progress: 20 });
  try {
    const source = await readSourceBytes();
    lastSourceName = source.fileName;
    
    splatMeshObject = new SplatMesh({
      fileBytes: source.bytes,
      fileName: source.fileName,
      extSplats: true,
    });
    await splatMeshObject.initialized;
    splatMeshObject.opacity = Number(splatOpacityInput.value);
    splatMeshObject.visible = splatShowInput.checked;
    
    const rotX = flipYInput.checked ? Math.PI : 0;
    splatMeshObject.rotation.x = rotX;
    group.rotation.x = rotX;
    
    scene.add(splatMeshObject);
    
    const box = splatMeshObject.getBoundingBox().clone();
    splatMeshObject.updateMatrixWorld(true);
    box.applyMatrix4(splatMeshObject.matrixWorld);
    
    const sphere = new THREE.Sphere();
    box.getBoundingSphere(sphere);
    if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
      lastCameraSphere = sphere;
      
      gridHelper = new THREE.GridHelper(sphere.radius * 3, 12, 0x9aa8a3, 0xc9d1ce);
      gridHelper.position.copy(sphere.center);
      gridHelper.position.y = box.min.y;
      scene.add(gridHelper);
      
      resetCamera();
    }
    
    emptyState.style.display = "none";
    splatVisible = true;
    pointsVisible = false;
    meshVisible = false;
    updateLayerVisibilities();
    setStatus({ stage: "Splat loaded cleanly", progress: 100, splats: `${formatCount(splatMeshObject.numSplats)}` });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus({ stage: `Failed to load splat: ${message}`, progress: 0 });
  }
}

animate();

// Automatically load default splat
setTimeout(() => void loadAndPreviewSplat(), 100);

async function runConversion(): Promise<void> {
  if (!navigator.gpu) {
    setStatus({ stage: "WebGPU unavailable", progress: 0 });
    return;
  }

  setRunning(true);
  clearOutput();

  try {
    setStatus({ stage: "WebGPU", progress: 40 });
    await nextPaint();
    
    if (!splatMeshObject) {
      await loadAndPreviewSplat();
    }
    
    if (!splatMeshObject) {
      throw new Error("No splat scene is loaded");
    }

    const inputSplats = splatMeshObject.numSplats;
    const maxSplats = readOptionalInteger(maxSplatsInput);
    const usedSplats = Math.min(inputSplats, maxSplats ?? inputSplats);
    const packed = packSplats(splatMeshObject, usedSplats, readNumber(boundsQuantileInput, 0.01));

    const outputMode = outputModeSelect.value;
    const mesh = outputMode === "both-dual" || outputMode === "both-tetra";
    const meshType = outputMode === "both-dual" ? "dual" : "tetrahedra";
    const returnVoxels = outputMode === "voxels";

    const gpuStarted = performance.now();
    const result = await voxelizeWithWebGpu(packed, {
      resolution: readInteger(resolutionInput, 96),
      sigmaRadius: readNumber(sigmaRadiusInput, 3),
      iso: readOptionalNumber(isoInput),
      isoPercentile: readNumber(isoPercentileInput, 0.85),
      maxPoints: readInteger(maxPointsInput, 50000),
      densityScale: readNumber(densityScaleInput, 1),
      minOpacity: 0,
      atomicScale: 512,
      jitter: 0.35,
      seed: 1,
      mesh,
      meshType,
      returnVoxels,
      smoothIterations: readInteger(smoothIterationsInput, 0),
      noColor: !colorOutputInput.checked,
    });
    const gpuMs = performance.now() - gpuStarted;

    setStatus({ stage: "Encoding", progress: 80 });
    await nextPaint();
    lastPointBytes = writePointCloudPlyBytes(result);
    lastMeshBytes = result.mesh ? writeTriangleMeshPlyBytes(result.mesh) : undefined;
    renderResult(result);

    const meshFaces = result.mesh ? result.mesh.indices.length / 3 : 0;
    setStatus({
      stage: `Done ${formatSeconds(gpuMs)}`,
      grid: `${result.dims.join("x")} iso=${result.isoThreshold.toPrecision(4)}`,
      output: `${formatCount(result.positions.length / 3)} pts / ${formatCount(meshFaces)} faces`,
      adapter: result.adapterInfo || "unknown",
      progress: 100,
    });
    downloadPointsButton.disabled = false;
    downloadMeshButton.disabled = !lastMeshBytes;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus({ stage: message, progress: 0 });
  } finally {
    setRunning(false);
  }
}

async function readSourceBytes(): Promise<{ bytes: Uint8Array; fileName: string }> {
  const file = fileInput.files?.[0];
  if (file) {
    return {
      bytes: new Uint8Array(await file.arrayBuffer()),
      fileName: file.name,
    };
  }

  const selected = bundledSources.find((source) => source.url === sourceSelect.value);
  if (!selected) {
    throw new Error("Choose a local file or add .sog, .spz, .splat, .ksplat, or .ply files to examples/splats.");
  }

  const response = await fetch(selected.url);
  if (!response.ok) {
    throw new Error(`Failed to load ${selected.fileName}: ${response.status}`);
  }
  return {
    bytes: new Uint8Array(await response.arrayBuffer()),
    fileName: selected.fileName,
  };
}

function renderResult(result: WebGpuVoxelResult): void {
  clearSceneObjects();

  defaultPointSize = Math.max(result.voxelSize * 0.35, 0.01);
  const pointGeometry = new THREE.BufferGeometry();
  pointGeometry.setAttribute("position", new THREE.BufferAttribute(result.positions, 3));
  pointGeometry.setAttribute("color", new THREE.Uint8BufferAttribute(result.colors, 3, true));
  const pointMaterial = new THREE.PointsMaterial({
    size: defaultPointSize * Number(pointSizeInput.value),
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: Number(pointsOpacityInput.value),
  });
  pointsObject = new THREE.Points(pointGeometry, pointMaterial);
  group.add(pointsObject);

  if (result.mesh && result.mesh.indices.length > 0) {
    const meshGeometry = new THREE.BufferGeometry();
    meshGeometry.setAttribute("position", new THREE.BufferAttribute(result.mesh.positions, 3));
    meshGeometry.setAttribute("color", new THREE.Uint8BufferAttribute(result.mesh.colors, 3, true));
    meshGeometry.setIndex(new THREE.BufferAttribute(result.mesh.indices, 1));
    meshGeometry.computeVertexNormals();
    const meshMaterial = new THREE.MeshStandardMaterial({
      roughness: 0.72,
      metalness: 0,
      side: THREE.DoubleSide,
      vertexColors: true,
      transparent: true,
      opacity: Number(meshOpacityInput.value),
      wireframe: meshWireframeInput.checked,
    });
    meshObject = new THREE.Mesh(meshGeometry, meshMaterial);
    group.add(meshObject);
  }

  if (result.voxelCenters && result.voxelCenters.length > 0 && result.voxelColors) {
    const voxelCount = result.voxelCenters.length / 3;
    const geometry = new THREE.BoxGeometry(result.voxelSize * 0.96, result.voxelSize * 0.96, result.voxelSize * 0.96);
    const material = new THREE.MeshStandardMaterial({
      roughness: 0.72,
      metalness: 0,
      transparent: true,
      opacity: Number(meshOpacityInput.value),
    });
    voxelsObject = new THREE.InstancedMesh(geometry, material, voxelCount);

    const dummy = new THREE.Object3D();
    const tempColor = new THREE.Color();
    for (let i = 0; i < voxelCount; i++) {
      dummy.position.set(
        result.voxelCenters[i * 3],
        result.voxelCenters[i * 3 + 1],
        result.voxelCenters[i * 3 + 2]
      );
      dummy.updateMatrix();
      voxelsObject.setMatrixAt(i, dummy.matrix);

      tempColor.setRGB(
        result.voxelColors[i * 3] / 255,
        result.voxelColors[i * 3 + 1] / 255,
        result.voxelColors[i * 3 + 2] / 255
      );
      voxelsObject.setColorAt(i, tempColor);
    }
    voxelsObject.instanceMatrix.needsUpdate = true;
    voxelsObject.instanceColor!.needsUpdate = true;
    group.add(voxelsObject);
  }

  const box = new THREE.Box3().setFromObject(group);
  const sphere = new THREE.Sphere();
  box.getBoundingSphere(sphere);
  if (Number.isFinite(sphere.radius) && sphere.radius > 0) {
    lastCameraSphere = sphere;
    gridHelper = new THREE.GridHelper(sphere.radius * 3, 12, 0x9aa8a3, 0xc9d1ce);
    gridHelper.position.copy(sphere.center);
    gridHelper.position.y = box.min.y;
    scene.add(gridHelper);
    resetCamera();
  }

  overlaySection.style.display = "block";
  splatVisible = true;
  pointsVisible = false;
  meshVisible = true;
  updateLayerVisibilities();
  emptyState.style.display = "none";
}

function updateLayerVisibilities(): void {
  if (splatMeshObject) {
    splatMeshObject.visible = splatVisible;
    splatShowInput.checked = splatVisible;
  }
  if (pointsObject) {
    pointsObject.visible = pointsVisible;
  }
  if (meshObject) {
    meshObject.visible = meshVisible;
  }
  if (voxelsObject) {
    voxelsObject.visible = meshVisible;
  }

  if (splatVisible) {
    toggleLayerSplat.classList.remove("secondary");
  } else {
    toggleLayerSplat.classList.add("secondary");
  }

  if (pointsVisible) {
    toggleLayerPoints.classList.remove("secondary");
  } else {
    toggleLayerPoints.classList.add("secondary");
  }

  if (meshVisible) {
    toggleLayerMesh.classList.remove("secondary");
  } else {
    toggleLayerMesh.classList.add("secondary");
  }
}

function clearOutput(): void {
  lastPointBytes = undefined;
  lastMeshBytes = undefined;
  downloadPointsButton.disabled = true;
  downloadMeshButton.disabled = true;
  clearSceneObjects();
  if (splatMeshObject) {
    scene.remove(splatMeshObject);
    splatMeshObject.dispose();
    splatMeshObject = undefined;
  }
  emptyState.style.display = "grid";
  overlaySection.style.display = "none";
}

function clearSceneObjects(): void {
  for (const object of [...group.children]) {
    group.remove(object);
    disposeObject(object);
  }
  if (gridHelper) {
    scene.remove(gridHelper);
    gridHelper.geometry.dispose();
    if (Array.isArray(gridHelper.material)) {
      gridHelper.material.forEach((material) => material.dispose());
    } else {
      gridHelper.material.dispose();
    }
    gridHelper = undefined;
  }
  pointsObject = undefined;
  meshObject = undefined;
  voxelsObject = undefined;
}

function disposeObject(object: THREE.Object3D): void {
  const maybeMesh = object as THREE.Object3D & {
    geometry?: THREE.BufferGeometry;
    material?: THREE.Material | THREE.Material[];
  };
  maybeMesh.geometry?.dispose();
  if (Array.isArray(maybeMesh.material)) {
    maybeMesh.material.forEach((material) => material.dispose());
  } else {
    maybeMesh.material?.dispose();
  }
}

// Simple close-up resets camera to look nicely at loaded scene
function resetCamera(): void {
  const sphere = lastCameraSphere;
  if (!sphere) {
    camera.position.set(4, 3, 4);
    controls.target.set(0, 0, 0);
    controls.update();
    return;
  }

  const radius = Math.max(sphere.radius, 1);
  camera.near = radius / 1000;
  camera.far = radius * 80;
  camera.updateProjectionMatrix();
  
  const dist = 3.5;
  camera.position.set(
    sphere.center.x + dist * 0.8,
    sphere.center.y + dist * 0.5,
    sphere.center.z + dist * 0.8
  );
  controls.target.copy(sphere.center);
  controls.update();
}

function resizeRenderer(): void {
  const rect = canvasHost.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate(): void {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function setStatus(patch: StatusPatch): void {
  if (patch.stage !== undefined) {
    stageValue.textContent = patch.stage;
  }
  if (patch.splats !== undefined) {
    splatsValue.textContent = patch.splats;
  }
  if (patch.grid !== undefined) {
    gridValue.textContent = patch.grid;
  }
  if (patch.output !== undefined) {
    outputValue.textContent = patch.output;
  }
  if (patch.adapter !== undefined) {
    adapterValue.textContent = patch.adapter;
  }
  if (patch.progress !== undefined) {
    progressBar.style.width = `${Math.max(0, Math.min(100, patch.progress))}%`;
  }
}

function setRunning(running: boolean): void {
  runButton.disabled = running;
  fileInput.disabled = running;
  sourceSelect.disabled = running || bundledSources.length === 0;
}

function downloadBytes(bytes: Uint8Array, fileName: string): void {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  const blob = new Blob([buffer], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(url);
}

function getElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing #${id}`);
  }
  return element as T;
}

function readInteger(input: HTMLInputElement, fallback: number): number {
  const parsed = Number(input.value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function readOptionalInteger(input: HTMLInputElement): number | undefined {
  const value = input.value.trim();
  if (!value) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

function readNumber(input: HTMLInputElement, fallback: number): number {
  const parsed = Number(input.value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readOptionalNumber(input: HTMLInputElement): number | undefined {
  const value = input.value.trim();
  if (!value) {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined;
}

function nextPaint(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

function formatCount(value: number): string {
  return Math.round(value).toLocaleString();
}

// Convert ms to beautiful formatted string
function formatSeconds(ms: number): string {
  return `${(ms / 1000).toFixed(2)}s`;
}

// Safe name helper
function safeBaseName(fileName: string): string {
  return fileName.replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "_") || "scene";
}

function populateBundledSourceSelect(): void {
  sourceSelect.innerHTML = "";
  if (bundledSources.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No bundled scenes";
    sourceSelect.appendChild(option);
    sourceSelect.disabled = true;
    return;
  }

  for (const source of bundledSources) {
    const option = document.createElement("option");
    option.value = source.url;
    option.textContent = source.label;
    sourceSelect.appendChild(option);
  }
}

function formatSceneLabel(fileName: string): string {
  const name = fileName.replace(/\.[^.]+$/, "");
  return name
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[-_]+/g, " ")
    .trim() || fileName;
}
