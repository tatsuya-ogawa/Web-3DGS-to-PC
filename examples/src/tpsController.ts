import * as THREE from "three";
import { ChunkVoxelManager } from "web-3dgs-to-pc/browser";

export const TPS_CHARACTER_SCALE = 0.5;

export class TpsController {
  public position: THREE.Vector3;
  public velocity: THREE.Vector3;
  public radius: number = 0.4 * TPS_CHARACTER_SCALE;
  public height: number = 1.8 * TPS_CHARACTER_SCALE;
  public onGround: boolean = false;
  public cameraDistance: number = 7.5 * TPS_CHARACTER_SCALE;

  private camera: THREE.PerspectiveCamera;
  private domElement: HTMLElement;
  private chunkManager: ChunkVoxelManager;

  // Camera look state
  public yaw: number = 0;
  public pitch: number = 0.35; // Start tilted downwards for a beautiful overview

  // Keyboard controls state
  private keys: Record<string, boolean> = {
    w: false,
    a: false,
    s: false,
    d: false,
    space: false,
  };

  private gravity: number = 22.0 * TPS_CHARACTER_SCALE;
  private jumpForce: number = 8.5 * TPS_CHARACTER_SCALE;
  private moveSpeed: number = 6.0 * TPS_CHARACTER_SCALE;
  private densityThreshold: number = 120; // threshold for solid voxel
  private groundSnapDistance: number = 0.65 * TPS_CHARACTER_SCALE;
  private maxStepHeight: number = 0.8 * TPS_CHARACTER_SCALE;
  private jumpRequested: boolean = false;
  private isDragging: boolean = false;
  private previousPosition: THREE.Vector3 = new THREE.Vector3();

  constructor(
    camera: THREE.PerspectiveCamera,
    domElement: HTMLElement,
    chunkManager: ChunkVoxelManager,
    initialPos: THREE.Vector3 = new THREE.Vector3(0, 5, 0)
  ) {
    this.camera = camera;
    this.domElement = domElement;
    this.chunkManager = chunkManager;
    this.position = initialPos.clone();
    this.velocity = new THREE.Vector3();

    this.setupListeners();
  }

  private setupListeners(): void {
    window.addEventListener("keydown", (e) => {
      const key = e.key.toLowerCase();
      if (key === "w" || key === "a" || key === "s" || key === "d") {
        this.keys[key] = true;
      }
      if (e.code === "Space") {
        this.keys.space = true;
        if (!e.repeat) {
          this.jumpRequested = true;
        }
        e.preventDefault();
      }
    });

    window.addEventListener("keyup", (e) => {
      const key = e.key.toLowerCase();
      if (key === "w" || key === "a" || key === "s" || key === "d") {
        this.keys[key] = false;
      }
      if (e.code === "Space") {
        this.keys.space = false;
        e.preventDefault();
      }
    });

    window.addEventListener("blur", () => {
      this.clearInputState();
    });

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.clearInputState();
      }
    });

    this.domElement.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      this.isDragging = true;
      this.domElement.setPointerCapture(e.pointerId);
      e.preventDefault();
    });

    this.domElement.addEventListener("pointerup", (e) => {
      this.isDragging = false;
      if (this.domElement.hasPointerCapture(e.pointerId)) {
        this.domElement.releasePointerCapture(e.pointerId);
      }
    });

    this.domElement.addEventListener("pointercancel", () => {
      this.isDragging = false;
    });

    this.domElement.addEventListener("contextmenu", (e) => e.preventDefault());

    this.domElement.addEventListener("pointermove", (e) => {
      if (!this.isDragging) return;

      this.yaw -= e.movementX * 0.0022;
      this.pitch -= e.movementY * 0.0022;

      // Clamp pitch to avoid flipping over
      const maxPitch = Math.PI / 2 - 0.08;
      this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));
      e.preventDefault();
    });

    window.addEventListener("wheel", (e) => {
      this.cameraDistance += e.deltaY * 0.015 * TPS_CHARACTER_SCALE;
      this.cameraDistance = Math.max(3.0 * TPS_CHARACTER_SCALE, Math.min(22.0 * TPS_CHARACTER_SCALE, this.cameraDistance));
    });
  }

  public update(dt: number): void {
    // Limit delta time to avoid large physics jumps during lag
    dt = Math.min(dt, 0.05);

    // 1. Apply gravity
    if (!this.onGround) {
      this.velocity.y -= this.gravity * dt;
    } else {
      this.velocity.y = 0;
    }

    // 2. Handle Jump
    if (this.onGround && this.jumpRequested) {
      this.velocity.y = this.jumpForce;
      this.onGround = false;
    }
    this.jumpRequested = false;

    // 3. Process movement direction from WASD
    const moveVector = new THREE.Vector3();
    if (this.keys.w) moveVector.z -= 1;
    if (this.keys.s) moveVector.z += 1;
    if (this.keys.a) moveVector.x -= 1;
    if (this.keys.d) moveVector.x += 1;

    moveVector.normalize();

    // Rotate movement vector by camera yaw (horizontal rotation)
    const forward = new THREE.Vector3(0, 0, -1).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);
    const right = new THREE.Vector3(1, 0, 0).applyAxisAngle(new THREE.Vector3(0, 1, 0), this.yaw);

    const horizontalVelocity = new THREE.Vector3();
    horizontalVelocity.addScaledVector(forward, -moveVector.z); // forward/back
    horizontalVelocity.addScaledVector(right, moveVector.x);    // strafe

    if (moveVector.lengthSq() > 0) {
      horizontalVelocity.normalize().multiplyScalar(this.moveSpeed);
    }

    this.previousPosition.copy(this.position);

    // Update player position
    this.position.x += horizontalVelocity.x * dt;
    this.position.y += this.velocity.y * dt;
    this.position.z += horizontalVelocity.z * dt;

    // 4. Resolve Terrain Collisions against Local Voxels
    const collisionState = this.resolveVoxelCollisions();
    if (collisionState === "loading") {
      this.position.copy(this.previousPosition);
      this.velocity.set(0, 0, 0);
      this.onGround = true;
    }

    // 5. Snap camera behind character
    this.updateCamera();
  }

  private resolveVoxelCollisions(): "hit" | "miss" | "loading" {
    let surface = this.chunkManager.querySupportHeightAtWorldXZ(
      this.position.x,
      this.position.z,
      this.densityThreshold,
      this.position.y,
    );

    if (surface.height === undefined && !surface.loading) {
      surface = this.chunkManager.querySurfaceHeightAtWorldXZ(
        this.position.x,
        this.position.z,
        this.densityThreshold,
        this.position.y + this.maxStepHeight,
      );
    }

    const surfaceY = surface.height;
    if (surfaceY === undefined) {
      this.onGround = false;
      return surface.loading ? "loading" : "miss";
    }

    const heightDelta = this.position.y - surfaceY;
    const canSnapDown = heightDelta >= 0 && heightDelta <= this.groundSnapDistance;
    const isBelowSupport = heightDelta < 0;
    const shouldFollowGround = this.onGround && (isBelowSupport || heightDelta <= this.maxStepHeight);

    if (this.velocity.y <= 0 && (canSnapDown || isBelowSupport || shouldFollowGround)) {
      this.position.y = surfaceY;
      this.velocity.y = 0;
      this.onGround = true;
      return "hit";
    }

    this.onGround = false;
    return "miss";
  }

  private updateCamera(): void {
    const headHeight = 1.5 * TPS_CHARACTER_SCALE;
    const targetPos = this.position.clone().add(new THREE.Vector3(0, headHeight, 0));

    const dist = this.cameraDistance;
    const offset = new THREE.Vector3(
      Math.sin(this.yaw) * Math.cos(this.pitch) * dist,
      Math.sin(this.pitch) * dist + 2.8 * TPS_CHARACTER_SCALE,
      Math.cos(this.yaw) * Math.cos(this.pitch) * dist
    );

    this.camera.position.copy(targetPos).add(offset);
    this.camera.lookAt(targetPos);
  }

  public respawn(pos: THREE.Vector3 = new THREE.Vector3(0, 5, 0)): void {
    this.position.copy(pos);
    this.velocity.set(0, 0, 0);
    this.onGround = false;
  }

  public isActive(): boolean {
    return (
      this.isDragging ||
      this.jumpRequested ||
      this.keys.w ||
      this.keys.a ||
      this.keys.s ||
      this.keys.d ||
      !this.onGround ||
      Math.abs(this.velocity.y) > 0.01
    );
  }

  private clearInputState(): void {
    this.keys.w = false;
    this.keys.a = false;
    this.keys.s = false;
    this.keys.d = false;
    this.keys.space = false;
    this.jumpRequested = false;
    this.isDragging = false;
    this.velocity.x = 0;
    this.velocity.z = 0;
  }
}
