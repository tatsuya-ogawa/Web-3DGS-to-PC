import type { GaussianSplat, Quat } from "../core/types.js";
import { normalizeQuat } from "../core/math.js";

export function parseSplat(buffer: ArrayBuffer | Buffer): GaussianSplat[] {
  const bytes = toUint8Array(buffer);
  const stride = 32;
  if (bytes.byteLength % stride !== 0) {
    throw new Error(`Invalid .splat file size ${bytes.byteLength}; expected a multiple of ${stride}`);
  }

  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = bytes.byteLength / stride;
  const splats: GaussianSplat[] = new Array(count);

  for (let i = 0; i < count; i += 1) {
    const base = i * stride;
    const center: [number, number, number] = [
      view.getFloat32(base, true),
      view.getFloat32(base + 4, true),
      view.getFloat32(base + 8, true),
    ];
    const scales: [number, number, number] = [
      view.getFloat32(base + 12, true),
      view.getFloat32(base + 16, true),
      view.getFloat32(base + 20, true),
    ];
    const color: [number, number, number] = [
      view.getUint8(base + 24),
      view.getUint8(base + 25),
      view.getUint8(base + 26),
    ];
    const opacity = view.getUint8(base + 27) / 255;
    const rotation: Quat = normalizeQuat([
      (view.getUint8(base + 28) - 128) / 128,
      (view.getUint8(base + 29) - 128) / 128,
      (view.getUint8(base + 30) - 128) / 128,
      (view.getUint8(base + 31) - 128) / 128,
    ]);

    splats[i] = { center, scales, rotation, opacity, color };
  }

  return splats;
}

function toUint8Array(buffer: ArrayBuffer | Buffer): Uint8Array {
  if (buffer instanceof Buffer) {
    return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  }
  return new Uint8Array(buffer);
}
