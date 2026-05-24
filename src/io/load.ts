import { readFile } from "node:fs/promises";
import { extname } from "node:path";
import type { GaussianSplat } from "../core/types.js";
import { parseGaussianPly } from "./ply.js";
import { parseSplat } from "./splat.js";

export async function loadGaussianSplats(path: string): Promise<GaussianSplat[]> {
  const bytes = await readFile(path);
  const ext = extname(path).toLowerCase();
  if (ext === ".splat") {
    return parseSplat(bytes);
  }
  if (ext === ".ply") {
    return parseGaussianPly(bytes);
  }
  throw new Error(`Unsupported input extension ${ext}. Supported: .splat, .ply`);
}
