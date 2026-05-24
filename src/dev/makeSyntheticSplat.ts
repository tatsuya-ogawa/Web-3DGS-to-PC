#!/usr/bin/env node
import { writeFile } from "node:fs/promises";
import { Buffer } from "node:buffer";

const outPath = process.argv[2] ?? "fixtures/synthetic.splat";
const count = Number(process.argv[3] ?? 64);
const buffer = Buffer.alloc(count * 32);

for (let i = 0; i < count; i += 1) {
  const angle = (i / count) * Math.PI * 2;
  const ring = i % 2 === 0 ? 0.55 : 0.25;
  const x = Math.cos(angle) * ring;
  const y = Math.sin(angle) * ring;
  const z = ((i % 8) - 3.5) * 0.06;
  const base = i * 32;

  buffer.writeFloatLE(x, base);
  buffer.writeFloatLE(y, base + 4);
  buffer.writeFloatLE(z, base + 8);
  buffer.writeFloatLE(0.05 + (i % 3) * 0.015, base + 12);
  buffer.writeFloatLE(0.035 + (i % 4) * 0.01, base + 16);
  buffer.writeFloatLE(0.045, base + 20);
  buffer[base + 24] = 90 + ((i * 37) % 150);
  buffer[base + 25] = 80 + ((i * 17) % 150);
  buffer[base + 26] = 120 + ((i * 29) % 120);
  buffer[base + 27] = 210;
  buffer[base + 28] = 255;
  buffer[base + 29] = 128;
  buffer[base + 30] = 128;
  buffer[base + 31] = 128;
}

await writeFile(outPath, buffer);
console.log(`Wrote ${count} synthetic splats to ${outPath}`);
