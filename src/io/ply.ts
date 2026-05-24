import { Buffer } from "node:buffer";
import type { GaussianSplat, Quat } from "../core/types.js";
import { clampByte, normalizeQuat } from "../core/math.js";
import type { PointCloud } from "../core/types.js";

type PlyFormat = "ascii" | "binary_little_endian";
type PlyType = "char" | "uchar" | "short" | "ushort" | "int" | "uint" | "float" | "double";

interface PlyProperty {
  name: string;
  type: PlyType;
}

interface PlyHeader {
  format: PlyFormat;
  vertexCount: number;
  properties: PlyProperty[];
  headerLength: number;
}

const SH_C0 = 0.28209479177387814;

export function parseGaussianPly(buffer: Buffer): GaussianSplat[] {
  const header = parseHeader(buffer);
  if (header.vertexCount < 1) {
    return [];
  }

  const rows =
    header.format === "ascii"
      ? readAsciiVertices(buffer, header)
      : readBinaryVertices(buffer, header);

  return rows.map((row, i) => rowToSplat(row, i));
}

export function writePointCloudPly(pointCloud: PointCloud): Buffer {
  const count = pointCloud.positions.length / 3;
  const header = Buffer.from(
    [
      "ply",
      "format binary_little_endian 1.0",
      `element vertex ${count}`,
      "property float x",
      "property float y",
      "property float z",
      "property uchar red",
      "property uchar green",
      "property uchar blue",
      "end_header",
      "",
    ].join("\n"),
    "utf8",
  );
  const stride = 15;
  const body = Buffer.allocUnsafe(count * stride);

  for (let i = 0; i < count; i += 1) {
    const out = i * stride;
    const pos = i * 3;
    body.writeFloatLE(pointCloud.positions[pos], out);
    body.writeFloatLE(pointCloud.positions[pos + 1], out + 4);
    body.writeFloatLE(pointCloud.positions[pos + 2], out + 8);
    body[out + 12] = pointCloud.colors[pos];
    body[out + 13] = pointCloud.colors[pos + 1];
    body[out + 14] = pointCloud.colors[pos + 2];
  }

  return Buffer.concat([header, body]);
}

function parseHeader(buffer: Buffer): PlyHeader {
  const marker = Buffer.from("end_header\n", "utf8");
  let markerIndex = buffer.indexOf(marker);
  let headerLength = markerIndex + marker.byteLength;
  if (markerIndex < 0) {
    const crlfMarker = Buffer.from("end_header\r\n", "utf8");
    markerIndex = buffer.indexOf(crlfMarker);
    headerLength = markerIndex + crlfMarker.byteLength;
  }
  if (markerIndex < 0) {
    throw new Error("Invalid PLY: missing end_header");
  }

  const text = buffer.subarray(0, headerLength).toString("utf8");
  const lines = text.split(/\r?\n/);
  let format: PlyFormat | undefined;
  let vertexCount = 0;
  const properties: PlyProperty[] = [];
  let inVertex = false;

  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts[0] === "format") {
      if (parts[1] !== "ascii" && parts[1] !== "binary_little_endian") {
        throw new Error(`Unsupported PLY format ${parts[1]}`);
      }
      format = parts[1];
    } else if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) {
        vertexCount = Number(parts[2]);
      }
    } else if (parts[0] === "property" && inVertex) {
      if (parts[1] === "list") {
        throw new Error("PLY list properties on vertices are not supported");
      }
      properties.push({ type: normalizePlyType(parts[1]), name: parts[2] });
    }
  }

  if (!format) {
    throw new Error("Invalid PLY: missing format");
  }

  return { format, vertexCount, properties, headerLength };
}

function readAsciiVertices(buffer: Buffer, header: PlyHeader): Array<Map<string, number>> {
  const text = buffer.subarray(header.headerLength).toString("utf8");
  const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
  const rows: Array<Map<string, number>> = [];

  for (let i = 0; i < header.vertexCount; i += 1) {
    const parts = lines[i].trim().split(/\s+/);
    const row = new Map<string, number>();
    for (let p = 0; p < header.properties.length; p += 1) {
      row.set(header.properties[p].name, Number(parts[p]));
    }
    rows.push(row);
  }

  return rows;
}

function readBinaryVertices(buffer: Buffer, header: PlyHeader): Array<Map<string, number>> {
  const stride = header.properties.reduce((sum, prop) => sum + plyTypeSize(prop.type), 0);
  const rows: Array<Map<string, number>> = [];
  let offset = header.headerLength;

  for (let i = 0; i < header.vertexCount; i += 1) {
    const row = new Map<string, number>();
    let cursor = offset;
    for (const prop of header.properties) {
      row.set(prop.name, readScalar(buffer, cursor, prop.type));
      cursor += plyTypeSize(prop.type);
    }
    rows.push(row);
    offset += stride;
  }

  return rows;
}

function rowToSplat(row: Map<string, number>, index: number): GaussianSplat {
  const center = requiredVec3(row, ["x", "y", "z"], index);
  const scaleLog = requiredVec3(row, ["scale_0", "scale_1", "scale_2"], index);
  const scales: [number, number, number] = [
    Math.exp(scaleLog[0]),
    Math.exp(scaleLog[1]),
    Math.exp(scaleLog[2]),
  ];
  const rotation: Quat = normalizeQuat(requiredVec4(row, ["rot_0", "rot_1", "rot_2", "rot_3"], index));
  const opacityRaw = getRequired(row, "opacity", index);
  const opacity = 1 / (1 + Math.exp(-opacityRaw));
  const color = readColor(row);
  return { center, scales, rotation, opacity, color };
}

function readColor(row: Map<string, number>): [number, number, number] {
  if (row.has("red") && row.has("green") && row.has("blue")) {
    const r = getNumber(row, "red");
    const g = getNumber(row, "green");
    const b = getNumber(row, "blue");
    return [
      clampByte(r <= 1 ? r * 255 : r),
      clampByte(g <= 1 ? g * 255 : g),
      clampByte(b <= 1 ? b * 255 : b),
    ];
  }

  if (row.has("f_dc_0") && row.has("f_dc_1") && row.has("f_dc_2")) {
    return [
      clampByte((SH_C0 * getNumber(row, "f_dc_0") + 0.5) * 255),
      clampByte((SH_C0 * getNumber(row, "f_dc_1") + 0.5) * 255),
      clampByte((SH_C0 * getNumber(row, "f_dc_2") + 0.5) * 255),
    ];
  }

  return [255, 255, 255];
}

function requiredVec3(row: Map<string, number>, names: [string, string, string], index: number): [number, number, number] {
  return [getRequired(row, names[0], index), getRequired(row, names[1], index), getRequired(row, names[2], index)];
}

function requiredVec4(
  row: Map<string, number>,
  names: [string, string, string, string],
  index: number,
): [number, number, number, number] {
  return [
    getRequired(row, names[0], index),
    getRequired(row, names[1], index),
    getRequired(row, names[2], index),
    getRequired(row, names[3], index),
  ];
}

function getRequired(row: Map<string, number>, name: string, index: number): number {
  const value = row.get(name);
  if (value === undefined || Number.isNaN(value)) {
    throw new Error(`PLY vertex ${index} is missing required property ${name}`);
  }
  return value;
}

function getNumber(row: Map<string, number>, name: string): number {
  const value = row.get(name);
  return value ?? 0;
}

function normalizePlyType(type: string): PlyType {
  if (type === "int8") return "char";
  if (type === "uint8") return "uchar";
  if (type === "int16") return "short";
  if (type === "uint16") return "ushort";
  if (type === "int32") return "int";
  if (type === "uint32") return "uint";
  if (type === "float32") return "float";
  if (type === "float64") return "double";
  if (["char", "uchar", "short", "ushort", "int", "uint", "float", "double"].includes(type)) {
    return type as PlyType;
  }
  throw new Error(`Unsupported PLY scalar type ${type}`);
}

function plyTypeSize(type: PlyType): number {
  switch (type) {
    case "char":
    case "uchar":
      return 1;
    case "short":
    case "ushort":
      return 2;
    case "int":
    case "uint":
    case "float":
      return 4;
    case "double":
      return 8;
  }
}

function readScalar(buffer: Buffer, offset: number, type: PlyType): number {
  switch (type) {
    case "char":
      return buffer.readInt8(offset);
    case "uchar":
      return buffer.readUInt8(offset);
    case "short":
      return buffer.readInt16LE(offset);
    case "ushort":
      return buffer.readUInt16LE(offset);
    case "int":
      return buffer.readInt32LE(offset);
    case "uint":
      return buffer.readUInt32LE(offset);
    case "float":
      return buffer.readFloatLE(offset);
    case "double":
      return buffer.readDoubleLE(offset);
  }
}
