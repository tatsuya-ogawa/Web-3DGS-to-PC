export interface BrowserPointCloud {
  positions: Float32Array;
  colors: Uint8Array;
}

export interface BrowserTriangleMesh {
  positions: Float32Array;
  colors: Uint8Array;
  indices: Uint32Array;
}

export function writePointCloudPlyBytes(pointCloud: BrowserPointCloud): Uint8Array {
  const count = pointCloud.positions.length / 3;
  const header = new TextEncoder().encode(
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
  );
  const stride = 15;
  const bytes = new Uint8Array(header.byteLength + count * stride);
  bytes.set(header, 0);
  const view = new DataView(bytes.buffer, bytes.byteOffset + header.byteLength, count * stride);

  for (let i = 0; i < count; i += 1) {
    const out = i * stride;
    const pos = i * 3;
    view.setFloat32(out, pointCloud.positions[pos], true);
    view.setFloat32(out + 4, pointCloud.positions[pos + 1], true);
    view.setFloat32(out + 8, pointCloud.positions[pos + 2], true);
    view.setUint8(out + 12, pointCloud.colors[pos]);
    view.setUint8(out + 13, pointCloud.colors[pos + 1]);
    view.setUint8(out + 14, pointCloud.colors[pos + 2]);
  }

  return bytes;
}

export function writeTriangleMeshPlyBytes(mesh: BrowserTriangleMesh): Uint8Array {
  const vertexCount = mesh.positions.length / 3;
  const faceCount = mesh.indices.length / 3;
  const header = new TextEncoder().encode(
    [
      "ply",
      "format binary_little_endian 1.0",
      `element vertex ${vertexCount}`,
      "property float x",
      "property float y",
      "property float z",
      "property uchar red",
      "property uchar green",
      "property uchar blue",
      `element face ${faceCount}`,
      "property list uchar int vertex_indices",
      "end_header",
      "",
    ].join("\n"),
  );

  const vertexStride = 15;
  const faceStride = 13;
  const vertexBytes = vertexCount * vertexStride;
  const faceBytes = faceCount * faceStride;
  const bytes = new Uint8Array(header.byteLength + vertexBytes + faceBytes);
  bytes.set(header, 0);

  const vertexView = new DataView(bytes.buffer, bytes.byteOffset + header.byteLength, vertexBytes);
  for (let i = 0; i < vertexCount; i += 1) {
    const out = i * vertexStride;
    const pos = i * 3;
    vertexView.setFloat32(out, mesh.positions[pos], true);
    vertexView.setFloat32(out + 4, mesh.positions[pos + 1], true);
    vertexView.setFloat32(out + 8, mesh.positions[pos + 2], true);
    vertexView.setUint8(out + 12, mesh.colors[pos]);
    vertexView.setUint8(out + 13, mesh.colors[pos + 1]);
    vertexView.setUint8(out + 14, mesh.colors[pos + 2]);
  }

  const faceView = new DataView(bytes.buffer, bytes.byteOffset + header.byteLength + vertexBytes, faceBytes);
  for (let i = 0; i < faceCount; i += 1) {
    const out = i * faceStride;
    const index = i * 3;
    faceView.setUint8(out, 3);
    faceView.setInt32(out + 1, mesh.indices[index], true);
    faceView.setInt32(out + 5, mesh.indices[index + 1], true);
    faceView.setInt32(out + 9, mesh.indices[index + 2], true);
  }

  return bytes;
}

export function bytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
}
