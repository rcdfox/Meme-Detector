import { access, cp, mkdir, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const root = path.resolve(import.meta.dirname, '..');
const publicDir = path.join(root, 'public');
const modelDir = path.join(publicDir, 'models');
const wasmSource = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
const wasmTarget = path.join(publicDir, 'wasm');

const models = [
  {
    file: 'face_landmarker.task',
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
  },
  {
    file: 'pose_landmarker_lite.task',
    url: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'
  }
];

async function exists(p) {
  try { await access(p); return true; } catch { return false; }
}

async function download(url, destination) {
  if (await exists(destination)) {
    const info = await stat(destination);
    if (info.size > 100_000) {
      console.log(`✓ ${path.basename(destination)} already exists`);
      return;
    }
  }
  console.log(`↓ Downloading ${path.basename(destination)}…`);
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`Download failed (${response.status}) for ${url}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  await BunlessWrite(destination, buffer);
  console.log(`✓ Saved ${(buffer.length / 1_000_000).toFixed(1)} MB`);
}

async function BunlessWrite(destination, buffer) {
  const { writeFile } = await import('node:fs/promises');
  await writeFile(destination, buffer);
}

async function main() {
  await mkdir(modelDir, { recursive: true });
  await mkdir(wasmTarget, { recursive: true });
  if (!(await exists(wasmSource))) {
    throw new Error('MediaPipe package is missing. Run `npm install` before `npm run setup`.');
  }
  console.log('→ Copying MediaPipe WASM runtime into public/wasm…');
  await cp(wasmSource, wasmTarget, { recursive: true, force: true });
  console.log('✓ WASM runtime copied');
  for (const model of models) await download(model.url, path.join(modelDir, model.file));
  console.log('\nSetup complete. You can now run `npm run dev`.');
  console.log('After setup, the app does not need a model CDN to process webcam frames.');
}

main().catch((error) => {
  console.error(`\nSetup failed: ${error.message}`);
  process.exitCode = 1;
});
