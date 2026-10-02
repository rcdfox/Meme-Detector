import test from 'node:test';
import assert from 'node:assert/strict';
import { computeCoverRect, computeFaceOverlayTransform, smoothOverlayTransform } from '../src/overlay.js';

function syntheticFace() {
  const landmarks = Array(468);
  landmarks[0] = { x: 0.50, y: 0.20 };
  landmarks[1] = { x: 0.50, y: 0.80 };
  landmarks[2] = { x: 0.30, y: 0.50 };
  landmarks[3] = { x: 0.70, y: 0.50 };
  landmarks[33] = { x: 0.38, y: 0.42 };
  landmarks[133] = { x: 0.44, y: 0.42 };
  landmarks[362] = { x: 0.56, y: 0.42 };
  landmarks[263] = { x: 0.62, y: 0.42 };
  return landmarks;
}

test('face overlay transform follows face bounds and eye roll', () => {
  const transform = computeFaceOverlayTransform(syntheticFace(), 1000, 800);
  assert.ok(transform);
  assert.equal(Math.round(transform.x), 500);
  assert.equal(Math.round(transform.width), 544);
  assert.equal(Math.round(transform.height), 643);
  assert.ok(Math.abs(transform.rotation) < 1e-9);
  assert.ok(transform.y < 400);
});

test('face overlay rotation follows tilted eyes', () => {
  const landmarks = syntheticFace();
  landmarks[33].y = 0.36;
  landmarks[133].y = 0.36;
  landmarks[362].y = 0.48;
  landmarks[263].y = 0.48;
  const transform = computeFaceOverlayTransform(landmarks, 1000, 1000);
  assert.ok(transform.rotation > 0.3);
  assert.ok(transform.rotation < 0.8);
});

test('smoothing moves partway toward the latest face transform', () => {
  const previous = { x: 0, y: 0, width: 100, height: 100, rotation: 0 };
  const next = { x: 100, y: 50, width: 200, height: 160, rotation: Math.PI / 2 };
  const smoothed = smoothOverlayTransform(previous, next, 0.25);
  assert.equal(smoothed.x, 25);
  assert.equal(smoothed.y, 12.5);
  assert.equal(smoothed.width, 125);
  assert.equal(smoothed.height, 115);
  assert.ok(Math.abs(smoothed.rotation - Math.PI / 8) < 1e-9);
});

test('cover rect fills target without distorting image aspect ratio', () => {
  const rect = computeCoverRect(400, 200, 300, 300);
  assert.equal(rect.width, 600);
  assert.equal(rect.height, 300);
  assert.equal(rect.x, -300);
  assert.equal(rect.y, -150);
});
