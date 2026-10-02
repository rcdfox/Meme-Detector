const finitePoint = (point) => point && Number.isFinite(point.x) && Number.isFinite(point.y);

const midpoint = (a, b) => {
  if (!finitePoint(a) || !finitePoint(b)) return null;
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
};

const normalizeAngle = (angle) => {
  let value = angle;
  while (value > Math.PI) value -= Math.PI * 2;
  while (value < -Math.PI) value += Math.PI * 2;
  return value;
};

export function computeFaceOverlayTransform(landmarks = [], canvasWidth = 1, canvasHeight = 1, options = {}) {
  const points = landmarks.filter(finitePoint);
  if (points.length < 4 || canvasWidth <= 0 || canvasHeight <= 0) return null;

  const minX = Math.min(...points.map((point) => point.x));
  const maxX = Math.max(...points.map((point) => point.x));
  const minY = Math.min(...points.map((point) => point.y));
  const maxY = Math.max(...points.map((point) => point.y));

  const faceWidth = (maxX - minX) * canvasWidth;
  const faceHeight = (maxY - minY) * canvasHeight;
  if (faceWidth < 2 || faceHeight < 2) return null;

  const widthScale = options.widthScale ?? 1.36;
  const heightScale = options.heightScale ?? 1.34;
  const verticalShift = options.verticalShift ?? -0.035;

  const leftEye = midpoint(landmarks[33], landmarks[133]);
  const rightEye = midpoint(landmarks[362], landmarks[263]);
  let rotation = 0;
  if (leftEye && rightEye) {
    const dx = (rightEye.x - leftEye.x) * canvasWidth;
    const dy = (rightEye.y - leftEye.y) * canvasHeight;
    if (Math.hypot(dx, dy) > 1) rotation = Math.atan2(dy, dx);
  }

  return {
    x: ((minX + maxX) / 2) * canvasWidth,
    y: ((minY + maxY) / 2) * canvasHeight + faceHeight * verticalShift,
    width: faceWidth * widthScale,
    height: faceHeight * heightScale,
    rotation
  };
}

export function smoothOverlayTransform(previous, next, alpha = 0.32) {
  if (!next) return null;
  if (!previous) return { ...next };
  const t = Math.max(0, Math.min(1, alpha));
  const angleDelta = normalizeAngle(next.rotation - previous.rotation);
  return {
    x: previous.x + (next.x - previous.x) * t,
    y: previous.y + (next.y - previous.y) * t,
    width: previous.width + (next.width - previous.width) * t,
    height: previous.height + (next.height - previous.height) * t,
    rotation: normalizeAngle(previous.rotation + angleDelta * t)
  };
}

export function computeCoverRect(imageWidth, imageHeight, targetWidth, targetHeight) {
  if (imageWidth <= 0 || imageHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) return null;
  const scale = Math.max(targetWidth / imageWidth, targetHeight / imageHeight);
  const width = imageWidth * scale;
  const height = imageHeight * scale;
  return {
    x: -width / 2,
    y: -height / 2,
    width,
    height
  };
}
