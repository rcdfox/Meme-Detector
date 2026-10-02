const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
const avg = (...values) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);

function categoryMap(categories = []) {
  return new Map(categories.map((c) => [c.categoryName ?? c.displayName, c.score ?? 0]));
}

export function extractFaceFeatures(categories = []) {
  const m = categoryMap(categories);
  const score = (name) => clamp01(m.get(name) ?? 0);
  const smileLeft = score('mouthSmileLeft');
  const smileRight = score('mouthSmileRight');
  const squintLeft = score('eyeSquintLeft');
  const squintRight = score('eyeSquintRight');
  const frownLeft = score('mouthFrownLeft');
  const frownRight = score('mouthFrownRight');
  const lowerDownLeft = score('mouthLowerDownLeft');
  const lowerDownRight = score('mouthLowerDownRight');
  const browInnerUp = score('browInnerUp');

  return {
    present: categories.length > 0,
    smile: avg(smileLeft, smileRight),
    smileMax: Math.max(smileLeft, smileRight),
    smileAsymmetry: Math.abs(smileLeft - smileRight),
    jawOpen: score('jawOpen'),
    eyeWide: avg(score('eyeWideLeft'), score('eyeWideRight')),
    blink: avg(score('eyeBlinkLeft'), score('eyeBlinkRight')),
    squint: avg(squintLeft, squintRight),
    squintMax: Math.max(squintLeft, squintRight),
    squintAsymmetry: Math.abs(squintLeft - squintRight),
    browInnerUp,
    browUp: Math.max(browInnerUp, avg(score('browOuterUpLeft'), score('browOuterUpRight'))),
    browDown: avg(score('browDownLeft'), score('browDownRight')),
    pucker: score('mouthPucker'),
    press: avg(score('mouthPressLeft'), score('mouthPressRight')),
    frown: avg(frownLeft, frownRight),
    frownMax: Math.max(frownLeft, frownRight),
    lowerDown: avg(lowerDownLeft, lowerDownRight),
    mouthShrugLower: score('mouthShrugLower'),
    mouthShrugUpper: score('mouthShrugUpper'),
    sideEye: Math.max(
      score('eyeLookOutLeft'), score('eyeLookOutRight'),
      score('eyeLookInLeft'), score('eyeLookInRight')
    )
  };
}

const FACE_NUMERIC_KEYS = [
  'smile', 'smileMax', 'smileAsymmetry',
  'jawOpen', 'eyeWide', 'blink',
  'squint', 'squintMax', 'squintAsymmetry',
  'browInnerUp', 'browUp', 'browDown',
  'pucker', 'press', 'frown', 'frownMax',
  'lowerDown', 'mouthShrugLower', 'mouthShrugUpper',
  'sideEye'
];

const FACE_RESPONSE_CEILINGS = {
  smile: 0.72,
  smileMax: 0.78,
  smileAsymmetry: 0.42,
  jawOpen: 0.68,
  eyeWide: 0.46,
  blink: 0.58,
  squint: 0.48,
  squintMax: 0.58,
  squintAsymmetry: 0.38,
  browInnerUp: 0.50,
  browUp: 0.52,
  browDown: 0.50,
  pucker: 0.55,
  press: 0.50,
  frown: 0.48,
  frownMax: 0.58,
  lowerDown: 0.52,
  mouthShrugLower: 0.48,
  mouthShrugUpper: 0.48,
  sideEye: 0.55
};

const median = (values) => {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export function computeFaceBaseline(samples = []) {
  const valid = samples.filter((sample) => sample?.present);
  const baseline = { present: valid.length > 0 };
  for (const key of FACE_NUMERIC_KEYS) {
    baseline[key] = median(valid.map((sample) => clamp01(sample?.[key] ?? 0)));
  }
  return baseline;
}

export function normalizeFaceFeatures(face = {}, baseline = {}) {
  if (!face.present) return { ...face, present: false };
  const normalized = { present: true };
  for (const key of FACE_NUMERIC_KEYS) {
    const raw = clamp01(face[key] ?? 0);
    const base = clamp01(baseline[key] ?? 0);
    const ceiling = Math.max(base + 0.12, FACE_RESPONSE_CEILINGS[key] ?? 0.55);
    const noiseFloor = key === 'eyeWide' || key === 'sideEye' ? 0.012 : 0.018;
    normalized[key] = clamp01((raw - base - noiseFloor) / Math.max(0.08, ceiling - base - noiseFloor));
  }
  return normalized;
}

export function smoothFeatureGroup(previous, next = {}, alpha = 0.42) {
  if (!previous) return { ...next };
  const t = Math.max(0, Math.min(1, alpha));
  const result = { ...next };
  for (const [key, value] of Object.entries(next)) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      const oldValue = Number.isFinite(previous[key]) ? previous[key] : value;
      result[key] = oldValue + (value - oldValue) * t;
    }
  }
  return result;
}

const P = {
  nose: 0,
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24
};

const point = (landmarks, i) => landmarks?.[i] ?? null;
const visible = (p) => p && (p.visibility == null || p.visibility >= 0.45);
const dist = (a, b) => a && b ? Math.hypot(a.x - b.x, a.y - b.y) : Infinity;
const angle = (a, b, c) => {
  if (!a || !b || !c) return 180;
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const denom = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y);
  if (!denom) return 180;
  return Math.acos(Math.max(-1, Math.min(1, dot / denom))) * 180 / Math.PI;
};

export function extractPoseFeatures(landmarks = []) {
  const nose = point(landmarks, P.nose);
  const ls = point(landmarks, P.leftShoulder);
  const rs = point(landmarks, P.rightShoulder);
  const le = point(landmarks, P.leftElbow);
  const re = point(landmarks, P.rightElbow);
  const lw = point(landmarks, P.leftWrist);
  const rw = point(landmarks, P.rightWrist);
  const lh = point(landmarks, P.leftHip);
  const rh = point(landmarks, P.rightHip);

  const present = [nose, ls, rs, le, re, lw, rw].filter(visible).length >= 5;
  const shoulderWidth = Math.max(0.08, dist(ls, rs));
  const centerX = ls && rs ? (ls.x + rs.x) / 2 : 0.5;
  const shoulderY = ls && rs ? (ls.y + rs.y) / 2 : 0.4;
  const hipY = lh && rh ? (lh.y + rh.y) / 2 : shoulderY + 0.3;

  const leftUp = visible(lw) && visible(ls) && lw.y < ls.y - 0.035;
  const rightUp = visible(rw) && visible(rs) && rw.y < rs.y - 0.035;
  const leftNearFace = visible(lw) && visible(nose) && dist(lw, nose) < shoulderWidth * 0.82;
  const rightNearFace = visible(rw) && visible(nose) && dist(rw, nose) < shoulderWidth * 0.82;
  const leftElbowAngle = angle(ls, le, lw);
  const rightElbowAngle = angle(rs, re, rw);

  const leftExtended = visible(lw) && visible(ls) && Math.abs(lw.x - ls.x) > shoulderWidth * 0.9 && leftElbowAngle > 135;
  const rightExtended = visible(rw) && visible(rs) && Math.abs(rw.x - rs.x) > shoulderWidth * 0.9 && rightElbowAngle > 135;

  const crossed = visible(lw) && visible(rw) && visible(ls) && visible(rs) &&
    lw.x > centerX && rw.x < centerX &&
    lw.y > shoulderY - 0.08 && lw.y < hipY &&
    rw.y > shoulderY - 0.08 && rw.y < hipY;

  const flex = leftUp && rightUp && leftElbowAngle < 115 && rightElbowAngle < 115 &&
    visible(le) && visible(re) && Math.abs(le.x - centerX) > shoulderWidth * 0.42 && Math.abs(re.x - centerX) > shoulderWidth * 0.42;

  const shrugLeft = visible(lw) && visible(ls) && dist(lw, ls) < shoulderWidth * 0.8 && leftElbowAngle < 125;
  const shrugRight = visible(rw) && visible(rs) && dist(rw, rs) < shoulderWidth * 0.8 && rightElbowAngle < 125;

  const dab = (leftNearFace && rightExtended) || (rightNearFace && leftExtended);

  return {
    present,
    bothHandsUp: leftUp && rightUp ? 1 : 0,
    oneHandUp: leftUp !== rightUp ? 1 : 0,
    handNearFace: leftNearFace || rightNearFace ? 1 : 0,
    handsNearFace: (leftNearFace ? 1 : 0) + (rightNearFace ? 1 : 0),
    armsCrossed: crossed ? 1 : 0,
    flex: flex ? 1 : 0,
    shrug: shrugLeft && shrugRight ? 1 : 0,
    dab: dab ? 1 : 0,
    shoulderTilt: ls && rs ? clamp01(Math.abs(ls.y - rs.y) / 0.18) : 0
  };
}

function gestureScore(result = {}, gestureName) {
  let best = 0;
  for (const handGestures of result.gestures ?? []) {
    for (const gesture of handGestures ?? []) {
      if (gesture?.categoryName === gestureName) best = Math.max(best, clamp01(gesture.score ?? 0));
    }
  }
  return best;
}

function handDistance(a, b) {
  return a && b ? Math.hypot(a.x - b.x, a.y - b.y) : Infinity;
}

function fingerExtended(landmarks, mcpIndex, pipIndex, tipIndex, ratio = 1.12) {
  const wrist = landmarks?.[0];
  const mcp = landmarks?.[mcpIndex];
  const pip = landmarks?.[pipIndex];
  const tip = landmarks?.[tipIndex];
  if (!wrist || !mcp || !pip || !tip) return false;
  return handDistance(tip, wrist) > handDistance(pip, wrist) * ratio &&
    handDistance(tip, mcp) > handDistance(pip, mcp) * 1.02;
}

function handShapeFeatures(landmarks = []) {
  if (landmarks.length < 21) {
    return {
      pinch: 0, fingerCount: 0, indexOnly: 0, shaka: 0, geometricThumbUp: 0,
      thumbExtended: false, indexExtended: false, middleExtended: false,
      ringExtended: false, pinkyExtended: false
    };
  }

  const palmScale = Math.max(0.035, handDistance(landmarks[5], landmarks[17]));
  const pinch = clamp01(1 - handDistance(landmarks[4], landmarks[8]) / (palmScale * 0.55));
  const thumbExtended = fingerExtended(landmarks, 1, 3, 4, 1.03);
  const indexExtended = fingerExtended(landmarks, 5, 6, 8);
  const middleExtended = fingerExtended(landmarks, 9, 10, 12);
  const ringExtended = fingerExtended(landmarks, 13, 14, 16);
  const pinkyExtended = fingerExtended(landmarks, 17, 18, 20);

  const indexOnly = indexExtended && !middleExtended && !ringExtended && !pinkyExtended ? 1 : 0;
  const shaka = thumbExtended && pinkyExtended && !indexExtended && !middleExtended && !ringExtended ? 1 : 0;
  const thumbVertical = landmarks[4].y < landmarks[2].y - palmScale * 0.18 &&
    landmarks[4].y < landmarks[0].y - palmScale * 0.25;
  const geometricThumbUp = thumbExtended && !indexExtended && !middleExtended &&
    !ringExtended && !pinkyExtended && thumbVertical ? 1 : 0;

  return {
    pinch,
    fingerCount: [thumbExtended, indexExtended, middleExtended, ringExtended, pinkyExtended].filter(Boolean).length,
    indexOnly,
    shaka,
    geometricThumbUp,
    thumbExtended,
    indexExtended,
    middleExtended,
    ringExtended,
    pinkyExtended
  };
}

function faceReference(faceLandmarks = []) {
  const valid = faceLandmarks.filter((point) => point && Number.isFinite(point.x) && Number.isFinite(point.y));
  if (valid.length < 4) return null;
  const xs = valid.map((point) => point.x);
  const faceWidth = Math.max(0.06, Math.max(...xs) - Math.min(...xs));
  const anchors = [faceLandmarks[1], faceLandmarks[13], faceLandmarks[14], faceLandmarks[152]]
    .filter((point) => point && Number.isFinite(point.x) && Number.isFinite(point.y));
  return { faceWidth, anchors };
}

export function extractHandFeatures(result = {}, faceLandmarks = []) {
  const hands = result.landmarks ?? [];
  const shapes = hands.map(handShapeFeatures);
  const face = faceReference(faceLandmarks);
  let bestGesture = { name: 'None', score: 0 };
  for (const handGestures of result.gestures ?? []) {
    const top = handGestures?.[0];
    if (top && (top.score ?? 0) > bestGesture.score) {
      bestGesture = { name: top.categoryName ?? 'None', score: clamp01(top.score ?? 0) };
    }
  }

  let indexNearFace = 0;
  if (face) {
    for (const landmarks of hands) {
      const indexTip = landmarks?.[8];
      if (!indexTip) continue;
      const nearest = Math.min(...face.anchors.map((anchor) => handDistance(indexTip, anchor)));
      indexNearFace = Math.max(indexNearFace, clamp01(1 - nearest / (face.faceWidth * 0.72)));
    }
  }

  const builtInThumbUp = gestureScore(result, 'Thumb_Up');
  const geometricThumbUp = Math.max(0, ...shapes.map((shape) => shape.geometricThumbUp));
  const indexOnly = Math.max(0, ...shapes.map((shape) => shape.indexOnly));
  const shaka = Math.max(0, ...shapes.map((shape) => shape.shaka));

  return {
    present: hands.length > 0,
    handCount: hands.length,
    twoHands: hands.length >= 2 ? 1 : 0,
    thumbUp: Math.max(builtInThumbUp, geometricThumbUp * 0.88),
    builtInThumbUp,
    geometricThumbUp,
    thumbDown: gestureScore(result, 'Thumb_Down'),
    pointingUp: gestureScore(result, 'Pointing_Up'),
    indexOnly,
    indexNearFace,
    shaka,
    victory: gestureScore(result, 'Victory'),
    openPalm: gestureScore(result, 'Open_Palm'),
    closedFist: gestureScore(result, 'Closed_Fist'),
    iLoveYou: gestureScore(result, 'ILoveYou'),
    pinch: Math.max(0, ...shapes.map((shape) => shape.pinch)),
    fingerCount: Math.max(0, ...shapes.map((shape) => shape.fingerCount)),
    fingerCountNorm: clamp01(Math.max(0, ...shapes.map((shape) => shape.fingerCount)) / 5),
    gestureName: bestGesture.name,
    gestureScore: bestGesture.score
  };
}

export const MEMES = [
  { id: 'deadpan-cat', title: 'Deadpan Cat', asset: '/memes/deadpan-cat.webp', hint: 'Hold a neutral, straight-faced stare at the camera.' },
  { id: 'silly-tongue-cat', title: 'Silly Tongue Cat', asset: '/memes/silly-tongue-cat.webp', hint: 'Open your mouth and make a shaka/call-me hand pose near the camera.' },
  { id: 'crying-cat', title: 'Crying Cat', asset: '/memes/crying-cat.webp', hint: 'Raise the inner brows and make an exaggerated sad mouth.' },
  { id: 'launch-cat', title: 'Launch Cat', asset: '/memes/launch-cat.webp', hint: 'Throw both hands up above your shoulders.' },
  { id: 'nerd-cat', title: 'Nerd Cat', asset: '/memes/nerd-cat.webp', hint: 'Raise your eyebrows and open your eyes wide.' },
  { id: 'happy-cat', title: 'Happy Cat', asset: '/memes/happy-cat.webp', hint: 'Give the camera a bright smile with open eyes.' },
  { id: 'tired-cat', title: 'Tired Cat', asset: '/memes/tired-cat.webp', hint: 'Squint while opening your mouth or making a tired face.' },
  { id: 'smug-cat', title: 'Smug Cat', asset: '/memes/smug-cat.webp', hint: 'Make a one-sided smirk with a squint or sideways glance.' },
  { id: 'buffering-cat', title: 'Buffering Cat', asset: '/memes/buffering-cat.webp', hint: 'Freeze with very wide eyes and a mostly closed mouth.' },
  { id: 'judging-cat', title: 'Judging Cat', asset: '/memes/judging-cat.webp', hint: 'Look sharply sideways and squint.' },
  { id: 'thinking-cat', title: 'Thinking Cat', asset: '/memes/thinking-cat.webp', hint: 'Hold one index finger near your mouth or chin in a thinking pose.' },
  { id: 'concerned-cat', title: 'Concerned Cat', asset: '/memes/concerned-cat.webp', hint: 'Furrow your brows and press/frown your mouth without opening it.' },
  { id: 'crying-thumbs-up-cat', title: 'Crying Thumbs-Up Cat', asset: '/memes/crying-thumbs-up-cat.webp', hint: 'Make a sad face and give the camera a thumbs-up.' }
];

const memeById = new Map(MEMES.map((m) => [m.id, m]));

export function rankMemeMatches(face = {}, pose = {}, hand = {}) {
  const f = {
    present: face.present ?? false,
    smile: clamp01(face.smile ?? 0),
    smileMax: clamp01(face.smileMax ?? face.smile ?? 0),
    smileAsymmetry: clamp01(face.smileAsymmetry ?? 0),
    jawOpen: clamp01(face.jawOpen ?? 0),
    eyeWide: clamp01(face.eyeWide ?? 0),
    blink: clamp01(face.blink ?? 0),
    squint: clamp01(face.squint ?? 0),
    squintMax: clamp01(face.squintMax ?? face.squint ?? 0),
    squintAsymmetry: clamp01(face.squintAsymmetry ?? 0),
    browInnerUp: clamp01(face.browInnerUp ?? face.browUp ?? 0),
    browUp: clamp01(face.browUp ?? 0),
    browDown: clamp01(face.browDown ?? 0),
    pucker: clamp01(face.pucker ?? 0),
    press: clamp01(face.press ?? 0),
    frown: clamp01(face.frown ?? 0),
    frownMax: clamp01(face.frownMax ?? face.frown ?? 0),
    lowerDown: clamp01(face.lowerDown ?? 0),
    mouthShrugLower: clamp01(face.mouthShrugLower ?? 0),
    mouthShrugUpper: clamp01(face.mouthShrugUpper ?? 0),
    sideEye: clamp01(face.sideEye ?? 0)
  };
  const p = {
    bothHandsUp: clamp01(pose.bothHandsUp ?? 0),
    oneHandUp: clamp01(pose.oneHandUp ?? 0),
    handNearFace: clamp01(pose.handNearFace ?? 0),
    handsNearFace: clamp01((pose.handsNearFace ?? 0) / 2),
    armsCrossed: clamp01(pose.armsCrossed ?? 0),
    flex: clamp01(pose.flex ?? 0),
    shrug: clamp01(pose.shrug ?? 0),
    dab: clamp01(pose.dab ?? 0)
  };
  const h = {
    thumbUp: clamp01(hand.thumbUp ?? 0),
    builtInThumbUp: clamp01(hand.builtInThumbUp ?? hand.thumbUp ?? 0),
    geometricThumbUp: clamp01(hand.geometricThumbUp ?? 0),
    thumbDown: clamp01(hand.thumbDown ?? 0),
    pointingUp: clamp01(hand.pointingUp ?? 0),
    indexOnly: clamp01(hand.indexOnly ?? 0),
    indexNearFace: clamp01(hand.indexNearFace ?? 0),
    shaka: clamp01(hand.shaka ?? 0),
    victory: clamp01(hand.victory ?? 0),
    openPalm: clamp01(hand.openPalm ?? 0),
    closedFist: clamp01(hand.closedFist ?? 0),
    iLoveYou: clamp01(hand.iLoveYou ?? 0),
    pinch: clamp01(hand.pinch ?? 0),
    fingerCountNorm: clamp01(hand.fingerCountNorm ?? 0)
  };

  const low = (value, ceiling) => clamp01((ceiling - value) / Math.max(ceiling, 0.001));
  const evidence = (...values) => Math.max(...values.map(clamp01));
  const faceActivity = evidence(
    f.smile, f.smileMax, f.smileAsymmetry,
    f.jawOpen, f.eyeWide, f.squint, f.squintMax, f.squintAsymmetry,
    f.browInnerUp, f.browUp, f.browDown,
    f.pucker, f.press, f.frown, f.frownMax, f.lowerDown,
    f.mouthShrugLower, f.mouthShrugUpper, f.sideEye
  );

  const rules = [
    {
      id: 'crying-thumbs-up-cat',
      required: h.thumbUp >= 0.46 &&
        evidence(f.frownMax, f.browInnerUp, f.lowerDown, f.mouthShrugLower) >= 0.16,
      score: 0.66 * h.thumbUp + 0.10 * f.frownMax + 0.10 * f.browInnerUp +
        0.08 * f.lowerDown + 0.06 * p.oneHandUp,
      threshold: 0.50,
      reason: 'thumbs-up + visible sad-face cue'
    },
    {
      id: 'launch-cat',
      required: p.bothHandsUp >= 0.64,
      score: 0.78 * p.bothHandsUp + 0.14 * h.openPalm + 0.08 * f.eyeWide,
      threshold: 0.62,
      reason: 'both hands clearly raised'
    },
    {
      id: 'thinking-cat',
      required: evidence(p.handNearFace, h.indexNearFace) >= 0.46 &&
        evidence(h.indexOnly, h.pointingUp) >= 0.46,
      score: 0.34 * h.indexNearFace + 0.26 * h.indexOnly + 0.18 * h.pointingUp +
        0.12 * p.handNearFace + 0.06 * f.pucker + 0.04 * f.sideEye,
      threshold: 0.48,
      reason: 'extended index finger held near mouth / chin'
    },
    {
      id: 'silly-tongue-cat',
      required: f.present && f.jawOpen >= 0.34 &&
        evidence(h.shaka, h.iLoveYou, h.openPalm * 0.55) >= 0.24,
      score: 0.40 * f.jawOpen + 0.12 * f.smileMax + 0.08 * f.eyeWide +
        0.30 * h.shaka + 0.06 * h.iLoveYou + 0.04 * h.openPalm,
      threshold: 0.46,
      reason: 'open mouth + shaka / playful hand pose'
    },
    {
      id: 'judging-cat',
      required: f.present && f.sideEye >= 0.40 && f.squint >= 0.18,
      score: 0.58 * f.sideEye + 0.28 * f.squint + 0.14 * f.frown,
      threshold: 0.52,
      reason: 'clear sideways gaze + squint'
    },
    {
      id: 'tired-cat',
      required: f.present && f.squint >= 0.38 &&
        evidence(f.jawOpen, f.blink, f.frown) >= 0.22 && f.smile < 0.32,
      score: 0.46 * f.squint + 0.24 * f.jawOpen + 0.18 * f.blink + 0.12 * f.frown,
      threshold: 0.51,
      reason: 'squint + tired mouth / blink'
    },
    {
      id: 'concerned-cat',
      required: f.present && f.jawOpen < 0.26 && f.smileMax < 0.30 &&
        f.browDown >= 0.24 && evidence(f.press, f.frownMax, f.squintMax) >= 0.16,
      score: 0.42 * f.browDown + 0.22 * f.press + 0.18 * f.frownMax +
        0.10 * f.squintMax + 0.08 * low(f.jawOpen, 0.30),
      threshold: 0.44,
      reason: 'furrowed brows + tense closed mouth'
    },
    {
      id: 'crying-cat',
      required: f.present && f.smileMax < 0.28 &&
        f.browInnerUp >= 0.20 &&
        evidence(f.frownMax, f.lowerDown, f.mouthShrugLower, f.press) >= 0.18,
      score: 0.30 * f.browInnerUp + 0.26 * f.frownMax + 0.18 * f.lowerDown +
        0.12 * f.mouthShrugLower + 0.08 * f.press + 0.06 * low(f.smileMax, 0.35),
      threshold: 0.43,
      reason: 'inner brows raised + sad mouth shape'
    },
    {
      id: 'buffering-cat',
      required: f.present && f.eyeWide >= 0.46 && f.jawOpen < 0.28 && f.smile < 0.28,
      score: 0.64 * f.eyeWide + 0.20 * low(f.jawOpen, 0.35) + 0.16 * low(f.smile, 0.35),
      threshold: 0.56,
      reason: 'very wide eyes + frozen mouth'
    },
    {
      id: 'nerd-cat',
      required: f.present && f.eyeWide >= 0.34 && f.browUp >= 0.34,
      score: 0.46 * f.eyeWide + 0.42 * f.browUp + 0.12 * f.smile,
      threshold: 0.53,
      reason: 'wide eyes + raised brows'
    },
    {
      id: 'smug-cat',
      required: f.present && f.jawOpen < 0.28 &&
        f.smileMax >= 0.22 &&
        evidence(f.smileAsymmetry, f.squintAsymmetry, f.sideEye) >= 0.16,
      score: 0.30 * f.smileMax + 0.28 * f.smileAsymmetry + 0.18 * f.squintMax +
        0.12 * f.squintAsymmetry + 0.12 * f.sideEye,
      threshold: 0.42,
      reason: 'one-sided smirk + squint / sideways glance'
    },
    {
      id: 'happy-cat',
      required: f.present && f.smile >= 0.48 && f.frown < 0.24 && f.browDown < 0.30,
      score: 0.72 * f.smile + 0.16 * low(f.frown, 0.45) + 0.12 * low(f.browDown, 0.45),
      threshold: 0.58,
      reason: 'clear sustained smile'
    },
    {
      id: 'deadpan-cat',
      required: f.present && faceActivity < 0.23 && f.jawOpen < 0.14 && f.smile < 0.14,
      score: 0.78 * low(faceActivity, 0.30) + 0.12 * low(f.jawOpen, 0.25) + 0.10 * low(f.smile, 0.25),
      threshold: 0.70,
      reason: 'stable neutral expression'
    }
  ];

  return rules
    .map(({ id, required, score, threshold, reason }) => ({
      ...memeById.get(id),
      score: clamp01(score),
      threshold,
      reason,
      matched: Boolean(required) && score >= threshold
    }))
    .sort((a, b) => (b.matched - a.matched) || (b.score - a.score));
}

export function chooseConfidentCandidate(ranked = [], options = {}) {
  const top = ranked[0];
  if (!top?.matched) return null;
  const second = ranked.find((candidate, index) => index > 0 && candidate.matched);
  if (!second) return top;

  const minMargin = options.minMargin ?? 0.075;
  const strongScore = options.strongScore ?? 0.82;
  if (top.score >= strongScore) return top;
  return top.score - second.score >= minMargin ? top : null;
}

export function chooseStableMatch(history, candidate, options = {}) {
  const minFrames = options.minFrames ?? 3;
  const maxHistory = options.maxHistory ?? 5;
  const next = [...history, candidate?.id ?? null].slice(-maxHistory);
  if (!candidate?.matched) return { history: next, match: null };
  const confirmations = next.filter((id) => id === candidate.id).length;
  return { history: next, match: confirmations >= minFrames ? candidate : null };
}
