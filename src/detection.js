const clamp01 = (v) => Math.max(0, Math.min(1, Number.isFinite(v) ? v : 0));
const avg = (...values) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);

function categoryMap(categories = []) {
  return new Map(categories.map((c) => [c.categoryName ?? c.displayName, c.score ?? 0]));
}

export function extractFaceFeatures(categories = []) {
  const m = categoryMap(categories);
  const score = (name) => clamp01(m.get(name) ?? 0);
  return {
    present: categories.length > 0,
    smile: avg(score('mouthSmileLeft'), score('mouthSmileRight')),
    jawOpen: score('jawOpen'),
    eyeWide: avg(score('eyeWideLeft'), score('eyeWideRight')),
    blink: avg(score('eyeBlinkLeft'), score('eyeBlinkRight')),
    squint: avg(score('eyeSquintLeft'), score('eyeSquintRight')),
    browUp: Math.max(score('browInnerUp'), avg(score('browOuterUpLeft'), score('browOuterUpRight'))),
    browDown: avg(score('browDownLeft'), score('browDownRight')),
    pucker: score('mouthPucker'),
    press: avg(score('mouthPressLeft'), score('mouthPressRight')),
    frown: avg(score('mouthFrownLeft'), score('mouthFrownRight')),
    sideEye: Math.max(
      score('eyeLookOutLeft'), score('eyeLookOutRight'),
      score('eyeLookInLeft'), score('eyeLookInRight')
    )
  };
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
  if (landmarks.length < 21) return { pinch: 0, fingerCount: 0 };
  const palmScale = Math.max(0.035, handDistance(landmarks[5], landmarks[17]));
  const pinch = clamp01(1 - handDistance(landmarks[4], landmarks[8]) / (palmScale * 0.55));
  const extended = [
    fingerExtended(landmarks, 1, 3, 4, 1.06),
    fingerExtended(landmarks, 5, 6, 8),
    fingerExtended(landmarks, 9, 10, 12),
    fingerExtended(landmarks, 13, 14, 16),
    fingerExtended(landmarks, 17, 18, 20)
  ];
  return { pinch, fingerCount: extended.filter(Boolean).length };
}

export function extractHandFeatures(result = {}) {
  const hands = result.landmarks ?? [];
  const shapes = hands.map(handShapeFeatures);
  let bestGesture = { name: 'None', score: 0 };
  for (const handGestures of result.gestures ?? []) {
    const top = handGestures?.[0];
    if (top && (top.score ?? 0) > bestGesture.score) {
      bestGesture = { name: top.categoryName ?? 'None', score: clamp01(top.score ?? 0) };
    }
  }

  return {
    present: hands.length > 0,
    handCount: hands.length,
    twoHands: hands.length >= 2 ? 1 : 0,
    thumbUp: gestureScore(result, 'Thumb_Up'),
    thumbDown: gestureScore(result, 'Thumb_Down'),
    pointingUp: gestureScore(result, 'Pointing_Up'),
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
  { id: 'silly-tongue-cat', title: 'Silly Tongue Cat', asset: '/memes/silly-tongue-cat.webp', hint: 'Open your mouth while smiling or looking excited.' },
  { id: 'crying-cat', title: 'Crying Cat', asset: '/memes/crying-cat.webp', hint: 'Make an exaggerated sad or worried face.' },
  { id: 'launch-cat', title: 'Launch Cat', asset: '/memes/launch-cat.webp', hint: 'Throw both hands up above your shoulders.' },
  { id: 'nerd-cat', title: 'Nerd Cat', asset: '/memes/nerd-cat.webp', hint: 'Raise your eyebrows and open your eyes wide.' },
  { id: 'happy-cat', title: 'Happy Cat', asset: '/memes/happy-cat.webp', hint: 'Give the camera a bright smile with open eyes.' },
  { id: 'tired-cat', title: 'Tired Cat', asset: '/memes/tired-cat.webp', hint: 'Squint while opening your mouth or making a tired face.' },
  { id: 'smug-cat', title: 'Smug Cat', asset: '/memes/smug-cat.webp', hint: 'Give a small smile with a slight squint or sideways glance.' },
  { id: 'buffering-cat', title: 'Buffering Cat', asset: '/memes/buffering-cat.webp', hint: 'Freeze with very wide eyes and a mostly closed mouth.' },
  { id: 'judging-cat', title: 'Judging Cat', asset: '/memes/judging-cat.webp', hint: 'Look sharply sideways and squint.' },
  { id: 'thinking-cat', title: 'Thinking Cat', asset: '/memes/thinking-cat.webp', hint: 'Bring one hand near your chin or mouth; pointing with your index finger helps.' },
  { id: 'concerned-cat', title: 'Concerned Cat', asset: '/memes/concerned-cat.webp', hint: 'Frown, lower your brows, or make an unimpressed face.' },
  { id: 'crying-thumbs-up-cat', title: 'Crying Thumbs-Up Cat', asset: '/memes/crying-thumbs-up-cat.webp', hint: 'Make a sad face and give the camera a thumbs-up.' }
];

const memeById = new Map(MEMES.map((m) => [m.id, m]));

export function rankMemeMatches(face = {}, pose = {}, hand = {}) {
  const f = {
    present: face.present ?? false,
    smile: face.smile ?? 0,
    jawOpen: face.jawOpen ?? 0,
    eyeWide: face.eyeWide ?? 0,
    blink: face.blink ?? 0,
    squint: face.squint ?? 0,
    browUp: face.browUp ?? 0,
    browDown: face.browDown ?? 0,
    pucker: face.pucker ?? 0,
    press: face.press ?? 0,
    frown: face.frown ?? 0,
    sideEye: face.sideEye ?? 0
  };
  const p = {
    bothHandsUp: pose.bothHandsUp ?? 0,
    oneHandUp: pose.oneHandUp ?? 0,
    handNearFace: pose.handNearFace ?? 0,
    handsNearFace: pose.handsNearFace ?? 0,
    armsCrossed: pose.armsCrossed ?? 0,
    flex: pose.flex ?? 0,
    shrug: pose.shrug ?? 0,
    dab: pose.dab ?? 0
  };
  const h = {
    thumbUp: hand.thumbUp ?? 0,
    thumbDown: hand.thumbDown ?? 0,
    pointingUp: hand.pointingUp ?? 0,
    victory: hand.victory ?? 0,
    openPalm: hand.openPalm ?? 0,
    closedFist: hand.closedFist ?? 0,
    iLoveYou: hand.iLoveYou ?? 0,
    pinch: hand.pinch ?? 0,
    fingerCountNorm: hand.fingerCountNorm ?? 0
  };

  const expressionActivity = Math.max(
    f.smile, f.jawOpen, f.eyeWide, f.browUp, f.browDown,
    f.pucker, f.frown, f.sideEye, f.squint
  );
  const neutral = f.present ? clamp01(1 - expressionActivity) : 0;
  const mouthClosed = clamp01(1 - f.jawOpen);
  const notSmiling = clamp01(1 - f.smile);

  const rules = [
    ['thinking-cat', 0.46 * p.handNearFace + 0.24 * h.pointingUp + 0.12 * h.pinch + 0.10 * f.pucker + 0.08 * f.sideEye, 0.48, 'hand near face + pointing / thoughtful expression'],
    ['crying-thumbs-up-cat', 0.56 * h.thumbUp + 0.24 * f.frown + 0.12 * f.browUp + 0.08 * p.oneHandUp, 0.44, 'sad expression + thumbs-up gesture'],
    ['launch-cat', 0.82 * p.bothHandsUp + 0.10 * h.openPalm + 0.08 * f.eyeWide, 0.60, 'both hands raised'],
    ['silly-tongue-cat', 0.48 * f.jawOpen + 0.30 * f.smile + 0.22 * f.eyeWide, 0.45, 'open mouth + smile + wide eyes'],
    ['tired-cat', 0.40 * f.squint + 0.28 * f.jawOpen + 0.20 * f.frown + 0.12 * f.blink, 0.38, 'squint + open mouth / tired expression'],
    ['judging-cat', 0.52 * f.sideEye + 0.28 * f.squint + 0.20 * f.frown, 0.38, 'side-eye + squint'],
    ['crying-cat', 0.48 * f.frown + 0.30 * f.browUp + 0.22 * notSmiling, 0.40, 'sad / worried expression'],
    ['concerned-cat', 0.42 * f.browDown + 0.30 * f.frown + 0.18 * f.squint + 0.10 * f.press, 0.36, 'lowered brows + frown'],
    ['buffering-cat', 0.58 * f.eyeWide + 0.24 * mouthClosed + 0.18 * notSmiling, 0.48, 'wide eyes + frozen mouth'],
    ['nerd-cat', 0.42 * f.eyeWide + 0.38 * f.browUp + 0.20 * f.smile, 0.43, 'wide eyes + raised brows'],
    ['happy-cat', 0.62 * f.smile + 0.23 * f.eyeWide + 0.15 * mouthClosed, 0.43, 'bright smile + open eyes'],
    ['smug-cat', 0.50 * f.smile + 0.27 * f.squint + 0.23 * f.sideEye, 0.40, 'small smile + squint / sideways glance'],
    ['deadpan-cat', 0.74 * neutral + 0.16 * f.press + 0.10 * mouthClosed, 0.72, 'neutral straight-faced stare']
  ];

  return rules
    .map(([id, score, threshold, reason]) => ({
      ...memeById.get(id),
      score: clamp01(score),
      threshold,
      reason,
      matched: score >= threshold
    }))
    .sort((a, b) => (b.matched - a.matched) || (b.score - a.score));
}

export function chooseStableMatch(history, candidate, options = {}) {
  const minFrames = options.minFrames ?? 3;
  const maxHistory = options.maxHistory ?? 5;
  const next = [...history, candidate?.id ?? null].slice(-maxHistory);
  if (!candidate?.matched) return { history: next, match: null };
  const confirmations = next.filter((id) => id === candidate.id).length;
  return { history: next, match: confirmations >= minFrames ? candidate : null };
}
