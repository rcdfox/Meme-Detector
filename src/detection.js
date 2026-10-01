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
    browUp: Math.max(score('browInnerUp'), avg(score('browOuterUpLeft'), score('browOuterUpRight'))),
    pucker: score('mouthPucker'),
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

export const MEMES = [
  { id: 'scream-cat', title: 'Screaming Cat', asset: '/memes/scream-cat.svg', hint: 'Open mouth + wide eyes, stronger with hands near face.' },
  { id: 'shocked-cat', title: 'Shocked Cat', asset: '/memes/shocked-cat.svg', hint: 'Open mouth and/or very wide eyes.' },
  { id: 'grin-cat', title: 'Grinning Cat', asset: '/memes/grin-cat.svg', hint: 'Smile broadly at the camera.' },
  { id: 'side-eye-cat', title: 'Side-Eye Cat', asset: '/memes/side-eye-cat.svg', hint: 'Look sharply sideways with a skeptical face.' },
  { id: 'thinking-cat', title: 'Thinking Cat', asset: '/memes/thinking-cat.svg', hint: 'Bring a hand near your chin/face.' },
  { id: 'eyebrow-cat', title: 'Suspicious Cat', asset: '/memes/eyebrow-cat.svg', hint: 'Raise your eyebrows and widen your eyes.' },
  { id: 'victory', title: 'Absolute Victory', asset: '/memes/victory.svg', hint: 'Raise both hands above your shoulders.' },
  { id: 'dab', title: 'Dab Detected', asset: '/memes/dab.svg', hint: 'One hand near face, opposite arm extended.' },
  { id: 'flex', title: 'Built Different', asset: '/memes/flex.svg', hint: 'Raise and bend both arms into a flex.' },
  { id: 'shrug', title: 'It Is What It Is', asset: '/memes/shrug.svg', hint: 'Bring both bent arms/hands near shoulder height.' },
  { id: 'crossed-arms', title: 'Not Impressed', asset: '/memes/crossed-arms.svg', hint: 'Cross your arms over your torso.' }
];

const memeById = new Map(MEMES.map((m) => [m.id, m]));

export function rankMemeMatches(face = {}, pose = {}) {
  const f = {
    smile: face.smile ?? 0,
    jawOpen: face.jawOpen ?? 0,
    eyeWide: face.eyeWide ?? 0,
    blink: face.blink ?? 0,
    browUp: face.browUp ?? 0,
    pucker: face.pucker ?? 0,
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

  const rules = [
    ['scream-cat', 0.43 * f.jawOpen + 0.24 * f.eyeWide + 0.33 * clamp01(p.handsNearFace / 2), 0.50, 'open mouth + wide eyes + hands near face'],
    ['shocked-cat', 0.58 * f.jawOpen + 0.42 * f.eyeWide, 0.46, 'open mouth + wide eyes'],
    ['grin-cat', 0.92 * f.smile + 0.08 * (1 - f.frown), 0.48, 'strong smile'],
    ['side-eye-cat', 0.82 * f.sideEye + 0.18 * f.frown, 0.43, 'sideways gaze'],
    ['thinking-cat', 0.62 * p.handNearFace + 0.18 * f.pucker + 0.20 * f.browUp, 0.50, 'hand close to face'],
    ['eyebrow-cat', 0.72 * f.browUp + 0.28 * f.eyeWide, 0.44, 'raised eyebrows + wide eyes'],
    ['victory', 0.90 * p.bothHandsUp + 0.10 * f.smile, 0.60, 'both hands raised'],
    ['dab', 0.96 * p.dab + 0.04 * f.smile, 0.70, 'dab geometry'],
    ['flex', 0.92 * p.flex + 0.08 * f.smile, 0.66, 'both arms flexed'],
    ['shrug', 0.76 * p.shrug + 0.15 * f.browUp + 0.09 * f.frown, 0.62, 'both hands around shoulder height'],
    ['crossed-arms', 0.88 * p.armsCrossed + 0.12 * f.frown, 0.66, 'arms crossed over torso']
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
