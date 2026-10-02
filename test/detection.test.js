import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseConfidentCandidate, chooseStableMatch, computeFaceBaseline, extractFaceFeatures, extractHandFeatures, normalizeFaceFeatures, rankMemeMatches, smoothFeatureGroup } from '../src/detection.js';

test('extractFaceFeatures averages symmetric smile blendshapes', () => {
  const face = extractFaceFeatures([
    { categoryName: 'mouthSmileLeft', score: 0.8 },
    { categoryName: 'mouthSmileRight', score: 0.6 }
  ]);
  assert.equal(face.smile, 0.7);
});

test('extractFaceFeatures exposes squint and brow-down signals', () => {
  const face = extractFaceFeatures([
    { categoryName: 'eyeSquintLeft', score: 0.6 },
    { categoryName: 'eyeSquintRight', score: 0.8 },
    { categoryName: 'browDownLeft', score: 0.4 },
    { categoryName: 'browDownRight', score: 0.6 }
  ]);
  assert.equal(face.squint, 0.7);
  assert.equal(face.browDown, 0.5);
});

test('wide-eyed closed-mouth expression matches buffering cat', () => {
  const ranked = rankMemeMatches({ present: true, eyeWide: 0.95, jawOpen: 0.02, smile: 0.02 }, {});
  assert.equal(ranked[0].id, 'buffering-cat');
  assert.equal(ranked[0].matched, true);
});

test('index finger near face ranks thinking cat as a match', () => {
  const ranked = rankMemeMatches(
    { present: true, pucker: 0.20 },
    { handNearFace: 1 },
    { indexOnly: 0.92, indexNearFace: 0.88 }
  );
  assert.equal(ranked[0].id, 'thinking-cat');
  assert.equal(ranked[0].matched, true);
});

test('hands-up pose ranks launch cat as a match', () => {
  const ranked = rankMemeMatches({ present: true }, { bothHandsUp: 1 });
  const launch = ranked.find((r) => r.id === 'launch-cat');
  assert.equal(launch.matched, true);
});

test('neutral visible face can match deadpan cat', () => {
  const ranked = rankMemeMatches({ present: true }, {});
  assert.equal(ranked[0].id, 'deadpan-cat');
  assert.equal(ranked[0].matched, true);
});

test('stable match requires repeated confirmation', () => {
  const candidate = { id: 'happy-cat', matched: true };
  let state = { history: [], match: null };
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match, null);
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match, null);
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match.id, 'happy-cat');
});

test('gesture recognizer result exposes thumbs-up confidence', () => {
  const hand = extractHandFeatures({
    landmarks: [Array.from({ length: 21 }, (_, i) => ({ x: i / 100, y: i / 100 }))],
    gestures: [[{ categoryName: 'Thumb_Up', score: 0.91 }]]
  });
  assert.equal(hand.present, true);
  assert.equal(hand.thumbUp, 0.91);
  assert.equal(hand.gestureName, 'Thumb_Up');
});

test('sad face plus thumbs-up ranks crying thumbs-up cat as a match', () => {
  const ranked = rankMemeMatches(
    { present: true, frown: 0.75, browUp: 0.55 },
    { oneHandUp: 1 },
    { thumbUp: 0.92 }
  );
  assert.equal(ranked[0].id, 'crying-thumbs-up-cat');
  assert.equal(ranked[0].matched, true);
});


test('neutral calibration removes a user-specific resting smile', () => {
  const baseline = computeFaceBaseline([
    { present: true, smile: 0.22, jawOpen: 0.03, eyeWide: 0.04 },
    { present: true, smile: 0.24, jawOpen: 0.02, eyeWide: 0.05 },
    { present: true, smile: 0.23, jawOpen: 0.04, eyeWide: 0.04 }
  ]);
  const normalized = normalizeFaceFeatures(
    { present: true, smile: 0.24, jawOpen: 0.03, eyeWide: 0.05 },
    baseline
  );
  assert.ok(normalized.smile < 0.05);
  assert.ok(normalized.jawOpen < 0.05);
});

test('normalization preserves a strong smile above neutral baseline', () => {
  const baseline = computeFaceBaseline([
    { present: true, smile: 0.12, jawOpen: 0.02 },
    { present: true, smile: 0.14, jawOpen: 0.03 },
    { present: true, smile: 0.13, jawOpen: 0.02 }
  ]);
  const normalized = normalizeFaceFeatures(
    { present: true, smile: 0.68, jawOpen: 0.03 },
    baseline
  );
  assert.ok(normalized.smile > 0.8);
});

test('feature smoothing damps single-frame spikes', () => {
  const previous = { smile: 0.1, jawOpen: 0.1 };
  const next = { smile: 0.9, jawOpen: 0.5 };
  const smoothed = smoothFeatureGroup(previous, next, 0.25);
  assert.equal(smoothed.smile, 0.3);
  assert.equal(smoothed.jawOpen, 0.2);
});

test('ambiguous close matches are rejected instead of forced', () => {
  const candidate = chooseConfidentCandidate([
    { id: 'a', matched: true, score: 0.61 },
    { id: 'b', matched: true, score: 0.58 }
  ], { minMargin: 0.09, strongScore: 0.84 });
  assert.equal(candidate, null);
});

test('strong high-confidence match bypasses ambiguity margin', () => {
  const candidate = chooseConfidentCandidate([
    { id: 'a', matched: true, score: 0.90 },
    { id: 'b', matched: true, score: 0.87 }
  ], { minMargin: 0.09, strongScore: 0.84 });
  assert.equal(candidate.id, 'a');
});

test('happy cat requires a clear smile rather than weak background expression', () => {
  const weak = rankMemeMatches({ present: true, smile: 0.24 }, {});
  const happyWeak = weak.find((candidate) => candidate.id === 'happy-cat');
  assert.equal(happyWeak.matched, false);

  const strong = rankMemeMatches({ present: true, smile: 0.82 }, {});
  const happyStrong = strong.find((candidate) => candidate.id === 'happy-cat');
  assert.equal(happyStrong.matched, true);
});


test('thinking cat does not trigger from hand-near-face without an index-finger cue', () => {
  const ranked = rankMemeMatches(
    { present: true, pucker: 0.3 },
    { handNearFace: 1 },
    { openPalm: 0.8 }
  );
  const thinking = ranked.find((candidate) => candidate.id === 'thinking-cat');
  assert.equal(thinking.matched, false);
});

test('silly tongue cat requires an open mouth plus playful hand posture', () => {
  const noHand = rankMemeMatches(
    { present: true, jawOpen: 0.82, smileMax: 0.55, eyeWide: 0.35 },
    {},
    {}
  ).find((candidate) => candidate.id === 'silly-tongue-cat');
  assert.equal(noHand.matched, false);

  const shaka = rankMemeMatches(
    { present: true, jawOpen: 0.82, smileMax: 0.55, eyeWide: 0.35 },
    {},
    { shaka: 1 }
  ).find((candidate) => candidate.id === 'silly-tongue-cat');
  assert.equal(shaka.matched, true);
});

test('crying cat responds to inner-brow raise plus sad mouth', () => {
  const ranked = rankMemeMatches({
    present: true,
    browInnerUp: 0.78,
    frownMax: 0.65,
    lowerDown: 0.50,
    mouthShrugLower: 0.35,
    smileMax: 0.03
  }, {}, {});
  assert.equal(ranked[0].id, 'crying-cat');
  assert.equal(ranked[0].matched, true);
});

test('concerned cat responds to furrowed brows and tense closed mouth', () => {
  const ranked = rankMemeMatches({
    present: true,
    browDown: 0.72,
    press: 0.55,
    frownMax: 0.30,
    squintMax: 0.22,
    jawOpen: 0.04,
    smileMax: 0.03,
    browInnerUp: 0.05
  }, {}, {});
  assert.equal(ranked[0].id, 'concerned-cat');
  assert.equal(ranked[0].matched, true);
});

test('smug cat recognizes an asymmetric smirk', () => {
  const ranked = rankMemeMatches({
    present: true,
    smileMax: 0.66,
    smile: 0.38,
    smileAsymmetry: 0.58,
    squintMax: 0.42,
    squintAsymmetry: 0.24,
    sideEye: 0.20,
    jawOpen: 0.03
  }, {}, {});
  assert.equal(ranked[0].id, 'smug-cat');
  assert.equal(ranked[0].matched, true);
});

test('crying thumbs-up cat accepts strong thumbs-up plus a modest sad cue', () => {
  const ranked = rankMemeMatches(
    { present: true, browInnerUp: 0.35, frownMax: 0.22, lowerDown: 0.18 },
    { oneHandUp: 1 },
    { thumbUp: 0.88 }
  );
  assert.equal(ranked[0].id, 'crying-thumbs-up-cat');
  assert.equal(ranked[0].matched, true);
});


function syntheticHandBase() {
  return Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.78 }));
}

function syntheticFaceLandmarks() {
  const face = Array.from({ length: 468 }, () => ({ x: 0.5, y: 0.5 }));
  face[1] = { x: 0.5, y: 0.45 };
  face[13] = { x: 0.5, y: 0.53 };
  face[14] = { x: 0.5, y: 0.55 };
  face[152] = { x: 0.5, y: 0.68 };
  face[10] = { x: 0.4, y: 0.30 };
  face[338] = { x: 0.6, y: 0.30 };
  return face;
}

test('hand geometry detects an isolated index finger near the face', () => {
  const hand = syntheticHandBase();
  hand[0] = { x: 0.5, y: 0.82 };
  hand[5] = { x: 0.5, y: 0.66 };
  hand[6] = { x: 0.5, y: 0.56 };
  hand[8] = { x: 0.5, y: 0.40 };
  hand[9] = { x: 0.54, y: 0.67 };
  hand[10] = { x: 0.54, y: 0.72 };
  hand[12] = { x: 0.54, y: 0.77 };
  hand[13] = { x: 0.58, y: 0.68 };
  hand[14] = { x: 0.58, y: 0.73 };
  hand[16] = { x: 0.58, y: 0.78 };
  hand[17] = { x: 0.62, y: 0.69 };
  hand[18] = { x: 0.62, y: 0.74 };
  hand[20] = { x: 0.62, y: 0.79 };
  hand[1] = { x: 0.44, y: 0.70 };
  hand[3] = { x: 0.45, y: 0.74 };
  hand[4] = { x: 0.46, y: 0.78 };

  const features = extractHandFeatures(
    { landmarks: [hand], gestures: [[]] },
    syntheticFaceLandmarks()
  );
  assert.equal(features.indexOnly, 1);
  assert.ok(features.indexNearFace > 0.55);
});

test('hand geometry detects a shaka/call-me pose', () => {
  const hand = syntheticHandBase();
  hand[0] = { x: 0.5, y: 0.82 };
  hand[1] = { x: 0.43, y: 0.70 };
  hand[3] = { x: 0.36, y: 0.62 };
  hand[4] = { x: 0.25, y: 0.52 };
  hand[5] = { x: 0.47, y: 0.68 };
  hand[6] = { x: 0.48, y: 0.73 };
  hand[8] = { x: 0.49, y: 0.78 };
  hand[9] = { x: 0.52, y: 0.68 };
  hand[10] = { x: 0.52, y: 0.73 };
  hand[12] = { x: 0.52, y: 0.78 };
  hand[13] = { x: 0.57, y: 0.68 };
  hand[14] = { x: 0.57, y: 0.73 };
  hand[16] = { x: 0.57, y: 0.78 };
  hand[17] = { x: 0.62, y: 0.68 };
  hand[18] = { x: 0.66, y: 0.57 };
  hand[20] = { x: 0.75, y: 0.40 };

  const features = extractHandFeatures({ landmarks: [hand], gestures: [[]] });
  assert.equal(features.shaka, 1);
  assert.equal(features.fingerCount, 2);
});
