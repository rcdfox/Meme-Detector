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

test('hand near face ranks thinking cat as a match', () => {
  const ranked = rankMemeMatches({ present: true, pucker: 0.35 }, { handNearFace: 1 });
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
