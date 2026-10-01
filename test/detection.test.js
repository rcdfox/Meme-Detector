import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseStableMatch, extractFaceFeatures, extractHandFeatures, rankMemeMatches } from '../src/detection.js';

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
  const ranked = rankMemeMatches({ present: true, pucker: 0.2 }, { handNearFace: 1 });
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
