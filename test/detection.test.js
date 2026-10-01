import test from 'node:test';
import assert from 'node:assert/strict';
import { chooseStableMatch, extractFaceFeatures, rankMemeMatches } from '../src/detection.js';

test('extractFaceFeatures averages symmetric smile blendshapes', () => {
  const face = extractFaceFeatures([
    { categoryName: 'mouthSmileLeft', score: 0.8 },
    { categoryName: 'mouthSmileRight', score: 0.6 }
  ]);
  assert.equal(face.smile, 0.7);
});

test('strong smile ranks grinning cat as a match', () => {
  const [top] = rankMemeMatches({ smile: 0.9 }, {});
  assert.equal(top.id, 'grin-cat');
  assert.equal(top.matched, true);
});

test('hands-up pose ranks victory as a match', () => {
  const ranked = rankMemeMatches({}, { bothHandsUp: 1 });
  const victory = ranked.find((r) => r.id === 'victory');
  assert.equal(victory.matched, true);
});

test('stable match requires repeated confirmation', () => {
  const candidate = { id: 'grin-cat', matched: true };
  let state = { history: [], match: null };
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match, null);
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match, null);
  state = chooseStableMatch(state.history, candidate, { minFrames: 3, maxHistory: 5 });
  assert.equal(state.match.id, 'grin-cat');
});
