// Plain-node checks for patrol_math.js (no test framework in this package):
//     node capture/mineflayer-bot/test_patrol_math.js
// Each case is one of the failures patrol v1 actually produced.

'use strict';

const assert = require('assert');
const pm = require('./patrol_math');

// A world window for exposure checks: everything is air except the given cells.
function airExcept(cells) {
  const solid = new Set(cells.map(([x, y, z]) => `${x},${y},${z}`));
  return (x, y, z) => !solid.has(`${x},${y},${z}`);
}

// --- roof cluster -> vantage gains height (the old code deleted the y axis) -----
{
  const roof = [];
  for (let x = 0; x < 5; x++) for (let z = 0; z < 5; z++) roof.push([x, 70, z]);
  const clusters = pm.clusterUnseen(roof);
  assert.strictEqual(clusters.length, 1);
  const exposure = pm.classifyExposure(clusters[0], airExcept(roof));
  assert.strictEqual(exposure, 'up');
  const vantage = pm.vantageFor(clusters[0], exposure, [2.5, 65, 2.5], [10, 64, 10]);
  assert.ok(vantage.pos[1] > 71, `roof vantage must be above the plane, got y=${vantage.pos[1]}`);
  assert.ok(vantage.onBuildLikely, 'high vantages are gated by the at-rest climb rule');
  console.log('ok: roof cluster -> elevated vantage');
}

// --- wall cluster -> horizontal outside vantage (v1 geometry, kept) --------------
{
  const wall = [];
  for (let x = 0; x < 7; x++) for (let y = 64; y < 67; y++) wall.push([x, y, 0]);
  const clusters = pm.clusterUnseen(wall);
  const exposure = pm.classifyExposure(clusters[0], airExcept(wall));
  assert.strictEqual(exposure, 'side');
  const vantage = pm.vantageFor(clusters[0], exposure, [3, 65, 5], [0, 64, 0]);
  assert.ok(Math.abs(vantage.pos[1] - clusters[0].centroid[1]) < 2, 'side vantage stays level');
  assert.ok(vantage.pos[2] < -3, 'side vantage stands OUTSIDE the wall face');
  assert.ok(!vantage.onBuildLikely);
  console.log('ok: wall cluster -> horizontal outside vantage');
}

// --- ceiling underside (enclosed room) -> stand beneath, look up -----------------
{
  const ceiling = [];
  for (let x = 1; x < 4; x++) for (let z = 1; z < 4; z++) ceiling.push([x, 68, z]);
  // enclosed: solid above (roof blocks themselves stacked), air only below
  const world = (x, y, z) => y < 68;   // air strictly below the slab
  const clusters = pm.clusterUnseen(ceiling);
  const exposure = pm.classifyExposure(clusters[0], world);
  assert.strictEqual(exposure, 'down');
  const vantage = pm.vantageFor(clusters[0], exposure, [2.5, 66, 2.5], [10, 64, 10]);
  assert.ok(vantage.pos[1] < 68, 'underside vantage stands beneath the ceiling');
  assert.ok(vantage.look[1] > vantage.pos[1], 'and looks UP at it');
  console.log('ok: ceiling underside -> beneath, looking up');
}

// --- two separated clusters -> the nearest is targeted, never their midpoint -----
{
  const near = [[0, 64, 0], [1, 64, 0], [0, 64, 1]];
  const far = [[60, 64, 60], [61, 64, 60], [60, 64, 61]];
  const clusters = pm.clusterUnseen([...near, ...far]);
  assert.strictEqual(clusters.length, 2, 'separated groups must stay separate clusters');
  const picked = pm.pickCluster(clusters, [2, 64, 2]);
  assert.ok(picked.centroid[0] < 10, 'the nearest cluster wins');
  console.log('ok: separated clusters -> nearest targeted, no phantom midpoint');
}

// --- accumulation: samples add up, absences age out -------------------------------
{
  const store = new Map();
  pm.accumulateCells(store, [[0, 64, 0], [1, 64, 0]], 4, 1000);
  pm.accumulateCells(store, [[2, 64, 0], [3, 64, 0]], 4, 2000);
  assert.strictEqual(store.size, 4, 'successive samples accumulate');
  // one cell keeps being re-sent; the rest go quiet past the expiry window
  pm.accumulateCells(store, [[0, 64, 0]], 4, 2000 + pm.ACCUMULATE_EXPIRE_MS + 1);
  assert.strictEqual(store.size, 1, 'unmentioned cells expire');
  // a big built_count drop clears the store (undo spree)
  pm.accumulateCells(store, [[9, 64, 9]], 0, 5000 + pm.ACCUMULATE_EXPIRE_MS);
  assert.ok(!pm.storeCells(store).some(([x]) => x === 0), 'a shrunken build resets the set');
  console.log('ok: accumulation adds up, expires, and resets on shrink');
}

// --- the sweep crosses the cluster's own extent, both axes ------------------------
{
  const roof = [];
  for (let x = 0; x < 9; x++) for (let z = 0; z < 9; z++) roof.push([x, 70, z]);
  const cluster = pm.clusterUnseen(roof)[0];
  const xs = new Set(), zs = new Set();
  for (let t = 0; t < 20000; t += 500) {
    const look = pm.sweepLook(cluster, t);
    xs.add(Math.round(look[0]));
    zs.add(Math.round(look[2]));
  }
  assert.ok(Math.max(...xs) - Math.min(...xs) > 4, 'sweep spans the cluster in x');
  assert.ok(Math.max(...zs) - Math.min(...zs) > 4, 'sweep spans the cluster in z');
  console.log('ok: sweep covers the cluster plane, not a fixed pendulum');
}

console.log('patrol_math: all checks passed');
