// Plain-node checks for placement_math.js (no test framework in this package):
//     node capture/mineflayer-bot/test_placement_math.js

'use strict';

const assert = require('assert');
const pm = require('./placement_math');

// A world window for support checks: everything is air except the given cells.
function solidOnly(cells) {
  const solid = new Set(cells.map(([x, y, z]) => `${x},${y},${z}`));
  return (x, y, z) => solid.has(`${x},${y},${z}`);
}

// --- ground placement clicks the top of the block below -------------------------
{
  const support = pm.supportFor([10, 65, 10], solidOnly([[10, 64, 10]]));
  assert.deepStrictEqual(support.ref, [10, 64, 10]);
  assert.deepStrictEqual(support.face, [0, 1, 0]);
  console.log('ok: ground placement stands on the block below');
}

// --- wall extension clicks the side of the existing wall block -------------------
{
  const support = pm.supportFor([11, 65, 10], solidOnly([[12, 65, 10]]));
  assert.deepStrictEqual(support.ref, [12, 65, 10]);
  assert.deepStrictEqual(support.face, [-1, 0, 0]);
  console.log('ok: side placement clicks the neighboring wall block');
}

// --- below-preference wins when several supports exist ---------------------------
{
  const support = pm.supportFor([5, 65, 5],
    solidOnly([[5, 64, 5], [6, 65, 5], [5, 66, 5]]));
  assert.deepStrictEqual(support.face, [0, 1, 0], 'prefers standing on the floor');
  console.log('ok: the floor face is preferred over sides and ceiling');
}

// --- a floating cell has no support and the game would refuse it ------------------
{
  assert.strictEqual(pm.supportFor([0, 80, 0], solidOnly([[0, 60, 0]])), null);
  console.log('ok: floating cell -> no support -> caller must skip');
}

// --- reach: near cell yes, far cell no -------------------------------------------
{
  assert.ok(pm.withinReach([10, 64, 10], [12, 64, 10]));
  assert.ok(!pm.withinReach([10, 64, 10], [20, 64, 10]));
  // the eye-height term matters: a cell 4 below the feet is farther than it looks
  assert.ok(!pm.withinReach([10, 64, 10], [10, 59, 10], 4.5));
  console.log('ok: reach measured from the eye to the cell center');
}

console.log('placement_math: all checks passed');
