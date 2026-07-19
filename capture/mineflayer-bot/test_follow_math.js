// Plain-node checks for follow_math.js (no test framework in this package):
//     node capture/mineflayer-bot/test_follow_math.js
// The far-follow cases are the 2026-07-18 failure: one goal set from spawn,
// path dead, no retry, whole session spent scanning the wrong neighborhood.

'use strict';

const assert = require('assert');
const fm = require('./follow_math');

// --- band: default, then closer after voicing, with the adoption hold ----------
{
  const s = fm.newState();
  assert.deepStrictEqual(fm.band(s, 1000), fm.BANDS.default);
  fm.noteEngaged(s, 2000);
  // wanting "engaged" starts the hold; adoption only after BAND_HOLD_MS
  assert.deepStrictEqual(fm.band(s, 2000), fm.BANDS.default);
  assert.deepStrictEqual(fm.band(s, 2000 + fm.BAND_HOLD_MS), fm.BANDS.engaged);
  // and it decays back to default once the engagement window passes
  const later = 2000 + fm.ENGAGED_HOLD_MS + 10;
  fm.band(s, later);                                   // candidate flips
  assert.deepStrictEqual(fm.band(s, later + fm.BAND_HOLD_MS), fm.BANDS.default);
}

// --- band: a fast-moving human earns room ---------------------------------------
{
  const s = fm.newState();
  fm.noteHumanPos(s, [0, 64, 0], 0);
  fm.noteHumanPos(s, [5, 64, 0], 1000);               // 5 blocks/s: sprinting
  assert.ok(s.humanSpeed > fm.BUSY_SPEED);
  fm.band(s, 1000);
  assert.deepStrictEqual(fm.band(s, 1000 + fm.BAND_HOLD_MS), fm.BANDS.busy);
}

// --- band: voicing outranks speed (engaged wins over busy) ----------------------
{
  const s = fm.newState();
  fm.noteHumanPos(s, [0, 64, 0], 0);
  fm.noteHumanPos(s, [6, 64, 0], 1000);
  fm.noteEngaged(s, 1000);
  fm.band(s, 1000);
  assert.deepStrictEqual(fm.band(s, 1000 + fm.BAND_HOLD_MS), fm.BANDS.engaged);
}

// --- far-follow: the first far tick issues a goal immediately -------------------
{
  const s = fm.newState();
  const step = fm.farStep(s, 30, 1000);
  assert.strictEqual(step.issue, true);
  assert.strictEqual(step.recovering, false);
  assert.strictEqual(step.announce, false);
}

// --- far-follow: progress is quiet; a stall re-issues the goal (the 07-18 bug) --
{
  const s = fm.newState();
  fm.farStep(s, 30, 0);                                // initial goal
  assert.strictEqual(fm.farStep(s, 25, 2000).issue, false);   // closing: leave it be
  // now the distance freezes (path dead) — after REFRESH_MS a fresh goal goes out
  let t = 2000;
  let reissued = false;
  for (let i = 0; i < 30 && !reissued; i++) {
    t += 1000;
    reissued = fm.farStep(s, 25, t).issue;
  }
  assert.ok(reissued, "a dead path must trigger a goal re-issue");
  assert.ok(fm.farStep(s, 25, t + 1000).recovering, "a stalled follow reads as recovering");
}

// --- far-follow: long enough + far enough -> say it out loud, once --------------
{
  const s = fm.newState();
  fm.farStep(s, 100, 0);
  const t = fm.ANNOUNCE_AFTER_MS + 1000;
  const step = fm.farStep(s, 100, t);
  assert.strictEqual(step.announce, true);
  assert.strictEqual(fm.farStep(s, 100, t + 1000).announce, false,  // gap guard
      "announcements must not repeat inside ANNOUNCE_GAP_MS");
}

// --- far-follow: never announce a short hop (close = quiet recovery) ------------
{
  const s = fm.newState();
  fm.farStep(s, 12, 0);
  assert.strictEqual(fm.farStep(s, 12, fm.ANNOUNCE_AFTER_MS + 1000).announce, false,
      "12 blocks away is a recovery, not a speech");
}

// --- far-follow: reset clears the episode ---------------------------------------
{
  const s = fm.newState();
  fm.farStep(s, 30, 0);
  fm.farReset(s);
  assert.strictEqual(s.farSinceMs, null);
  assert.strictEqual(fm.farStep(s, 30, 50000).issue, true);   // a fresh episode
}

console.log("follow_math: all checks passed");
