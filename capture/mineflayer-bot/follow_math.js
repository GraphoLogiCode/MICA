// The follow policy's pure half (no bot, no mineflayer, tests with plain node).
// agent.js owns the body; this module answers two questions it kept getting
// wrong:
//
// 1. HOW CLOSE should the agent stand right now? The old band was a fixed
//    4-8 blocks whatever was happening. Now it depends on the moment: step IN
//    after voicing a suggestion (talking to someone from 8 blocks away is not
//    social), give ROOM while the human is moving hard, hold the default band
//    otherwise. A candidate band must persist briefly before it is adopted, so
//    the agent does not dance between bands at a boundary.
//
// 2. IS the far-follow actually working? On 2026-07-18 the agent set one
//    pathfinder goal from spawn, the path failed somewhere over ~100 blocks of
//    terrain, and nothing ever retried — it spent the whole session scanning
//    the wrong neighborhood. The progress tracker notices "far and not getting
//    closer", asks for a fresh goal every few seconds, and after long enough
//    tells the caller to say something out loud so the human knows.

'use strict';

const BANDS = {
  default: { near: 4, far: 8 },
  engaged: { near: 3, far: 6 },    // it just spoke: come make eye contact
  busy: { near: 6, far: 10 },      // the human is moving hard: give room
};
const ENGAGED_HOLD_MS = 12000;     // a voiced suggestion keeps the close band this long
const BUSY_SPEED = 4.0;            // blocks/s (walk ~4.3, sprint ~5.6)
const BAND_HOLD_MS = 2000;         // a new band must persist this long before adoption
const REFRESH_MS = 5000;           // no progress for this long -> re-issue the goal
const ANNOUNCE_AFTER_MS = 30000;   // stuck this long -> worth saying out loud
const ANNOUNCE_GAP_MS = 60000;     // but never nag more often than this
const ANNOUNCE_MIN_D = 20;         // and only when genuinely far
const PATROL_MAX_D = 20;           // beyond this, reaching the human outranks patrol

function newState() {
  return {
    // band selection ("never engaged" must not read as "engaged at time 0")
    lastEngagedMs: -Infinity,
    humanPos: null, humanPosMs: 0, humanSpeed: 0,
    band: 'default', candidate: 'default', candidateSinceMs: 0,
    // far-follow progress (null = no far episode in flight; a timestamp of 0
    // must stay a valid timestamp)
    farSinceMs: null, bestD: Infinity, lastProgressMs: 0,
    lastGoalMs: -Infinity, lastAnnounceMs: -Infinity,
  };
}

function noteEngaged(state, nowMs) {
  state.lastEngagedMs = nowMs;
}

function noteHumanPos(state, pos, nowMs) {
  // Speed from successive samples, only over a window long enough to mean
  // something (the follow loop runs at 4 Hz; a single frame is noise).
  if (state.humanPos && nowMs - state.humanPosMs >= 250) {
    const dx = pos[0] - state.humanPos[0], dy = pos[1] - state.humanPos[1],
          dz = pos[2] - state.humanPos[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    state.humanSpeed = dist / ((nowMs - state.humanPosMs) / 1000);
    state.humanPos = pos;
    state.humanPosMs = nowMs;
  } else if (!state.humanPos) {
    state.humanPos = pos;
    state.humanPosMs = nowMs;
  }
}

function band(state, nowMs) {
  // What the moment calls for...
  let want = 'default';
  if (nowMs - state.lastEngagedMs < ENGAGED_HOLD_MS) want = 'engaged';
  else if (state.humanSpeed >= BUSY_SPEED) want = 'busy';
  // ...adopted only after it holds for a moment (boundary churn guard).
  if (want !== state.candidate) {
    state.candidate = want;
    state.candidateSinceMs = nowMs;
  } else if (want !== state.band && nowMs - state.candidateSinceMs >= BAND_HOLD_MS) {
    state.band = want;
  }
  return BANDS[state.band];
}

function farStep(state, d, nowMs) {
  // One call per follow tick while the human is beyond the band. Returns what
  // the body should do: issue (set a fresh pathfinder goal), recovering (the
  // last goal made no progress — label the mode so the status file shows it),
  // announce (say out loud that the path is failing).
  const first = state.farSinceMs === null;
  if (first) {
    state.farSinceMs = nowMs;
    state.bestD = d;
    state.lastProgressMs = nowMs;
  } else if (d < state.bestD - 1) {          // a full block closer counts as progress
    state.bestD = d;
    state.lastProgressMs = nowMs;
  }
  const stalled = nowMs - state.lastProgressMs >= REFRESH_MS;
  const issue = first || (stalled && nowMs - state.lastGoalMs >= REFRESH_MS);
  if (issue) state.lastGoalMs = nowMs;
  const announce = stalled && d >= ANNOUNCE_MIN_D
      && nowMs - state.farSinceMs >= ANNOUNCE_AFTER_MS
      && nowMs - state.lastAnnounceMs >= ANNOUNCE_GAP_MS;
  if (announce) state.lastAnnounceMs = nowMs;
  return { issue, recovering: stalled, announce };
}

function farReset(state) {
  state.farSinceMs = null;
  state.bestD = Infinity;
  state.lastProgressMs = 0;
  state.lastGoalMs = -Infinity;
}

module.exports = {
  BANDS, ENGAGED_HOLD_MS, BUSY_SPEED, BAND_HOLD_MS, REFRESH_MS,
  ANNOUNCE_AFTER_MS, ANNOUNCE_GAP_MS, ANNOUNCE_MIN_D, PATROL_MAX_D,
  newState, noteEngaged, noteHumanPos, band, farStep, farReset,
};
