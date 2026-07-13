// Placement — the pure math (no bot, no mineflayer, so it tests with plain
// node). agent.js owns the body; this module answers two questions about one
// directed placement: is the target close enough to act on from where the agent
// stands (withinReach), and which existing block does the new one attach to
// (supportFor — Minecraft never hangs a block in thin air; every placement is
// "click a face of a neighbor").
//
// The face vector convention matches mineflayer's placeBlock(reference, face):
// the vector points FROM the reference block TOWARD the cell being filled.

'use strict';

const REACH = 4.5;             // survival-player arm length, in blocks

// Neighbor order is a preference, not just an enumeration: placing on top of the
// block below looks the most natural in-game and never occludes the agent's own
// view, so "below the target" is tried first, then the four sides, then hanging
// from above as the last resort.
const NEIGHBOR_FACES = [
  { offset: [0, -1, 0], face: [0, 1, 0] },    // stand it on the block below
  { offset: [1, 0, 0], face: [-1, 0, 0] },
  { offset: [-1, 0, 0], face: [1, 0, 0] },
  { offset: [0, 0, 1], face: [0, 0, -1] },
  { offset: [0, 0, -1], face: [0, 0, 1] },
  { offset: [0, 1, 0], face: [0, -1, 0] },    // hang it from the block above
];

// Distance from the agent's eye line to the target cell's center. Reach in
// Minecraft is measured to where the crosshair ray hits, so cell center is the
// honest stand-in without simulating the whole ray.
function withinReach(botPos, cell, reach) {
  const r = reach == null ? REACH : reach;
  const dx = botPos[0] - (cell[0] + 0.5);
  const dy = botPos[1] + 1.62 - (cell[1] + 0.5);   // eye height, same as the scan
  const dz = botPos[2] - (cell[2] + 0.5);
  return Math.sqrt(dx * dx + dy * dy + dz * dz) <= r;
}

// The support face for placing at `cell`: the first neighbor (in preference
// order) that is solid ground to click on. `isSolidAt(x, y, z)` is the caller's
// live-world read. Returns { ref: [x,y,z], face: [fx,fy,fz] } or null when the
// cell floats with no solid neighbor — a placement the game itself would refuse.
function supportFor(cell, isSolidAt) {
  for (const { offset, face } of NEIGHBOR_FACES) {
    const ref = [cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2]];
    if (isSolidAt(ref[0], ref[1], ref[2])) return { ref, face };
  }
  return null;
}

module.exports = { REACH, withinReach, supportFor };
