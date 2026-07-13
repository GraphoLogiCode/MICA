// Coverage patrol v2 — the pure math (no bot, no mineflayer, so it tests with
// plain node). agent.js owns the body; this module answers three questions:
// which unseen cells travel together (clusterUnseen), which way does a cluster
// face (classifyExposure), and where should the agent stand and look to see it
// (vantageFor / sweepLook).
//
// Why this exists (2026-07-11): the old patrol aimed at the global centroid of
// EVERY unseen cell — gaps on the roof, the far wall, and the interior averaged
// to a phantom target in the middle of the structure — and then flattened the
// approach direction to horizontal, deleting exactly the component a ceiling
// needs (up). Clusters fix the phantom; exposure-aware vantages fix the roof.

'use strict';

const CLUSTER_GRID = 8;        // cells bucket on an 8-cube grid; adjacent buckets merge
const STANDOFF = 7;            // blocks back from the cluster face (unchanged from v1)

// --- accumulation: the body's working set of built cells ------------------------
// The mind sends a fresh random SAMPLE of built cells each ~1 Hz write (large
// builds do not fit one status line). The body accumulates samples into a
// timestamped map so its target set converges on the whole build; entries
// refresh whenever re-sent and expire when the mind stops mentioning them for a
// while (a broken block stops being sent, so it ages out).
const ACCUMULATE_EXPIRE_MS = 90000;

function accumulateCells(store, sampledCells, builtCount, nowMs) {
  if (!sampledCells) return store;
  if (builtCount != null && builtCount < store.size * 0.5) {
    store.clear();               // the build shrank a lot (undo spree): start over
  }
  for (const cell of sampledCells) {
    store.set(`${cell[0]},${cell[1]},${cell[2]}`, { cell, seenMs: nowMs });
  }
  for (const [key, entry] of store) {
    if (nowMs - entry.seenMs > ACCUMULATE_EXPIRE_MS) store.delete(key);
  }
  return store;
}

function storeCells(store) {
  return Array.from(store.values(), (entry) => entry.cell);
}

// --- clustering: which unseen cells belong together ------------------------------
// Bucket on a coarse grid, then flood-merge buckets that touch (26-adjacency in
// bucket space). Returns clusters sorted by size, each { cells, centroid, bbox }.
function clusterUnseen(cells) {
  const buckets = new Map();
  for (const cell of cells) {
    const key = `${Math.floor(cell[0] / CLUSTER_GRID)},${Math.floor(cell[1] / CLUSTER_GRID)},`
      + `${Math.floor(cell[2] / CLUSTER_GRID)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(cell);
  }
  const visited = new Set();
  const clusters = [];
  for (const start of buckets.keys()) {
    if (visited.has(start)) continue;
    const members = [];
    const queue = [start];
    visited.add(start);
    while (queue.length) {
      const key = queue.pop();
      members.push(...buckets.get(key));
      const [bx, by, bz] = key.split(',').map(Number);
      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const neighbor = `${bx + dx},${by + dy},${bz + dz}`;
            if (!visited.has(neighbor) && buckets.has(neighbor)) {
              visited.add(neighbor);
              queue.push(neighbor);
            }
          }
        }
      }
    }
    clusters.push(describeCluster(members));
  }
  clusters.sort((a, b) => b.cells.length - a.cells.length);
  return clusters;
}

function describeCluster(cells) {
  let sx = 0, sy = 0, sz = 0;
  const bbox = { minX: Infinity, minY: Infinity, minZ: Infinity,
                 maxX: -Infinity, maxY: -Infinity, maxZ: -Infinity };
  for (const [x, y, z] of cells) {
    sx += x; sy += y; sz += z;
    bbox.minX = Math.min(bbox.minX, x); bbox.maxX = Math.max(bbox.maxX, x);
    bbox.minY = Math.min(bbox.minY, y); bbox.maxY = Math.max(bbox.maxY, y);
    bbox.minZ = Math.min(bbox.minZ, z); bbox.maxZ = Math.max(bbox.maxZ, z);
  }
  const n = cells.length || 1;
  return { cells, centroid: [sx / n + 0.5, sy / n + 0.5, sz / n + 0.5], bbox };
}

// The cluster the agent should work on: the nearest of the biggest — big gaps
// first, ties (within 2x) broken by walking distance. Never a midpoint between
// clusters, which is what the old global centroid effectively produced.
function pickCluster(clusters, agentPos) {
  if (!clusters.length) return null;
  const biggest = clusters[0].cells.length;
  const candidates = clusters.filter((c) => c.cells.length * 2 >= biggest);
  candidates.sort((a, b) => distance(agentPos, a.centroid) - distance(agentPos, b.centroid));
  return candidates[0];
}

function distance(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2);
}

// --- exposure: which way does the cluster face -----------------------------------
// isAir(x, y, z) is the caller's window into the world (bot.blockAt in agent.js,
// a plain set lookup in tests). Sampled over up to 32 cells of the cluster.
function classifyExposure(cluster, isAir) {
  const step = Math.max(1, Math.floor(cluster.cells.length / 32));
  let up = 0, down = 0, side = 0;
  for (let index = 0; index < cluster.cells.length; index += step) {
    const [x, y, z] = cluster.cells[index];
    if (isAir(x, y + 1, z)) up++;
    if (isAir(x, y - 1, z)) down++;
    if (isAir(x + 1, y, z) || isAir(x - 1, y, z) || isAir(x, y, z + 1) || isAir(x, y, z - 1)) side++;
  }
  // Dominance order matters: a roof slab is up-exposed AND side-exposed at its
  // rim — "up" wins because the top faces are the ones a ground vantage misses.
  if (up >= side && up >= down && up > 0) return 'up';
  if (side > 0) return 'side';
  if (down > 0) return 'down';
  return 'enclosed';
}

// --- vantage: where to stand ------------------------------------------------------
// Returns { pos: [x, y, z], look: [x, y, z], onBuildLikely } — the goal agent.js
// hands the pathfinder, and whether reaching it probably means standing on the
// build (the at-rest climb rule gates that case).
function vantageFor(cluster, exposure, buildCentroid, agentPos) {
  const c = cluster.centroid;
  let out = [c[0] - buildCentroid[0], 0, c[2] - buildCentroid[2]];
  let norm = Math.hypot(out[0], out[2]);
  if (norm < 0.5) {                       // cluster sits over the build's middle
    out = [agentPos[0] - c[0], 0, agentPos[2] - c[2]];
    norm = Math.hypot(out[0], out[2]);
  }
  if (norm < 0.5) { out = [1, 0, 0]; norm = 1; }
  out = [out[0] / norm, 0, out[2] / norm];

  if (exposure === 'up') {
    // Roof/ceiling tops: height is the missing ingredient the old patrol deleted.
    // Stand off the rim AND above the plane, looking down across it.
    return {
      pos: [c[0] + out[0] * STANDOFF * 0.7,
            cluster.bbox.maxY + 1 + STANDOFF * 0.7,
            c[2] + out[2] * STANDOFF * 0.7],
      look: c,
      onBuildLikely: true,               // high ground is often the build itself
    };
  }
  if (exposure === 'down' || exposure === 'enclosed') {
    // Ceiling undersides / interior pockets: a voxel counts as scanned from any
    // face, so beneath-and-inside is legitimate coverage — stand under, look up.
    return {
      pos: [c[0], Math.max(cluster.bbox.minY - 2, 0), c[2]],
      look: c,
      onBuildLikely: false,
    };
  }
  // Side faces: the v1 geometry, kept — but only reached for genuinely
  // side-facing clusters now, so the y-flatten degeneracy is gone.
  return {
    pos: [c[0] + out[0] * STANDOFF, c[1], c[2] + out[2] * STANDOFF],
    look: c,
    onBuildLikely: false,
  };
}

// --- gaze: how to sweep the cluster ----------------------------------------------
// A two-axis Lissajous across the cluster's own extent (the old sweep was a
// fixed +-4 horizontal pendulum — a roof plane never got crossed edge to edge).
function sweepLook(cluster, nowMs) {
  const c = cluster.centroid;
  const halfX = Math.max((cluster.bbox.maxX - cluster.bbox.minX) / 2, 1.5);
  const halfY = Math.max((cluster.bbox.maxY - cluster.bbox.minY) / 2, 1.5);
  const halfZ = Math.max((cluster.bbox.maxZ - cluster.bbox.minZ) / 2, 1.5);
  return [
    c[0] + Math.sin(nowMs / 1500) * halfX,
    c[1] + Math.sin(nowMs / 2300) * halfY,   // different period: the path precesses
    c[2] + Math.cos(nowMs / 1500) * halfZ,
  ];
}

module.exports = {
  ACCUMULATE_EXPIRE_MS, CLUSTER_GRID, STANDOFF,
  accumulateCells, storeCells, clusterUnseen, pickCluster,
  classifyExposure, vantageFor, sweepLook, distance,
};
