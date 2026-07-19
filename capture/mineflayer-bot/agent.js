'use strict';
// MICA_AI — the agent's embodiment. It joins the human's world as a REAL PLAYER
// named MICA_AI, which is the whole trick of the actor contract (contracts/b0.py):
// the capture attributes block events by username, so everything this player will
// ever place is tagged as the agent's with no special writer path. Replicas join
// as MICA_AI_1, MICA_AI_2, ... (set MICA_AGENT_NAME).
//
// HOW TO CONNECT (the rig is singleplayer): the human plays their normal modded
// singleplayer world and presses Open to LAN (cheats ON if the A7 test will run).
// The mod's place/break hooks live in the client's INTEGRATED server, so once
// MICA_AI is in the world, its placements are recorded with actor "MICA_AI"
// by the existing capture — mixed-actor sessions are proof-grade as-is. This
// script discovers the LAN world automatically (Minecraft announces LAN games on
// multicast 224.0.2.60:4445); set MICA_PORT to skip discovery.
//
// AUTHORITY (D5, presence v1.1 "active shadowing", pinned 2026-07-04; §9 amendment
// 2026-07-12): the agent TRACES the player. It follows at a 4-8 block band (paths
// closer beyond 8, backs away inside 4), stays out of the human's workspace (backs
// off the pipeline's focus block), looks where the pipeline says the human's
// attention is, and may say ONE throttled advisory line about what the belief
// tracker currently believes. The body changes the world ONLY on the mind's gate
// blocks: a 'gather' state runs one whitelisted fetch errand, a 'place_low_risk'
// state with a directive places exactly ONE mind-authorized reversible block —
// both exist only when run_live was started with their flags (--gather / --place),
// and chat "stop" halts either instantly. The A7 test flag remains the smoke-only
// exception that places without the gate.
//
// THE AGENT'S MIND lives in run_live.py (D1 behavior + D2 structure + belief, and
// with --pixels the VPT/MineCLIP head). This script is only the BODY: it reads the
// mind's ~1 Hz snapshot from live_status.json (written next to the session; found
// via the MICA_RAW_DIRS roots) — it never touches the single-consumer live socket.
//
// On spawn it opens the MICA FlowViz dashboard in the browser so the live data
// flow is visible the moment the AI is in the world (an already-running instance
// is reused, never duplicated).
//
// Env: MICA_HOST (127.0.0.1), MICA_PORT (default: LAN auto-discovery, else 25565),
//      MICA_VERSION (1.16.5), MICA_AGENT_NAME (MICA_AI),
//      MICA_RAW_DIRS (';'-joined capture roots; default = the repo's capture/raw),
//      MICA_FLOWVIZ_DIR (D:\2026projects\mica-flowviz), MICA_FLOWVIZ_PORT (8321),
//      MICA_NO_FLOWVIZ=1 (skip the dashboard), MICA_QUIET=1 (no advisory chat),
//      MICA_A7_TEST=1 (place three test blocks after 10 s — smoke sessions only).

const dgram = require('dgram');
const fs = require('fs');
const { exec, spawn } = require('child_process');
const http = require('http');
const net = require('net');
const path = require('path');
const mineflayer = require('mineflayer');
const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
const patrolMath = require('./patrol_math');   // coverage patrol v2: the pure math
const placeMath = require('./placement_math'); // gate placement: the pure math
const Vec3 = require('vec3');

const HOST = process.env.MICA_HOST || '127.0.0.1';
const VERSION = process.env.MICA_VERSION || '1.16.5';
const NAME = process.env.MICA_AGENT_NAME || 'MICA_AI';
// A7 depends on the body's own blocks being attributable to the AGENT (MICA_AI or
// MICA_AI_<n>), so the evidence filters exclude them. A name that does not match
// that shape would make every block the body places read as HUMAN evidence — the
// belief confirming itself through the agent's own hand (review 2026-07-17 F7).
// Refuse to start rather than silently poison a whole session's evidence.
if (!/^MICA_AI(_.*)?$/.test(NAME)) {   // exactly is_agent_actor()'s rule (contracts/b0.py)
  console.error(`[agent] MICA_AGENT_NAME='${NAME}' is not MICA_AI or MICA_AI_<suffix>; `
    + 'the A7 evidence filter would tag this body as human. Refusing to join.');
  process.exit(1);
}
const VIEWER_PORT = parseInt(process.env.MICA_VIEWER_PORT || '3007', 10);
const FLOWVIZ_DIR = process.env.MICA_FLOWVIZ_DIR || 'D:\\2026projects\\mica-flowviz';
const FLOWVIZ_PORT = parseInt(process.env.MICA_FLOWVIZ_PORT || '8321', 10);
const RAW_DIR = path.join(__dirname, '..', 'raw');
const RAW_ROOTS = (process.env.MICA_RAW_DIRS || RAW_DIR).split(';').filter(Boolean);

// The agent-eye scan (h3d_scan sensor, design note "Agent-Scan 3D Channel"):
// a LiDAR-style ray grid over the first-person frustum, cast from the live eye
// pose against the world data this client has received. Occlusion is real —
// a wall hides what is behind it — so covering a building requires moving
// around it. The scan only RECORDS what was seen; deciding which cells belong
// to the build happens mind-side, where the session's base snapshot is known.
const SCAN_COLS = 32;              // rays across ±45° horizontal
const SCAN_ROWS = 18;              // rays across ±35° vertical (70° FOV)
const SCAN_H_HALF = Math.PI / 4;   // 90° horizontal spread (16:9 at 70° vertical)
const SCAN_V_HALF = (35 * Math.PI) / 180;
const SCAN_RANGE_MIN = 32;         // blocks; the original fixed range (small builds)
const SCAN_RANGE_MAX = 96;         // rays stay cheap; beyond this the grid is too sparse
const SCAN_EVERY_MS = 250;         // 4 Hz, alongside the follow/gaze loops
const EYE_HEIGHT = 1.62;

// The shadowing band (D5 presence v1.1) and the advisory throttles.
const FOLLOW_FAR = 8;          // beyond this, path toward the human
const FOLLOW_NEAR = 4;         // inside this, back away (proximity, applied to self)
const WORKSPACE_R = 3;         // never stand this close to the human's focus block
const STATUS_STALE_S = 5;      // a live_status.json older than this means no pipeline
const ADVISORY_P = 0.5;        // belief confidence needed before saying anything
const ADVISORY_STREAK = 5;     // consecutive 1 Hz reads the top goal must hold
const ADVISORY_GAP_MS = 30000; // at most one advisory line per 30 s

// Find the Open-to-LAN world's port: the game broadcasts "[MOTD]...[/MOTD][AD]port[/AD]"
// on multicast every ~1.5 s. Falls back to 25565 (a dedicated server) after the timeout.
function discoverLanPort(timeoutMs, callback) {
  const sock = dgram.createSocket({ type: 'udp4', reuseAddr: true });
  const timer = setTimeout(() => { try { sock.close(); } catch (e) {} callback(null); }, timeoutMs);
  sock.on('error', () => { clearTimeout(timer); try { sock.close(); } catch (e) {} callback(null); });
  sock.on('message', (msg) => {
    const match = /\[AD\](\d+)\[\/AD\]/.exec(msg.toString());
    if (match) {
      clearTimeout(timer);
      try { sock.close(); } catch (e) {}
      callback(parseInt(match[1], 10));
    }
  });
  sock.bind(4445, () => {
    try { sock.addMembership('224.0.2.60'); } catch (e) { /* multicast may be off; timeout handles it */ }
  });
}

function isAgentName(name) {
  // mirrors contracts/b0.py is_agent_actor: MICA_AI exactly, or MICA_AI_<suffix>
  return name === 'MICA_AI' || name.startsWith('MICA_AI_');
}

// --- the mind's snapshot: run_live.py rewrites live_status.json ~1 Hz -----------
function readLiveStatus() {
  let newest = null;
  for (const root of RAW_ROOTS) {
    try {
      const status = JSON.parse(fs.readFileSync(path.join(root.trim(), 'live_status.json'), 'utf8'));
      if (!newest || (status.ts || 0) > (newest.ts || 0)) newest = status;
    } catch (e) { /* no pipeline writing in this root */ }
  }
  if (!newest || Date.now() / 1000 - newest.ts > STATUS_STALE_S) return null;
  return newest;
}

// --- FlowViz: reuse a running instance, else start one, then open the browser ---
function flowvizUp(callback) {
  const req = http.get({ host: '127.0.0.1', port: FLOWVIZ_PORT, path: '/', timeout: 1500 },
    (res) => { res.resume(); callback(true); });
  req.on('error', () => callback(false));
  req.on('timeout', () => { req.destroy(); callback(false); });
}

function openBrowser() {
  exec(`start "" http://localhost:${FLOWVIZ_PORT}`, { shell: 'cmd.exe', windowsHide: true });
  console.log(`[${NAME}] live data flow: http://localhost:${FLOWVIZ_PORT}`);
}

function ensureFlowViz() {
  if (process.env.MICA_NO_FLOWVIZ === '1') return;
  flowvizUp((up) => {
    if (up) return openBrowser();
    console.log(`[${NAME}] starting FlowViz from ${FLOWVIZ_DIR}`);
    // The capture location follows the launcher profile (mica.captureDir), so
    // pass every known root; MICA_RAW_DIRS (';'-joined) overrides the default.
    const rawDirs = process.env.MICA_RAW_DIRS || RAW_DIR;
    const child = spawn('python', ['flowviz.py', '--raw-dir', rawDirs],
      { cwd: FLOWVIZ_DIR, detached: true, stdio: 'ignore', windowsHide: true });
    child.on('error', (err) =>
      console.error(`[${NAME}] could not start FlowViz (${err.message}) — start it by hand`));
    child.unref();
    setTimeout(openBrowser, 2000);   // give the tailer a moment to bind its port
  });
}

// --- the player ---
function start(port) {
  const bot = mineflayer.createBot({
    host: HOST, port, username: NAME, version: VERSION, auth: 'offline',
  });
  bot.loadPlugin(pathfinder);

  function nearestHuman() {
    // bot.players is undefined until login completes; the status writer runs
    // from tick one, so this must tolerate the pre-login state. With more than
    // one human in the world, follow/gaze must bind to the NEAREST, not an
    // arbitrary map-order player (review 2026-07-17 F26).
    const players = bot.players || {};
    const me = bot.entity && bot.entity.position;
    let best = null;
    let bestDist = Infinity;
    for (const name of Object.keys(players)) {
      const player = players[name];
      if (isAgentName(name) || !player || !player.entity) continue;
      const d = me ? me.distanceTo(player.entity.position) : 0;
      if (d < bestDist) { bestDist = d; best = player; }
    }
    return best;
  }

  function nearestDroppedItem(me) {
    // The closest dropped-item entity within toss range. 1.16 mineflayer names
    // them 'item' (older builds say objectType 'Item'); WHAT the item is stays
    // version-fragile metadata, so while a shortage request is outstanding,
    // anything tossed nearby is worth walking to — the receipt check in liveTick
    // decides whether it was the right block.
    let best = null;
    let bestDistance = 16;
    for (const id of Object.keys(bot.entities || {})) {
      const entity = bot.entities[id];
      if (!entity || !entity.position) continue;
      if (entity.name !== 'item' && entity.objectType !== 'Item') continue;
      const distance = me.distanceTo(entity.position);
      if (distance < bestDistance) {
        best = entity;
        bestDistance = distance;
      }
    }
    return best;
  }

  // --- agent status file: the AGENT-side perspective for FlowViz -------------
  // One flushed JSON line per second into the capture folder; FlowViz tails it
  // next to the session logs and renders the agent panel from it. Truncated on
  // every launch (the tailer treats the shrink as a rewrite and starts over).
  const statusPath = path.join(RAW_DIR, `agent-${NAME}.status.jsonl`);
  // The scan file: one line per 4 Hz sweep that saw something new. A previous
  // launch's scan is rotated aside — never deleted — so a finished session's scan
  // survives for the post-session report. The rotated name carries the SESSION ID
  // when the sweeps are tagged with one (they are, once the mind's live_status
  // names the session), so the file says which session it belongs to instead of
  // leaving the report to guess by clock. Untagged files keep the mtime name.
  const scanPath = path.join(RAW_DIR, `agent-${NAME}.scan.jsonl`);
  function lastTaggedSession(file) {
    // The newest sweep line that names its session. Scan files stay small (a few
    // hundred KB), so one whole read at launch costs nothing.
    try {
      const lines = fs.readFileSync(file, 'utf8').split('\n');
      for (let i = lines.length - 1; i >= 0; i--) {
        if (!lines[i]) continue;
        try {
          const row = JSON.parse(lines[i]);
          if (row.session) return row.session;
        } catch (err) { /* torn line: keep looking */ }
      }
    } catch (err) { /* unreadable: fall back to the mtime name */ }
    return null;
  }
  try {
    const prev = fs.statSync(scanPath);
    if (prev.size > 0) {
      const prevSession = lastTaggedSession(scanPath);
      let rotated = scanPath.replace(/\.jsonl$/, `.${prevSession || Math.floor(prev.mtimeMs)}.jsonl`);
      if (fs.existsSync(rotated)) {
        // Same session, second agent launch: keep both files apart.
        rotated = scanPath.replace(/\.jsonl$/, `.${prevSession}.${Math.floor(prev.mtimeMs)}.jsonl`);
      }
      fs.renameSync(scanPath, rotated);
    }
  } catch (err) { /* no previous scan */ }
  const scanSeen = new Set();     // "x,y,z" of every cell recorded this session
  let scanSessionId = null;       // the capture session the mind says is open —
                                  // sticky: kept after the pipeline goes quiet, so
                                  // late sweeps still say which session they saw
  let agentState = 'connecting';
  let lastAction = null;
  let followMode = 'holding';
  let liveStatus = null;          // the mind's newest fresh snapshot (or null)
  let viewerPort = null;          // the port the agent camera ACTUALLY bound (null = down)
  let viewerError = null;         // why the camera is down, readable from the dashboard
  // --- the materials constraint (D5 §4, 2026-07-06): what the agent HOLDS is what
  // it may place. The inventory and any chat-granted substitutions travel to the
  // mind in the status line below; the gate caps its committed prefix on them.
  const materialGrants = {};           // block -> substitute the human said "yes" to
  const deniedSubstitutes = new Set(); // "block:substitute" refused — never re-asked
  let pendingAsk = null;               // {block, substitute, atMs} awaiting yes/no
  let lastMaterialAskMs = 0;
  // The shortage request ("agent asks, you provide", 2026-07-13): what the agent
  // has asked the human for, and the last voiced shortage so it never repeats
  // itself for the same missing set.
  const neededBlocks = new Set();
  let lastNeedSig = null;
  let lastNeedMs = 0;
  // --- gate placement (D5 §9 amendment 2026-07-12): the body executes ONE mind-
  // authorized block at a time, and only while the gate keeps saying so.
  let placing = null;                  // { id, cell, block } while an errand runs
  let placeStopped = false;            // the human said stop — stands until "go on"
  let lastPlaceResult = null;          // { id, cell, block, ok, note } of the last errand
  const placedDirectives = new Set();  // directive ids already acted on (never repeat)
  let lastPlaceChatMs = 0;
  function inventorySnapshot() {
    const counts = {};
    try {
      for (const item of bot.inventory.items()) {
        counts[item.name] = (counts[item.name] || 0) + item.count;
      }
    } catch (err) { /* inventory not readable yet (pre-spawn) */ }
    return counts;
  }
  function writeStatus() {
    const e = bot.entity;
    const human = nearestHuman();
    const line = JSON.stringify({
      ts: Date.now(),
      name: NAME,
      state: agentState,
      port,
      pos: e && e.position ? [e.position.x, e.position.y, e.position.z] : null,
      yaw: e ? e.yaw : null,
      pitch: e ? e.pitch : null,
      watching: human ? (human.username || human.name || null) : null,
      follow: followMode,
      believes: liveStatus ? { top_goal: liveStatus.top_goal, p: liveStatus.p_top_goal } : null,
      viewer_port: viewerPort,
      viewer_error: viewerError,
      scan_cells: scanSeen.size,
      last_action: lastAction,
      inventory: inventorySnapshot(),
      material_grants: materialGrants,
      // The placement story, mind-readable: the errand in flight (if any) and the
      // last completed directive's outcome. The B0 capture stays the authoritative
      // record of what actually landed (actor MICA_AI); this is the live readout.
      placing: placing ? { id: placing.id, cell: placing.cell, block: placing.block } : null,
      last_place: lastPlaceResult,
    }) + '\n';
    try { fs.appendFileSync(statusPath, line); } catch (err) { /* disk hiccup: skip a beat */ }
  }
  try { fs.writeFileSync(statusPath, ''); } catch (err) { /* created on first append */ }
  const statusTimer = setInterval(writeStatus, 250);   // 4 Hz: the scan view renders from this pose
  const timers = [statusTimer];
  writeStatus();

  // --- active shadowing (D5 presence v1.1): follow, hold, back off -----------
  function focusVec() {
    const fb = liveStatus && liveStatus.focus_block;
    return fb ? new Vec3(fb[0] + 0.5, fb[1] + 0.5, fb[2] + 0.5) : null;
  }

  function setGoal(goal, dynamic) {
    try { bot.pathfinder.setGoal(goal, dynamic || false); } catch (e) { /* mid-respawn */ }
  }

  // --- coverage patrol (the scan channel's legs): when the human is idle or far,
  // walk to a vantage on the build's UNSEEN side and sweep the gaze across it.
  // The scan sensor itself is untouched — it records whatever the eyes pass over;
  // patrol only decides where the body stands. Safety order is unchanged and
  // above patrol: gate YIELD, the workspace rule, and the personal band all win,
  // and the moment the human acts nearby the normal follow behavior returns.
  const PATROL_MIN_UNSEEN = 0.10;    // engage only while >10% of the build is unseen
  const PATROL_REPOSITION_MS = 8000; // one vantage move at most every 8 s
  const AT_REST_MS = 60000;          // build at rest = human idle + no new blocks this long
  let lastPatrolMoveMs = 0;
  let patrolCluster = null;          // the unseen cluster the eyes sweep while patrolling
  // Patrol v2 (2026-07-11, patrol_math.js): the body accumulates the mind's
  // per-write built-cell SAMPLES into a working set (large builds no longer fit
  // one status line), clusters the unseen cells instead of averaging them into a
  // phantom mid-structure target, and picks exposure-aware vantages — height for
  // roofs, beneath-looking-up for ceiling undersides. Standing ON the human's
  // build is allowed only while the build is at rest (the user's climb rule).
  const builtStore = new Map();
  let lastBuiltCount = 0;
  let lastBuiltChangeMs = 0;

  function buildAtRest(now) {
    const idle = liveStatus && liveStatus.current_behavior === 'idle';
    return idle && now - lastBuiltChangeMs >= AT_REST_MS;
  }

  function isAirAt(x, y, z) {
    const block = bot.blockAt(new Vec3(x, y, z));
    return !block || block.boundingBox === 'empty';
  }

  function standingOnBuild() {
    const me = bot.entity && bot.entity.position;
    if (!me) return false;
    const below = `${Math.floor(me.x)},${Math.floor(me.y) - 1},${Math.floor(me.z)}`;
    return builtStore.has(below);
  }

  function patrolTick(me, them, focus, now) {
    const known = patrolMath.storeCells(builtStore);
    if (!known.length) return false;
    const unseen = known.filter(c => !scanSeen.has(`${c[0]},${c[1]},${c[2]}`));
    if (unseen.length / known.length < PATROL_MIN_UNSEEN) {
      if (followMode === 'patrolling') { setGoal(null); followMode = 'holding'; patrolCluster = null; }
      return false;                   // coverage is good: nothing to patrol for
    }
    const clusters = patrolMath.clusterUnseen(unseen);
    const cluster = patrolMath.pickCluster(clusters, [me.x, me.y, me.z]);
    if (!cluster) return false;
    const buildC = patrolMath.clusterUnseen(known)[0]
      ? patrolMath.clusterUnseen(known).reduce((a, b) => (a.cells.length >= b.cells.length ? a : b)).centroid
      : cluster.centroid;
    const exposure = patrolMath.classifyExposure(cluster, isAirAt);
    let vantage = patrolMath.vantageFor(cluster, exposure, buildC, [me.x, me.y, me.z]);
    if (vantage.onBuildLikely && !buildAtRest(now)) {
      // The climb rule: high vantages can mean standing on the human's build, so
      // they wait for the at-rest window; until then take the level view of the
      // same cluster (partial coverage now beats trespassing).
      vantage = patrolMath.vantageFor(cluster, 'side', buildC, [me.x, me.y, me.z]);
    }
    const spot = new Vec3(vantage.pos[0], vantage.pos[1], vantage.pos[2]);
    // The same vetoes as everything else: never into the workspace, never into
    // the human's personal space — if the vantage would violate them, no patrol.
    if (focus && spot.distanceTo(focus) < WORKSPACE_R + 1) return false;
    if (them && spot.distanceTo(them) < FOLLOW_NEAR + 1) return false;
    if (now - lastPatrolMoveMs >= PATROL_REPOSITION_MS) {
      setGoal(new goals.GoalNear(spot.x, spot.y, spot.z, 2));
      lastPatrolMoveMs = now;
      lastAction = `patrol: ${unseen.length} unseen, ${exposure} cluster of ${cluster.cells.length}`;
    }
    patrolCluster = cluster;          // lookTick sweeps this cluster's real extent
    followMode = 'patrolling';
    return true;
  }

  let lastBackoffMs = 0;
  function followTick() {
    if (agentState !== 'present') return;
    if (gathering) return;   // a gather errand owns the pathfinder until it ends
    if (placing) return;     // so does a placement errand
    const human = nearestHuman();
    if (!human || !human.entity) {
      if (followMode !== 'searching') { setGoal(null); followMode = 'searching'; }
      return;
    }
    const me = bot.entity.position;
    const them = human.entity.position;
    const d = me.distanceTo(them);
    const focus = focusVec();
    const now = Date.now();

    // The workspace rule: never stand where the human is working.
    if (focus && me.distanceTo(focus) < WORKSPACE_R) {
      if (now - lastBackoffMs > 750) {          // pathfinder churn guard
        const away = me.minus(focus).normalize().scaled(WORKSPACE_R + 1);
        const spot = me.plus(away);
        setGoal(new goals.GoalNear(spot.x, spot.y, spot.z, 1));
        lastBackoffMs = now;
      }
      followMode = 'yielding workspace';
      return;
    }
    // Receiving posture (user decision 2026-07-13): while a shortage request is
    // outstanding, a human coming close is most likely BRINGING materials — the
    // band retreat (and the yield-widened band) fought every hand-over, with the
    // agent literally fleeing the delivery. So instead: walk to nearby dropped
    // items (pickup is automatic on contact) and stand still for an approaching
    // human. The workspace rule above still wins, and the posture ends the
    // moment the shortage clears (the thank-you in liveTick clears it).
    if (neededBlocks.size) {
      const drop = nearestDroppedItem(me);
      if (drop) {
        if (now - lastBackoffMs > 750) {
          setGoal(new goals.GoalNear(drop.position.x, drop.position.y,
                                     drop.position.z, 1));
          lastBackoffMs = now;
        }
        followMode = 'collecting';
        return;
      }
      if (d < FOLLOW_NEAR + 4) {
        setGoal(null);
        followMode = 'receiving';
        return;
      }
    }
    // The D5 gate's YIELD widens the personal band for a few seconds: the agent
    // steps further out the moment the gate says the human is too close to its
    // would-be target. Same retreat leg, bigger radius, nothing new to test.
    const nearBand = Date.now() < yieldUntilMs ? FOLLOW_NEAR + 4 : FOLLOW_NEAR;
    // Coverage patrol slots in BELOW the vetoes above and ABOVE plain following:
    // it may claim the tick only while the human is idle or far, and never during
    // a gate YIELD. The instant the human acts nearby, the branches below resume.
    const humanIdle = liveStatus && liveStatus.current_behavior === 'idle';
    if ((humanIdle || d > FOLLOW_FAR) && now >= yieldUntilMs
        && d >= nearBand && patrolTick(me, them, focus, now)) {
      return;
    }
    if (followMode === 'patrolling') { patrolCluster = null; followMode = 'holding'; }
    // The climb rule's other half: the moment the build stops being at rest, an
    // agent standing ON it gets off — retreat toward the human's band, which is
    // always ground the human can stand on too.
    if (standingOnBuild() && !buildAtRest(now)) {
      if (now - lastBackoffMs > 750) {
        setGoal(new goals.GoalFollow(human.entity, FOLLOW_NEAR + 1), true);
        lastBackoffMs = now;
      }
      followMode = 'dismounting';
      return;
    }
    if (d > FOLLOW_FAR && Date.now() >= yieldUntilMs) {
      if (followMode !== 'following') {
        setGoal(new goals.GoalFollow(human.entity, FOLLOW_NEAR + 1), true);
        followMode = 'following';
      }
    } else if (d < nearBand) {
      if (now - lastBackoffMs > 750) {
        const away = me.minus(them).normalize().scaled(nearBand + 1 - d);
        const spot = me.plus(away);
        setGoal(new goals.GoalNear(spot.x, spot.y, spot.z, 1));
        lastBackoffMs = now;
      }
      followMode = 'backing off';
    } else if (followMode !== 'holding') {
      setGoal(null);
      followMode = 'holding';
    }
  }

  // Gaze, real-time: mirror where the HUMAN is looking, from their entity's live
  // yaw/pitch (mineflayer updates these ~per tick — no 1 Hz status-file latency).
  // Project a point a few blocks along their view direction: that is where their
  // crosshair rests, which is what "watching what the player does" means.
  function lookTick() {
    if (agentState !== 'present') return;
    if (placing) return;     // placeBlock manages its own aim; don't yank the view
    if (followMode === 'patrolling' && patrolCluster) {
      // On patrol the eyes belong to the scan: a two-axis sweep across the
      // cluster's own extent (patrol v2 — the old fixed pendulum never crossed
      // a roof plane), instead of mirroring the human (idle or far — that's
      // why patrol engaged).
      const look = patrolMath.sweepLook(patrolCluster, Date.now());
      bot.lookAt(new Vec3(look[0], look[1], look[2])).catch(() => {});
      return;
    }
    const human = nearestHuman();
    if (!human || !human.entity) return;
    const e = human.entity;
    const head = e.position.offset(0, 1.6, 0);
    let target = head;
    if (e.yaw != null && e.pitch != null) {
      const dir = new Vec3(-Math.sin(e.yaw) * Math.cos(e.pitch),
                           Math.sin(e.pitch) * -1,
                           -Math.cos(e.yaw) * Math.cos(e.pitch));
      target = head.plus(dir.scaled(4));
    }
    bot.lookAt(target).catch(() => {});
  }

  // --- the advisory line (embodied pre-B6 SUGGEST, D5 pin): one throttled ----
  // chat message when the belief has held a confident top goal for a while.
  let streakGoal = null;
  let streak = 0;
  let lastAnnounced = null;
  let lastAdvisoryMs = 0;
  let lastEscapeWarnMs = 0;
  // --- D5 gate rendering: when the pipeline publishes a `gate` block, the REAL ---
  // gate drives what the agent says and does; the pre-B6 advisory line below stays
  // only as the fallback for gate-less runs (exactly the handover the vault pinned).
  let lastGateChatMs = 0;
  let lastGateState = null;
  // R-6: summaries are no longer null in suggest state, so "re-voice when a
  // summary exists" would repeat the identical offer every 30 s. Re-voice only
  // when the offer itself changed (new block/cell), or on a state change.
  let lastVoicedSummary = null;
  let yieldUntilMs = 0;
  function renderGate(gate) {
    if (!gate || !gate.state) return false;
    const now = Date.now();
    if (gate.state === 'yield') {
      // step back: reuse the proximity band's retreat leg by biasing the follow
      // distance out; the band logic in presenceTick does the walking.
      yieldUntilMs = now + 4000;
      lastAction = 'gate: yield (backing off)';
    } else if ((gate.state === 'suggest' || gate.state === 'preview')
               && process.env.MICA_QUIET !== '1' && agentState === 'present'
               && now - lastGateChatMs >= ADVISORY_GAP_MS
               && (gate.state !== lastGateState
                   || (gate.proposal_summary
                       && gate.proposal_summary !== lastVoicedSummary))) {
      lastGateChatMs = now;
      lastVoicedSummary = gate.proposal_summary || null;
      if (gate.state === 'suggest') {
        bot.chat(`Shall I help? I could add ${gate.proposal_summary || 'the next piece'}`
          + ` (conf ${Math.round((gate.conf || 0) * 100)}%)`);
      } else {
        bot.chat(`Previewing: ${gate.proposal_summary || 'a placement'} — say no to wave me off`);
        if (gate.target_cell) {
          const [gx, gy, gz] = gate.target_cell;
          bot.lookAt(new Vec3(gx + 0.5, gy + 0.5, gz + 0.5)).catch(() => {});
        }
      }
      // The acceptance signal's VOICING record (D9 §3): a suggestion only counts
      // as suggested if this throttled line actually reached the human. The mind
      // cannot know that (the 30 s throttle lives here), so the body logs each
      // voicing; the offline join (scripts/suggestion_acceptance.py) matches these
      // lines to trace rows and to the human's next placements.
      fs.appendFile(path.join(RAW_DIR, `agent-${NAME}.voiced.jsonl`),
        JSON.stringify({ ts: now, state: gate.state,
                         summary: gate.proposal_summary || null,
                         conf: gate.conf != null ? gate.conf : null,
                         cell: gate.target_cell || null }) + '\n', () => {});
      lastAction = `gate: ${gate.state}`;
    } else if (gate.state === 'gather' && gate.gather && !gathering && !gatherStopped
               && agentState === 'present') {
      // The mind cleared the D5 lattice (declared target or theta_place, safe
      // window, hysteresis); the body announces and runs ONE errand at a time.
      runGatherErrand(liveStatus, gate.gather);
      lastAction = 'gate: gather';
    } else if (gate.state === 'place_low_risk' && gate.place && !placing && !gathering
               && !placeStopped && agentState === 'present'
               && !placedDirectives.has(gate.place.id)) {
      // The mind authorized exactly ONE reversible block (§9 amendment: declared
      // target or theta_place, staircase, materials, safe window, hysteresis all
      // already cleared). The body walks, re-checks the live world, places once.
      runPlaceErrand(gate.place);
      lastAction = 'gate: place';
    }
    // The materials ask (D5 §4): substitution is never silent — when the gate says
    // it is short a block and sees a same-family stand-in in the inventory, ask
    // ONCE and wait for a chat yes/no. A refusal is remembered; silence times out.
    if (pendingAsk && now - pendingAsk.atMs > 60000) pendingAsk = null;
    const ask = gate.materials && gate.materials.ask;
    if (ask && agentState === 'present' && process.env.MICA_QUIET !== '1'
        && !pendingAsk && !materialGrants[ask.block]
        && !deniedSubstitutes.has(`${ask.block}:${ask.substitute}`)
        && now - lastMaterialAskMs >= ADVISORY_GAP_MS) {
      lastMaterialAskMs = now;
      pendingAsk = { block: ask.block, substitute: ask.substitute, atMs: now };
      bot.chat(`I'm short ${ask.short} ${ask.block} for what I'd add — okay to use `
        + `${ask.substitute} instead? say yes or no`);
      lastAction = `materials ask: ${ask.block} -> ${ask.substitute}`;
    }
    // The shortage request (user decision 2026-07-13: "agent asks, you provide").
    // When placement is armed and the gate is short a block with NO substitute to
    // offer (ask above is null), say plainly what is needed so the human can toss
    // it over. Once per shortage — a new request only when the missing set changes
    // — and the thank-you in liveTick confirms receipt.
    const missing = (gate.materials && gate.materials.missing) || {};
    const needSig = Object.keys(missing).sort()
      .map((block) => `${block}:${missing[block]}`).join(',');
    if (gate.authority === 'place' && needSig && !ask && !placeStopped
        && agentState === 'present' && process.env.MICA_QUIET !== '1'
        && needSig !== lastNeedSig && now - lastNeedMs >= ADVISORY_GAP_MS) {
      lastNeedMs = now;
      lastNeedSig = needSig;
      const wants = Object.keys(missing).sort()
        .map((block) => `${missing[block]} ${block}`).join(', ');
      bot.chat(`I need ${wants} to help build — toss them to me and I'll place `
        + `when it's safe.`);
      for (const block of Object.keys(missing)) neededBlocks.add(block);
      lastAction = `materials request: ${wants}`;
    }
    lastGateState = gate.state;
    return true;
  }

  // --- autonomous gathering (D5 §4 amendment, 2026-07-10) ---------------------
  // Runs ONLY while the REAL gate publishes state === 'gather' (the mind already
  // checked declaration/threshold + safe window + hysteresis). The body re-checks
  // its own guardrails anyway: whitelisted source only, never inside the capture
  // region plus a 16-block buffer, never a mind-known built cell, one errand at a
  // time, one stack cap, announced first, chat "stop" aborts instantly.
  const GATHER_BUFFER = 16;
  const GATHER_WHITELIST = new Set(['oak_log', 'spruce_log', 'birch_log', 'stone',
    'cobblestone', 'dirt', 'sand', 'gravel', 'poppy', 'dandelion']);
  let gathering = null;          // { source, count, mined, forBlock } while an errand runs
  let gatherStopped = false;     // the human said stop — stands until they say "go on"

  function insideProtected(pos, status) {
    const region = status && status.region;
    if (!region) return true;    // unknown region: protect everything (fail safe)
    return pos.x >= region[0] - GATHER_BUFFER && pos.x <= region[3] + GATHER_BUFFER
        && pos.z >= region[2] - GATHER_BUFFER && pos.z <= region[5] + GATHER_BUFFER;
  }

  async function runGatherErrand(status, plan) {
    const errand = ((plan && plan.plan) || []).find((entry) => entry.gather);
    if (!errand || !GATHER_WHITELIST.has(errand.gather)) return;
    gathering = { source: errand.gather, count: Math.min(errand.count || 1, 64),
                  mined: 0, forBlock: errand.for };
    bot.chat(`Gathering ${gathering.count} ${gathering.source} for the ${plan.target}`
      + ' — say stop to cancel.');
    lastAction = `gather: ${gathering.source}`;
    try {
      const type = bot.registry && bot.registry.blocksByName[gathering.source];
      if (!type) return;
      const built = new Set(((status && status.built_cells) || [])
        .map(([x, y, z]) => `${x},${y},${z}`));
      const unreachable = new Set();   // walk-timeout cells: never retried this errand
      while (gathering && !gatherStopped && gathering.mined < gathering.count) {
        const found = bot.findBlocks({ matching: type.id, maxDistance: 48, count: 24 })
          .filter((pos) => !insideProtected(pos, status)
            && !built.has(`${pos.x},${pos.y},${pos.z}`)
            && !unreachable.has(`${pos.x},${pos.y},${pos.z}`));
        if (!found.length) {
          bot.chat(`No ${gathering.source} in reach outside your build area — `
            + 'leaving that to you.');
          break;
        }
        // P-3: the same 20 s walk race the placement errand runs — an unreachable
        // ore/log skips to the next candidate, never hangs the errand.
        try {
          await gotoWithTimeout(new goals.GoalGetToBlock(
            found[0].x, found[0].y, found[0].z), 20000);
        } catch (err) {
          unreachable.add(`${found[0].x},${found[0].y},${found[0].z}`);
          continue;
        }
        if (!gathering || gatherStopped) break;
        const block = bot.blockAt(found[0]);
        if (!block || block.name !== gathering.source) continue;   // world moved on
        await bot.dig(block);
        gathering.mined += 1;
      }
      if (gathering && gathering.mined >= gathering.count) {
        bot.chat(`Done — ${gathering.mined} ${gathering.source} gathered.`);
      }
    } catch (err) {
      lastAction = `gather aborted: ${err.message}`;
    } finally {
      gathering = null;
      try { bot.pathfinder.setGoal(null); } catch (e) { /* mid-respawn */ }
    }
  }

  // An unreachable goal must never freeze the body (review 2026-07-13 F3): every
  // errand walk races a hard timeout. The caller's finally clears the goal; the
  // timer is cleared so a finished walk leaves nothing ticking (P-3).
  function gotoWithTimeout(goal, ms) {
    let timer = null;
    return Promise.race([
      bot.pathfinder.goto(goal),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('walk timed out')), ms);
      }),
    ]).finally(() => clearTimeout(timer));
  }

  // --- gate placement errand (D5 §9 amendment, 2026-07-12) --------------------
  // Runs ONLY while the REAL gate publishes state === 'place_low_risk' with a
  // directive the body has not seen before. The mind already authorized it; the
  // body still re-checks the live world (the cell must be empty, a support face
  // must exist) because the mind's world knowledge is one second old, and it
  // re-reads the gate after walking because authority can drop mid-approach.
  function isSolidAt(x, y, z) {
    // prismarine-block's boundingBox is 'block' or 'empty' — 'block' is the
    // solid case (isAirAt above tests the other side of the same coin).
    const block = bot.blockAt(new Vec3(x, y, z));
    return !!block && block.boundingBox === 'block';
  }

  function finishPlace(directive, ok, note) {
    lastPlaceResult = { id: directive.id, cell: directive.cell,
                        block: directive.block, ok, note };
    lastAction = ok ? `placed ${directive.block} @ (${directive.cell})`
                    : `place skipped: ${note}`;
    if (!ok) console.log(`[${NAME}] directive ${directive.id} not placed: ${note}`);
  }

  async function runPlaceErrand(directive) {
    placedDirectives.add(directive.id);   // one attempt per directive, ever
    placing = { id: directive.id, cell: directive.cell, block: directive.block };
    const [cx, cy, cz] = directive.cell;
    const now = Date.now();
    if (now - lastPlaceChatMs > ADVISORY_GAP_MS && process.env.MICA_QUIET !== '1') {
      lastPlaceChatMs = now;   // one line per burst, not one per block
      bot.chat(`Adding ${directive.block} to the build — say stop to cancel.`);
    }
    try {
      const item = bot.inventory.items().find((i) => i.name === directive.block);
      if (!item) return finishPlace(directive, false, 'block not in inventory');
      const me = bot.entity.position;
      if (!placeMath.withinReach([me.x, me.y, me.z], directive.cell)) {
        // An unreachable target must not freeze the body (review 2026-07-13 F3).
        await gotoWithTimeout(new goals.GoalNear(cx, cy, cz, 3), 20000);
      }
      if (placeStopped || !placing) return finishPlace(directive, false, 'stopped');
      // Authority can drop while walking (the human came back): obey the freshest
      // gate state, not the one that started the errand.
      const fresh = readLiveStatus();
      if (!fresh || !fresh.gate || fresh.gate.state !== 'place_low_risk') {
        return finishPlace(directive, false, 'gate withdrew during approach');
      }
      const target = bot.blockAt(new Vec3(cx, cy, cz));
      if (target && target.boundingBox !== 'empty') {
        return finishPlace(directive, false, 'cell no longer empty');
      }
      const support = placeMath.supportFor(directive.cell, isSolidAt);
      if (!support) return finishPlace(directive, false, 'no support face to click');
      await bot.equip(item, 'hand');
      const ref = bot.blockAt(new Vec3(support.ref[0], support.ref[1], support.ref[2]));
      if (!ref) return finishPlace(directive, false, 'support block unloaded');
      await bot.placeBlock(ref, new Vec3(support.face[0], support.face[1], support.face[2]));
      const landed = bot.blockAt(new Vec3(cx, cy, cz));
      const ok = !!landed && landed.name === directive.block;
      finishPlace(directive, ok, ok ? null : `landed ${landed ? landed.name : 'nothing'}`);
      if (ok) console.log(`[${NAME}] placed ${directive.block} @ (${cx},${cy},${cz})`
        + ` [directive ${directive.id}, route ${directive.route}]`);
    } catch (err) {
      finishPlace(directive, false, err.message);
    } finally {
      placing = null;
      try { bot.pathfinder.setGoal(null); } catch (e) { /* mid-respawn */ }
      writeStatus();   // publish the outcome now, not up to 250 ms later
    }
  }

  // The human's answer to a pending materials ask, and the gather kill-switch.
  // Only an explicit "yes" grants the one block->substitute equivalence
  // (session-scoped); "no" is remembered so the same question is never asked
  // twice. The grant reaches the mind through the status file's material_grants.
  bot.on('chat', (username, message) => {
    if (username === NAME) return;
    const said = String(message).trim().toLowerCase();
    // Match "stop" ANYWHERE ("please stop", "mica stop", "STOP IT"), but not a
    // negation ("don't stop") — human panic phrasing, not just the announced word
    // (review 2026-07-17 F24). The announcements still say "say stop to cancel".
    if (/\bstop\b/.test(said) && !/\b(don'?t|do not|never|no need to)\s+stop\b/.test(said)) {
      if (gathering || placing) bot.chat('Stopping — errand dropped.');
      gathering = null;
      placing = null;
      gatherStopped = true;
      placeStopped = true;   // one word halts every autonomous world change
      // Abort an in-flight dig too: the errand pointer is cleared above, but a
      // bot.dig() already awaiting completion keeps mining for up to seconds
      // otherwise — the "halts instantly" contract must cover the swing already
      // underway (review 2026-07-17 F5).
      try { bot.stopDigging(); } catch (e) { /* not digging */ }
      try { bot.pathfinder.setGoal(null); } catch (e) { /* mid-respawn */ }
      return;
    }
    if (/^(go on|resume)\b/.test(said) && (gatherStopped || placeStopped)) {
      gatherStopped = false;
      placeStopped = false;
      bot.chat('Okay — I may gather or place again when the gate allows it.');
      return;
    }
    if (!pendingAsk) return;
    const text = said;
    if (/^(yes|yeah|ok|okay|sure)\b/.test(text)) {
      materialGrants[pendingAsk.block] = pendingAsk.substitute;
      bot.chat(`Noted — ${pendingAsk.substitute} stands in for ${pendingAsk.block} this session.`);
      lastAction = `materials grant: ${pendingAsk.block} -> ${pendingAsk.substitute}`;
      pendingAsk = null;
      writeStatus();
    } else if (/^(no|nope|nah|don't|dont)\b/.test(text)) {
      deniedSubstitutes.add(`${pendingAsk.block}:${pendingAsk.substitute}`);
      bot.chat('Understood — I\'ll plan without it.');
      lastAction = `materials denied: ${pendingAsk.block}`;
      pendingAsk = null;
      writeStatus();
    }
  });

  function liveTick() {
    liveStatus = readLiveStatus();
    if (liveStatus && liveStatus.session_id) scanSessionId = liveStatus.session_id;
    // Receipt check for the shortage request: the moment a block the agent asked
    // for shows up in its inventory, say so — the human should never have to
    // wonder whether the hand-over worked. Clearing the signature lets the next
    // (different) shortage be voiced without waiting out the throttle.
    if (neededBlocks.size && agentState === 'present') {
      const counts = inventorySnapshot();
      for (const block of Array.from(neededBlocks)) {
        if ((counts[block] || 0) > 0) {
          neededBlocks.delete(block);
          lastNeedSig = null;
          if (process.env.MICA_QUIET !== '1') {
            bot.chat(`Got the ${block} — thanks. I'll place when it's safe.`);
          }
          lastAction = `materials received: ${block}`;
          writeStatus();
        }
      }
    }
    if (liveStatus && liveStatus.built_cells) {
      // Patrol v2: each status write carries a fresh SAMPLE of the built set —
      // accumulate them so the patrol's target set covers the whole build.
      patrolMath.accumulateCells(builtStore, liveStatus.built_cells,
                                 liveStatus.built_count, Date.now());
      if ((liveStatus.built_count || 0) !== lastBuiltCount) {
        lastBuiltCount = liveStatus.built_count || 0;
        lastBuiltChangeMs = Date.now();   // the at-rest clock for the climb rule
      }
    }
    if (liveStatus && renderGate(liveStatus.gate)) {
      return;   // the gate spoke (or chose silence); the fallback advisory stays quiet
    }
    if (!liveStatus || !liveStatus.top_goal || (liveStatus.p_top_goal || 0) < ADVISORY_P) {
      streak = 0;
      streakGoal = null;
      return;
    }
    // Capture-health alarms belong IN THE GAME: the builder is fullscreen and sees
    // neither the terminal nor the dashboard. A real session lost its structure
    // evidence to region escapes while both were screaming off-screen. Escapes are
    // unrecoverable for the session, so say so once a minute until they stop.
    const escapes = liveStatus.crop_escapes || 0;
    if (escapes > 0 && Date.now() - lastEscapeWarnMs > 60000 && agentState === 'present') {
      lastEscapeWarnMs = Date.now();
      bot.chat(`!! MICA: you are building OUTSIDE the capture region (${escapes} escaped) — `
        + `structure evidence is VOID for this session. Quit the GAME fully, relaunch, `
        + `then place your first block at the build site.`);
      lastAction = `escape warning (${escapes})`;
    }
    if (liveStatus.top_goal === streakGoal) streak += 1;
    else { streakGoal = liveStatus.top_goal; streak = 1; }
    const now = Date.now();
    if (streak >= ADVISORY_STREAK && streakGoal !== lastAnnounced
        && now - lastAdvisoryMs >= ADVISORY_GAP_MS && process.env.MICA_QUIET !== '1'
        && agentState === 'present') {
      const pct = Math.round(liveStatus.p_top_goal * 100);
      const article = /^[aeiou]/i.test(streakGoal) ? 'an' : 'a';
      bot.chat(`I think you're building ${article} ${streakGoal} build (${pct}%)`);
      lastAction = `advisory: ${streakGoal} ${pct}%`;
      lastAnnounced = streakGoal;
      lastAdvisoryMs = now;
    }
  }

  // --- the agent-eye scan (h3d_scan sensor) ----------------------------------
  // Sweep a fixed ray grid over the first-person frustum from the live eye pose.
  // Every ray stops at the first block it hits (real occlusion); cells never
  // seen before are appended to the scan file with the pose that saw them.
  // Range is ADAPTIVE (user decision 2026-07-13): the old fixed 32 blocks went
  // blind past the far edge of a grown capture region — the range now follows
  // the region's own diagonal (the mind publishes it in live_status), clamped so
  // rays stay cheap. The ray BUDGET stays fixed: at long range the grid is
  // sparser per block, and the coverage patrol closes that by walking nearer —
  // coverage converges by movement, not by ray spam.
  function scanRange() {
    const region = liveStatus && liveStatus.region;
    if (!region) return SCAN_RANGE_MIN;
    const dx = region[3] - region[0];
    const dy = region[4] - region[1];
    const dz = region[5] - region[2];
    const diagonal = Math.sqrt(dx * dx + dy * dy + dz * dz);
    return Math.max(SCAN_RANGE_MIN, Math.min(SCAN_RANGE_MAX, Math.ceil(diagonal) + 8));
  }

  function scanTick() {
    if (agentState !== 'present' || !bot.entity || !bot.entity.position) return;
    const e = bot.entity;
    const eye = e.position.offset(0, EYE_HEIGHT, 0);
    const range = scanRange();
    const found = [];
    for (let col = 0; col < SCAN_COLS; col++) {
      const yaw = e.yaw + (col / (SCAN_COLS - 1) - 0.5) * 2 * SCAN_H_HALF;
      for (let row = 0; row < SCAN_ROWS; row++) {
        const pitch = e.pitch + (row / (SCAN_ROWS - 1) - 0.5) * 2 * SCAN_V_HALF;
        // same direction convention the gaze mirror uses (verified in-game)
        const dir = new Vec3(-Math.sin(yaw) * Math.cos(pitch), -Math.sin(pitch),
                             -Math.cos(yaw) * Math.cos(pitch));
        let hit = null;
        try {
          hit = bot.world.raycast(eye, dir, range);
        } catch (err) { /* a chunk mid-load: this ray just misses */ }
        if (!hit || !hit.position || hit.name === 'air') continue;
        const p = hit.position;
        const key = `${p.x},${p.y},${p.z}`;
        if (scanSeen.has(key)) continue;
        scanSeen.add(key);
        found.push([p.x, p.y, p.z, hit.name]);
      }
    }
    if (!found.length) return;
    const line = JSON.stringify({
      ts: Date.now(),
      tick: bot.time ? bot.time.age : null,
      session: scanSessionId,    // which capture session these cells were seen in
      pose: [e.position.x, e.position.y, e.position.z, e.yaw, e.pitch],
      cells: found,
    }) + '\n';
    try { fs.appendFileSync(scanPath, line); } catch (err) { /* disk hiccup: skip a sweep */ }
  }

  // The A7 smoke check: three stone blocks next to the agent, spaced 2 s apart.
  // Each one must be recorded with actor MICA_AI (the integrated server's hooks),
  // excluded from the evidence (run_d1 lists them under "agent"; D2's built-set
  // ignores them), and present in the replay check's world — session proof-grade.
  // Needs cheats enabled in the Open-to-LAN dialog for /give.
  async function a7TestPlacement() {
    try {
      agentState = 'a7_test';
      setGoal(null);
      bot.chat(`/give ${NAME} minecraft:stone 8`);
      await new Promise((r) => setTimeout(r, 1000));
      const stone = bot.inventory.items().find((item) => item.name === 'stone');
      if (!stone) { console.log(`[${NAME}] A7 test: no stone (cheats on in the LAN dialog?)`); agentState = 'present'; return; }
      await bot.equip(stone, 'hand');
      for (let i = 0; i < 3; i++) {
        const spot = bot.entity.position.floored().offset(2 + i, -1, 2);
        const below = bot.blockAt(spot);
        if (!below || below.name === 'air') continue;
        try {
          await bot.placeBlock(below, new Vec3(0, 1, 0));
          lastAction = `a7 place stone @ (${spot.x},${spot.y + 1},${spot.z})`;
          console.log(`[${NAME}] A7 test: placed stone @ (${spot.x},${spot.y + 1},${spot.z})`);
        } catch (err) {
          console.log(`[${NAME}] A7 test: placement ${i} failed (${err.message})`);
        }
        await new Promise((r) => setTimeout(r, 2000));
      }
      bot.chat(`${NAME}: A7 test done — my blocks are tagged ${NAME} and stay out of the evidence.`);
      agentState = 'present';
    } catch (err) {
      console.error(`[${NAME}] A7 test error: ${err.message}`);
      agentState = 'present';
    }
  }

  bot.once('spawn', () => {
    agentState = 'present';
    console.log(`[${NAME}] joined ${HOST}:${port} (mc ${VERSION}) — shadowing; world`
      + ' changes only when the gate authorizes them');
    bot.chat(`${NAME} online — I'll follow and watch; I only build or gather when `
      + `the gate says so, and "stop" always halts me.`);
    ensureFlowViz();
    writeStatus();

    // The body may move (pathfinder) but must never change the world: no digging,
    // no scaffold placement, no tower-building to reach a path.
    const movements = new Movements(bot);
    movements.canDig = false;
    movements.allow1by1towers = false;
    movements.scafoldingBlocks = [];
    movements.placeCost = 1000000;      // belt and braces: placing is never worth it
    // The pathfinder cannot dig or place, but vanilla physics let a jumping/falling
    // player TRAMPLE farmland to dirt (popping the crop) — a world change the "never
    // change the world" config otherwise misses, and possibly an unlogged D2
    // divergence (review 2026-07-17 F15). Steer paths off farmland and crops.
    try {
      const avoid = ['farmland', 'wheat', 'carrots', 'potatoes', 'beetroots',
        'melon_stem', 'pumpkin_stem', 'sweet_berry_bush'];
      for (const name of avoid) {
        const block = bot.registry && bot.registry.blocksByName[name];
        if (block) movements.blocksToAvoid.add(block.id);
      }
    } catch (e) { /* registry shape varies by mineflayer version; best-effort */ }
    bot.pathfinder.setMovements(movements);

    // Tracking loops at 4 Hz (was 1/0.67 Hz — visibly laggy): both read live entity
    // data, so the only remaining latency is the interval itself plus path physics.
    timers.push(setInterval(followTick, 250));
    timers.push(setInterval(lookTick, 250));
    timers.push(setInterval(liveTick, 1000));   // belief/advisory stays 1 Hz by nature
    if (process.env.MICA_NO_SCAN !== '1') {
      timers.push(setInterval(scanTick, SCAN_EVERY_MS));   // the agent-eye scan (h3d_scan)
    }

    // The agent's own camera: a live first-person render of what MICA_AI sees,
    // served in the browser (prismarine-viewer). FlowViz embeds it as the
    // "agent camera" view; MICA_NO_VIEWER=1 skips it.
    //
    // Two hard lessons are baked in here. First: when this camera fails to start,
    // the reason must land in the status file (viewer_error) — a whole session
    // once ran with the camera dead and the only trace was one line in the rig
    // terminal. Second: prismarine-viewer's web server has no error handler, so
    // handing it a port that is already taken would crash this entire process
    // (a zombie agent once sat on the default port). So each candidate port is
    // probed with a throwaway server first, and whatever port really binds is
    // what the status file publishes.
    function probePort(port, cb) {
      const probe = net.createServer();
      probe.once('error', () => cb(false));
      probe.once('listening', () => probe.close(() => cb(true)));
      probe.listen(port, '127.0.0.1');
    }
    function startViewer(candidates) {
      if (!candidates.length) {
        viewerError = `all camera ports in use (${VIEWER_PORT}-${VIEWER_PORT + 2})`;
        console.error(`[${NAME}] ${viewerError}`);
        writeStatus();
        return;
      }
      const port = candidates[0];
      probePort(port, (free) => {
        if (!free) return startViewer(candidates.slice(1));
        try {
          require('prismarine-viewer').mineflayer(bot, {
            port, firstPerson: true, viewDistance: 4,
          });
          viewerPort = port;      // the port that really bound, not the wished-for one
          viewerError = null;
          console.log(`[${NAME}] agent camera: http://localhost:${port}`);
        } catch (err) {
          viewerError = err.message;   // e.g. "Cannot find module 'canvas'"
          console.error(`[${NAME}] viewer failed to start (${err.message}) — continuing without`);
        }
        writeStatus();          // publish the outcome now, not a second later
      });
    }
    if (process.env.MICA_NO_VIEWER !== '1') {
      startViewer([VIEWER_PORT, VIEWER_PORT + 1, VIEWER_PORT + 2]);
    }

    // The A7 smoke test places three ungated stone blocks 10 s after spawn. It is
    // a manual-rig check ONLY — an env var left set from a past smoke run must not
    // fire ungated placements into a real gated capture (review 2026-07-17 F25).
    // Guard: skip (loudly) if a live gate is already driving this session.
    if (process.env.MICA_A7_TEST === '1') {
      setTimeout(() => {
        const live = readLiveStatus();
        if (live && live.gate) {
          console.log(`[${NAME}] A7 smoke test SKIPPED — a live gated session is `
            + 'running (unset MICA_A7_TEST for real captures).');
          return;
        }
        a7TestPlacement();
      }, 10000);
    }
  });

  bot.on('error', (err) => console.error(`[${NAME}] error:`, err.message));
  bot.on('kicked', (reason) => console.error(`[${NAME}] kicked:`, reason));
  // Zombie watchdog (a hung agent once blocked the watcher's re-arm and silently
  // killed the next session's launch): the server sends a time update every second
  // while we are connected — half a minute of silence in the 'present' state means
  // the connection is gone even if 'end' never fired. Exit hard; the watcher needs
  // both children DEAD to re-arm.
  let lastServerTimeMs = Date.now();
  bot.on('time', () => { lastServerTimeMs = Date.now(); });
  timers.push(setInterval(() => {
    if (agentState === 'present' && Date.now() - lastServerTimeMs > 30000) {
      console.error(`[${NAME}] no server heartbeat for 30s — exiting (watchdog)`);
      agentState = 'disconnected';
      writeStatus();
      process.exit(0);
    }
  }, 5000));
  bot.on('end', () => {
    agentState = 'disconnected';
    writeStatus();
    for (const timer of timers) clearInterval(timer);
    // The materials one-liner (chat is gone once disconnected, so it goes to the
    // rig console); the full ledger is the mind's <session>.materials_report.json.
    const grants = Object.keys(materialGrants).length
      ? JSON.stringify(materialGrants) : 'none';
    console.log(`[${NAME}] materials: substitution grants ${grants} — `
      + 'full report written next to the session logs');
    console.log(`[${NAME}] disconnected`);
    // Exit outright: a leftover timer would keep this process alive, and the
    // lan_autostart watcher only re-arms once BOTH rig halves have exited.
    process.exit(0);
  });
}

if (process.env.MICA_PORT) {
  start(parseInt(process.env.MICA_PORT, 10));
} else {
  console.log(`[${NAME}] looking for an Open-to-LAN world (8 s) ...`);
  discoverLanPort(8000, (port) => {
    if (port) console.log(`[${NAME}] found LAN world on port ${port}`);
    else console.log(`[${NAME}] no LAN announcement heard — trying 25565 (set MICA_PORT to override)`);
    start(port || 25565);
  });
}
