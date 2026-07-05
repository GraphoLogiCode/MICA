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
// AUTHORITY (D5, presence v1.1 "active shadowing", pinned 2026-07-04): the agent
// TRACES the player but never builds. It follows at a 4-8 block band (paths closer
// beyond 8, backs away inside 4), stays out of the human's workspace (backs off
// the pipeline's focus block), looks where the pipeline says the human's attention
// is, and may say ONE throttled advisory line about what the belief tracker
// currently believes. Placement stays disabled; the one exception is the explicit
// A7 test flag, used during a smoke session to verify the actor filter in-game.
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
const Vec3 = require('vec3');

const HOST = process.env.MICA_HOST || '127.0.0.1';
const VERSION = process.env.MICA_VERSION || '1.16.5';
const NAME = process.env.MICA_AGENT_NAME || 'MICA_AI';
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
const SCAN_RANGE = 32;             // blocks; matches the far edge of a build region
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
    // from tick one, so this must tolerate the pre-login state.
    const players = bot.players || {};
    for (const name of Object.keys(players)) {
      const player = players[name];
      if (!isAgentName(name) && player && player.entity) return player;
    }
    return null;
  }

  // --- agent status file: the AGENT-side perspective for FlowViz -------------
  // One flushed JSON line per second into the capture folder; FlowViz tails it
  // next to the session logs and renders the agent panel from it. Truncated on
  // every launch (the tailer treats the shrink as a rewrite and starts over).
  const statusPath = path.join(RAW_DIR, `agent-${NAME}.status.jsonl`);
  // The scan file: one line per 4 Hz sweep that saw something new. A previous
  // launch's scan is rotated aside with a timestamp — never deleted — so a
  // finished session's scan survives for the post-session report.
  const scanPath = path.join(RAW_DIR, `agent-${NAME}.scan.jsonl`);
  try {
    const prev = fs.statSync(scanPath);
    if (prev.size > 0) {
      fs.renameSync(scanPath, scanPath.replace(/\.jsonl$/, `.${Math.floor(prev.mtimeMs)}.jsonl`));
    }
  } catch (err) { /* no previous scan */ }
  const scanSeen = new Set();     // "x,y,z" of every cell recorded this session
  let agentState = 'connecting';
  let lastAction = null;
  let followMode = 'holding';
  let liveStatus = null;          // the mind's newest fresh snapshot (or null)
  let viewerPort = null;          // the port the agent camera ACTUALLY bound (null = down)
  let viewerError = null;         // why the camera is down, readable from the dashboard
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

  let lastBackoffMs = 0;
  function followTick() {
    if (agentState !== 'present') return;
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
    if (d > FOLLOW_FAR) {
      if (followMode !== 'following') {
        setGoal(new goals.GoalFollow(human.entity, FOLLOW_NEAR + 1), true);
        followMode = 'following';
      }
    } else if (d < FOLLOW_NEAR) {
      if (now - lastBackoffMs > 750) {
        const away = me.minus(them).normalize().scaled(FOLLOW_NEAR + 1 - d);
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
  function liveTick() {
    liveStatus = readLiveStatus();
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
  function scanTick() {
    if (agentState !== 'present' || !bot.entity || !bot.entity.position) return;
    const e = bot.entity;
    const eye = e.position.offset(0, EYE_HEIGHT, 0);
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
          hit = bot.world.raycast(eye, dir, SCAN_RANGE);
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
    console.log(`[${NAME}] joined ${HOST}:${port} (mc ${VERSION}) — shadowing, placement disabled`);
    bot.chat(`${NAME} online — I'll follow and watch; placement disabled (D5 v1.1).`);
    ensureFlowViz();
    writeStatus();

    // The body may move (pathfinder) but must never change the world: no digging,
    // no scaffold placement, no tower-building to reach a path.
    const movements = new Movements(bot);
    movements.canDig = false;
    movements.allow1by1towers = false;
    movements.scafoldingBlocks = [];
    movements.placeCost = 1000000;      // belt and braces: placing is never worth it
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

    if (process.env.MICA_A7_TEST === '1') setTimeout(a7TestPlacement, 10000);
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
