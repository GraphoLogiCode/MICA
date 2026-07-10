'use strict';
// The rig automation watcher. Start it once (start-rig.bat, npm run lan, or the
// optional logon task from scripts/install_rig_task.ps1) and leave it alone;
// everything else is event-driven:
//
//   MINECRAFT LAUNCHED   the capture mod writes fabric-*.manifest.json at game
//                        launch — a new manifest in any capture root is the
//                        trigger. Only then does run_live start (GPU models load
//                        during your launcher/menu time; VRAM is FREE while no
//                        game is up). fs.watch per root + a poll fallback.
//
//   OPEN TO LAN          Minecraft announces [AD]port[/AD] on multicast
//                        224.0.2.60:4445 — that spawns agent.js (the BODY) with
//                        the discovered port. One agent per world, cooldown on exit.
//
//   WORLD CLOSED         run_live (the MIND) exits at socket EOF. The watcher then
//                        runs the POST-SESSION CHAIN for that session (see
//                        POST_SESSION_STEPS below) and re-arms run_live only while
//                        javaw.exe is still alive — when Minecraft is gone, the GPU
//                        is released until the next launch.
//
// DEDUPE: one watcher per machine (exclusive localhost lock port); one run_live /
// one agent at a time (slot guards); one chain per session ever (in-memory set +
// a chain_done scan of the log at startup — a watcher restart never re-processes).
// The CASCADE RETRAIN IS NEVER FIRED FROM HERE — run_cascade_a.py stays a human
// decision by pinned design; the chain only logs readiness.
//
// LOG: capture/raw/rig_log.jsonl — one JSON line per event (trigger time, run
// status, completed steps, failures). The console keeps the tagged live output.
//
// Env passthrough: everything agent.js reads (MICA_A7_TEST, MICA_AGENT_NAME,
// MICA_RAW_DIRS, ...) works here too. MICA_PYTHON overrides the python command.
// MICA_PIXELS and MICA_H3D both default ON (set '0' to disable) — the rig runs
// the FULL demo config: both model channels + the calibrated v1 heads.

const dgram = require('dgram');
const fs = require('fs');
const net = require('net');
const path = require('path');
const { spawn, execFile } = require('child_process');

const MICA_ROOT = path.join(__dirname, '..', '..');
const PYTHON = process.env.MICA_PYTHON || 'python';
// Every model weight is local; nothing under the rig ever needs the HF Hub live.
// Set process-wide so EVERY child inherits it — run_live AND the post-session
// chain (a chain step's model load once died mid-run on the Hub round-trip).
if (!('HF_HUB_OFFLINE' in process.env)) process.env.HF_HUB_OFFLINE = '1';
// eager (default): run_live starts NOW and stays warm — models are always loaded
// when a session attaches, whatever order you start things in. lazy: run_live
// starts only when a Minecraft capture manifest appears and the GPU is released
// between play periods (set MICA_PREWARM=lazy if you want the VRAM back).
const PREWARM = (process.env.MICA_PREWARM || 'eager').toLowerCase();
const RESTART_COOLDOWN_MS = 10000;   // pause before a child slot restarts / re-arms
const LOCK_PORT = 45677;             // exclusive: the one-watcher-per-machine lock
const MANIFEST_POLL_MS = 5000;       // fallback sweep when fs.watch misses an event
const RAW_ROOTS = (process.env.MICA_RAW_DIRS
  || path.join(MICA_ROOT, 'capture', 'raw')).split(';').map(s => s.trim()).filter(Boolean);
const LOG_PATH = path.join(RAW_ROOTS[0], 'rig_log.jsonl');

// The post-session chain, in order — data, not code: add or remove steps here.
// Each step is a python script under scripts/ with args built from the session id.
// A non-zero exit stops THAT session's chain (logged step_fail); the watcher lives on.
const POST_SESSION_STEPS = [
  { name: 'evidence',  script: 'after_game.py',
    args: (sid) => ['--session', sid, '--evidence-only'] },   // gate + d1 + d2, no label
  { name: 'scan',      script: 'run_agent_scan.py',
    args: (sid) => [sid, '--h3d'] },   // Uni3D over the agent's OWN scan: coverage +
                                       // cosine(h3d_scan, exact h3d); no agent = skip
  { name: 'report',    script: 'make_session_report.py', args: (sid) => [sid] },
  { name: 'readiness', script: 'after_game.py', args: () => ['--status'] },
];

let runLive = null;            // the mind: one at a time (single-consumer socket)
let runLiveSession = null;     // the session id parsed from its output
let agent = null;              // the body: one per LAN world
let agentCooldownUntil = 0;
let shuttingDown = false;

// ------------------------------------------------------------------ the log

function log(event, extra) {
  const row = { ts: new Date().toISOString(), event, ...(extra || {}) };
  try { fs.appendFileSync(LOG_PATH, JSON.stringify(row) + '\n'); } catch (err) { /* keep going */ }
  const bits = [extra && extra.session, extra && extra.step,
                extra && extra.exit_code !== undefined ? `exit ${extra.exit_code}` : null,
                extra && extra.detail].filter(Boolean);
  console.log(`[rig] ${event}${bits.length ? ' — ' + bits.join(' · ') : ''}`);
}

// One chain per session EVER: sessions whose chain already completed (this run or
// any earlier one, read back from the log) are never re-processed.
const processedSessions = new Set();
function loadProcessedFromLog() {
  let text;
  try { text = fs.readFileSync(LOG_PATH, 'utf8'); } catch (err) { return; }
  for (const line of text.split('\n')) {
    if (!line.trim()) continue;
    try {
      const row = JSON.parse(line);
      if (row.event === 'chain_done' && row.session) processedSessions.add(row.session);
    } catch (err) { /* a torn line never blocks the watcher */ }
  }
}

// ------------------------------------------------------- child output piping

function tagPipe(child, tag, onLine) {
  // One console, tagged lines. run_live redraws its status with \r, so treat
  // \r as a line break too — each redraw just becomes its own log line here.
  let buffers = { stdout: '', stderr: '' };
  for (const stream of ['stdout', 'stderr']) {
    child[stream].setEncoding('utf8');
    child[stream].on('data', (chunk) => {
      buffers[stream] += chunk;
      const lines = buffers[stream].split(/\r\n|\n|\r/);
      buffers[stream] = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        console.log(`[${tag}] ${line}`);
        if (onLine) onLine(line);
      }
    });
  }
}

// ------------------------------------------------- trigger 1: game launched

// A manifest present before the watcher started is old news; only APPEARANCE
// after startup is a launch. (If Minecraft is already mid-game at startup, the
// javaw check below arms run_live anyway.)
const seenManifests = new Set();
function listManifests(root) {
  try {
    return fs.readdirSync(root)
      .filter((name) => /^fabric-.*\.manifest\.json$/.test(name))
      .map((name) => path.join(root, name));
  } catch (err) { return []; }
}
function primeSeenManifests() {
  for (const root of RAW_ROOTS) for (const full of listManifests(root)) seenManifests.add(full);
}
function checkForNewManifest() {
  for (const root of RAW_ROOTS) {
    for (const full of listManifests(root)) {
      if (seenManifests.has(full)) continue;
      seenManifests.add(full);
      const sid = path.basename(full).slice(0, -'.manifest.json'.length);
      log('minecraft_detected', { session: sid, detail: 'capture manifest appeared' });
      spawnRunLive();
    }
  }
}

function minecraftAlive(callback) {
  execFile('tasklist', ['/FI', 'IMAGENAME eq javaw.exe', '/FO', 'CSV', '/NH'],
    (err, stdout) => callback(!err && /javaw\.exe/i.test(stdout || '')));
}

// ------------------------------------------------------- the mind: run_live

function spawnRunLive() {
  if (shuttingDown || runLive !== null) return;   // one mind at a time
  // The full demo config: both GPU model channels pre-warmed, plus the trained v1
  // belief heads (D5 §9: the demo runs on the heads its thresholds were frozen
  // against). Your own environment still wins — the defaults sit BEFORE the
  // ...process.env spread.
  // HF_HUB_OFFLINE: every model weight is on disk, but the MineCLIP/VPT load
  // path still pings the HF Hub each start ("unauthenticated requests" warning).
  // Offline mode reads the same local caches without the network round-trip;
  // verified the full GPU load works under it. Downloads (setup_uni3d.py) run
  // outside the rig, so nothing here ever needs the Hub live.
  runLive = spawn(PYTHON, ['scripts/run_live.py', '--wait-session', '--heads', 'v1'],
    { cwd: MICA_ROOT, stdio: ['ignore', 'pipe', 'pipe'],
      env: { MICA_PIXELS: '1', MICA_H3D: '1', HF_HUB_OFFLINE: '1', ...process.env } });
  runLiveSession = null;
  tagPipe(runLive, 'run_live', (line) => {
    const match = /live session: (fabric-\S+)/.exec(line);
    if (match && runLiveSession !== match[1]) {
      runLiveSession = match[1];
      log('runlive_attached', { session: runLiveSession });
    }
  });
  runLive.on('error', (err) =>
    log('error', { detail: `could not start run_live (${err.message}) — is ${PYTHON} on PATH?` }));
  runLive.on('exit', (code) => {
    const sid = runLiveSession;
    runLive = null;
    runLiveSession = null;
    log('runlive_exit', { session: sid || undefined,
                          exit_code: code === null ? 'killed' : code });
    if (shuttingDown) return;
    if (sid) runChain(sid);
    if (PREWARM === 'eager') {
      // Always-warm: restart immediately so the models are loaded before the
      // next session, whenever it comes.
      log('runlive_rearm', { detail: `eager pre-warm — restarting in ${RESTART_COOLDOWN_MS / 1000}s` });
      setTimeout(spawnRunLive, RESTART_COOLDOWN_MS);
      return;
    }
    // lazy: re-arm only while the game is still up (a second world may follow);
    // when Minecraft is gone, stay idle — the GPU is free until the next launch.
    minecraftAlive((alive) => {
      if (shuttingDown) return;
      if (alive) {
        log('runlive_rearm', { detail: `javaw still running — next world in ${RESTART_COOLDOWN_MS / 1000}s` });
        setTimeout(spawnRunLive, RESTART_COOLDOWN_MS);
      } else {
        log('minecraft_closed', { detail: 'GPU released; watching for the next launch' });
      }
    });
  });
  log('runlive_start', { detail: 'models pre-warming; attaches when you enter a world' });
}

// --------------------------------------------------------- the body: agent

function spawnAgent(port) {
  log('lan_announced', { detail: `port ${port} — sending in MICA_AI` });
  agent = spawn(process.execPath, ['agent.js'], {
    cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, MICA_PORT: String(port) },
  });
  tagPipe(agent, 'agent');
  agent.on('error', (err) => log('error', { detail: `could not start agent (${err.message})` }));
  agent.on('exit', (code) => {
    agent = null;
    agentCooldownUntil = Date.now() + RESTART_COOLDOWN_MS;
    log('agent_exit', { exit_code: code === null ? 'killed' : code,
                        detail: 'the next Open to LAN relaunches it' });
  });
  log('agent_start', { detail: `joining on port ${port}` });
}

// ------------------------------------------------- the post-session chain

let chainRunning = false;
const chainQueue = [];
const CHAIN_RECHECK_MS = 30000;    // a menu exit re-checks until the manifest finalizes
const MAX_CHAIN_DEFERRALS = 40;    // ~20 min; then run anyway and let the B0 gate judge
const deferring = new Set();       // sessions with a recheck timer already pending
const chainDeferrals = {};         // sid -> how many times we waited

function manifestFinalized(sid) {
  // The mod writes declared_event_count only when the session truly closes —
  // THE "session over" signal. A socket EOF is not one: exiting to the menu
  // closes the socket, but re-entering the world CONTINUES the same session.
  for (const root of RAW_ROOTS) {
    try {
      const meta = JSON.parse(fs.readFileSync(
        path.join(root, `${sid}.manifest.json`), 'utf8'));
      return Number(meta.declared_event_count ?? -1) >= 0;
    } catch (err) { /* not in this root (or mid-write): try the next */ }
  }
  return false;
}

function runChain(sid) {
  if (processedSessions.has(sid)) {
    log('chain_skipped', { session: sid, detail: 'already processed (dedupe)' });
    return;
  }
  if (deferring.has(sid)) return;   // a recheck loop for this session already runs
  if (!manifestFinalized(sid)) {
    // Chaining here once burned a real session: the mid-game attempt failed on
    // the provisional manifest AND its dedupe mark skipped the real end later.
    // Defer without marking processed; the recheck (or the next world exit)
    // picks it up once the manifest says the session is actually over.
    chainDeferrals[sid] = (chainDeferrals[sid] || 0) + 1;
    if (chainDeferrals[sid] <= MAX_CHAIN_DEFERRALS) {
      log('chain_deferred', { session: sid,
        detail: `manifest not finalized (session still open?) — recheck in ${CHAIN_RECHECK_MS / 1000}s` });
      deferring.add(sid);
      setTimeout(() => { deferring.delete(sid); runChain(sid); }, CHAIN_RECHECK_MS);
      return;
    }
    log('chain_forced', { session: sid,
      detail: 'manifest never finalized (crash?) — running anyway; the gate will say so' });
  }
  processedSessions.add(sid);
  chainQueue.push(sid);
  drainChain();
}

function drainChain() {
  if (chainRunning || chainQueue.length === 0) return;
  chainRunning = true;
  runStep(chainQueue.shift(), 0, () => { chainRunning = false; drainChain(); });
}

function runStep(sid, index, done) {
  if (index >= POST_SESSION_STEPS.length) {
    log('chain_done', { session: sid });
    return done();
  }
  const step = POST_SESSION_STEPS[index];
  log('step_start', { session: sid, step: step.name });
  const child = spawn(PYTHON, [path.join('scripts', step.script), ...step.args(sid)],
    { cwd: MICA_ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
  tagPipe(child, step.name);
  child.on('error', (err) => {
    log('step_fail', { session: sid, step: step.name, detail: err.message });
    done();
  });
  child.on('exit', (code) => {
    if (code === 0) {
      log('step_done', { session: sid, step: step.name });
      runStep(sid, index + 1, done);
    } else {
      log('step_fail', { session: sid, step: step.name,
                         exit_code: code === null ? 'killed' : code,
                         detail: 'chain stopped for this session; watcher keeps running' });
      done();   // no chain_done: a watcher restart may retry this session
    }
  });
}

// ------------------------------------------------------------------ startup

// One watcher per machine: an exclusive localhost port is the lock (the UDP
// socket below uses reuseAddr for multicast, so it cannot be the lock).
const lock = net.createServer();
lock.on('error', () => {
  console.error('[rig] another watcher already holds the lock — exiting');
  process.exit(0);
});
lock.listen(LOCK_PORT, '127.0.0.1', () => {
  loadProcessedFromLog();
  primeSeenManifests();
  log('watcher_start', {
    detail: `roots ${RAW_ROOTS.join(';')} — waiting for Minecraft (manifest) + Open to LAN (multicast)` });

  const sock = dgram.createSocket({ type: 'udp4', reuseAddr: true });
  sock.on('message', (msg) => {
    const match = /\[AD\](\d+)\[\/AD\]/.exec(msg.toString());
    if (!match) return;
    if (agent === null && Date.now() >= agentCooldownUntil) {
      spawnAgent(parseInt(match[1], 10));
    }
  });
  sock.on('error', (err) => {
    log('error', { detail: `multicast listen failed (${err.message})` });
    process.exit(1);
  });
  sock.bind(4445, () => {
    try { sock.addMembership('224.0.2.60'); } catch (e) {
      log('error', { detail: 'could not join the LAN multicast group — is multicast disabled?' });
    }
  });

  for (const root of RAW_ROOTS) {
    try { fs.watch(root, () => checkForNewManifest()); } catch (err) { /* poll covers it */ }
  }
  setInterval(checkForNewManifest, MANIFEST_POLL_MS);

  if (PREWARM === 'eager') {
    // Always-warm (the default): load the models NOW, before Minecraft even
    // launches — an attach can never catch them cold, whatever order you start
    // things in. (A real session attached mid-load once and its first 14 scored
    // records passed unenriched.) MICA_PREWARM=lazy trades this for free VRAM.
    log('prewarm', { detail: 'eager — loading models now (MICA_PREWARM=lazy to idle instead)' });
    spawnRunLive();
  } else {
    // lazy: watcher started while a game is already up (or mid-session
    // restart) — the manifest is old news, but the game isn't: arm now.
    minecraftAlive((alive) => {
      if (alive) {
        log('minecraft_detected', { detail: 'javaw already running at watcher start' });
        spawnRunLive();
      }
    });
  }
});

// Ctrl+C takes the children down with the watcher.
process.on('SIGINT', () => {
  shuttingDown = true;
  for (const child of [runLive, agent]) {
    if (child) { try { child.kill(); } catch (e) {} }
  }
  process.exit(0);
});
