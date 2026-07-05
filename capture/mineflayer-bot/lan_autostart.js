'use strict';
// The one-command rig launcher. Run this once (npm run lan, or start-rig.bat at
// the repo root) and leave it open; then just play.
//
// TWO halves, two lifecycles:
//
//   python scripts/run_live.py --wait-session   — the MIND. Started IMMEDIATELY so
//        both GPU models (VPT/MineCLIP, and Uni3D with MICA_H3D=1) load while you
//        are still in the launcher/menus, and the session's very first records get
//        enriched (a real session once spent its opening ~40 s "px:loading").
//        It waits for a session by itself, attaches the moment you are in a world,
//        exits when the world closes, and is restarted here after a short pause —
//        never two at once (the mod's live socket is single-consumer).
//
//   node agent.js                                — the BODY. MICA_AI can only join a
//        LAN world, so it launches when Minecraft announces one on multicast
//        224.0.2.60:4445 (Esc -> Open to LAN) and gets the discovered port directly.
//        When it exits (world closed / kicked), the next announcement relaunches it
//        after a short cooldown.
//
// Both children's output shows here, line-tagged. Env passthrough: everything
// agent.js reads (MICA_A7_TEST, MICA_AGENT_NAME, MICA_RAW_DIRS, ...) works here
// too. MICA_PYTHON overrides the python command. MICA_PIXELS defaults ON;
// MICA_H3D passes through (default off — export MICA_H3D=1 to enable).

const dgram = require('dgram');
const path = require('path');
const { spawn } = require('child_process');

const MICA_ROOT = path.join(__dirname, '..', '..');
const PYTHON = process.env.MICA_PYTHON || 'python';
const RESTART_COOLDOWN_MS = 10000;   // pause before a child slot restarts / re-arms

let runLive = null;            // the mind: persistent, restarted after each exit
let agent = null;              // the body: one per LAN world
let agentCooldownUntil = 0;
let shuttingDown = false;

function tagPipe(child, tag) {
  // One console, tagged lines. run_live redraws its status with \r, so treat
  // \r as a line break too — each redraw just becomes its own log line here.
  let buffers = { stdout: '', stderr: '' };
  for (const stream of ['stdout', 'stderr']) {
    child[stream].setEncoding('utf8');
    child[stream].on('data', (chunk) => {
      buffers[stream] += chunk;
      const lines = buffers[stream].split(/\r\n|\n|\r/);
      buffers[stream] = lines.pop();
      for (const line of lines) if (line.trim()) console.log(`[${tag}] ${line}`);
    });
  }
}

function spawnRunLive() {
  if (shuttingDown) return;
  runLive = spawn(PYTHON, ['scripts/run_live.py', '--wait-session'],
    { cwd: MICA_ROOT, stdio: ['ignore', 'pipe', 'pipe'],
      env: { MICA_PIXELS: '1', ...process.env } });
  tagPipe(runLive, 'run_live');
  runLive.on('error', (err) =>
    console.error(`[rig] could not start run_live (${err.message}) — is ${PYTHON} on PATH?`));
  runLive.on('exit', (code) => {
    runLive = null;
    if (shuttingDown) return;
    console.log(`[rig] run_live exited (${code === null ? 'killed' : code})`
      + ` — restarting in ${RESTART_COOLDOWN_MS / 1000}s`);
    setTimeout(spawnRunLive, RESTART_COOLDOWN_MS);
  });
  console.log('[rig] run_live started (models pre-warming; attaches when you enter a world)');
}

function spawnAgent(port) {
  console.log(`[rig] LAN world announced on port ${port} — sending in MICA_AI`);
  agent = spawn(process.execPath, ['agent.js'], {
    cwd: __dirname, stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, MICA_PORT: String(port) },
  });
  tagPipe(agent, 'agent');
  agent.on('error', (err) => console.error(`[rig] could not start agent (${err.message})`));
  agent.on('exit', (code) => {
    agent = null;
    agentCooldownUntil = Date.now() + RESTART_COOLDOWN_MS;
    console.log(`[rig] agent exited (${code === null ? 'killed' : code})`
      + ' — the next Open to LAN relaunches it');
  });
}

const sock = dgram.createSocket({ type: 'udp4', reuseAddr: true });
sock.on('message', (msg) => {
  const match = /\[AD\](\d+)\[\/AD\]/.exec(msg.toString());
  if (!match) return;
  if (agent === null && Date.now() >= agentCooldownUntil) {
    spawnAgent(parseInt(match[1], 10));
  }
});
sock.on('error', (err) => {
  console.error(`[rig] multicast listen failed (${err.message})`);
  process.exit(1);
});
sock.bind(4445, () => {
  try { sock.addMembership('224.0.2.60'); } catch (e) {
    console.error('[rig] could not join the LAN multicast group — is multicast disabled?');
  }
  console.log('[rig] watching for Open to LAN ... (play singleplayer, then Esc -> Open to LAN)');
});

spawnRunLive();

// Ctrl+C takes the children down with the watcher.
process.on('SIGINT', () => {
  shuttingDown = true;
  for (const child of [runLive, agent]) {
    if (child) { try { child.kill(); } catch (e) {} }
  }
  process.exit(0);
});
