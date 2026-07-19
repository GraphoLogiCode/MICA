// Launch smoke check for agent.js (no server needed):
//     node capture/mineflayer-bot/test_agent_launch.js
// `node --check` only parses — it cannot catch runtime startup crashes like the
// 2026-07-19 temporal-dead-zone ReferenceError (writeStatus ran at launch and
// read a variable declared further down). This actually LAUNCHES the agent
// against a dead port: the startup path must complete, and the only acceptable
// failure is the connection refusal itself.

'use strict';

const assert = require('assert');
const path = require('path');
const { spawn } = require('child_process');

const child = spawn(process.execPath, [path.join(__dirname, 'agent.js')], {
  env: { ...process.env, MICA_PORT: '9', MICA_QUIET: '1' },
  stdio: ['ignore', 'pipe', 'pipe'],
});
let out = '';
child.stdout.on('data', (chunk) => { out += chunk; });
child.stderr.on('data', (chunk) => { out += chunk; });

const killTimer = setTimeout(() => child.kill('SIGKILL'), 8000);
child.on('exit', () => {
  clearTimeout(killTimer);
  assert.ok(!/ReferenceError|SyntaxError|TypeError/.test(out),
    'agent.js crashed during startup:\n' + out.slice(0, 800));
  assert.ok(/ECONNREFUSED/.test(out),
    'expected the dead-port refusal (did the agent even try to join?):\n'
    + out.slice(0, 800));
  console.log('agent launch: startup path clean (only the dead port refused)');
});
