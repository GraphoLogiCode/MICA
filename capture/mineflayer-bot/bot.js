'use strict';
// Observer bot: joins the local server as a second client and writes the
// server-visible half of B0 to a JSONL file — block changes, plus the watched
// player's position, look, and held item. It CANNOT see the human's keys/mouse,
// crosshair, hotbar, menu, or screen; those are client-only and need the mod.
// Those fields are written as null, so the coverage check reports them honestly.
//
// It also serves a live 3D dashboard (no native deps): a 3D grid of what it sees
// on the left, logs + latency on the right, at http://localhost:3007. And it
// follows the watched player so their builds stay in range.
//
// Env: MICA_HOST, MICA_PORT, MICA_VERSION, MICA_RUN_SECONDS (0 = until Ctrl+C),
//      MICA_VIEWER_PORT (default 3007).

const fs = require('fs');
const path = require('path');
const http = require('http');
const mineflayer = require('mineflayer');

const HOST = process.env.MICA_HOST || '127.0.0.1';
const PORT = parseInt(process.env.MICA_PORT || '25565', 10);
const VERSION = process.env.MICA_VERSION || '1.16.5';
const RUN_SECONDS = parseInt(process.env.MICA_RUN_SECONDS || '0', 10);
const VIEWER_PORT = parseInt(process.env.MICA_VIEWER_PORT || '3007', 10);

const sessionId = 'mineflayer-' + new Date().toISOString().replace(/[:.]/g, '-');
const rawDir = path.join(__dirname, '..', 'raw');
fs.mkdirSync(rawDir, { recursive: true });
const jsonlPath = path.join(rawDir, sessionId + '.jsonl');
const manifestPath = path.join(rawDir, sessionId + '.manifest.json');
const out = fs.createWriteStream(jsonlPath, { flags: 'a' });

let threeJs = null;
try {
  threeJs = fs.readFileSync(path.join(__dirname, 'static', 'three.min.js'));
} catch (err) {
  console.error('[MICA] static/three.min.js missing — the 3D view needs it (re-run the vendor step).');
}

const bot = mineflayer.createBot({
  host: HOST, port: PORT, username: 'MICA-Recorder', version: VERSION, auth: 'offline',
});

let sessionStartMs = null;
let tickIndex = 0;       // our own contiguous moment counter (one per physics tick)
let eventCounter = 0;    // a unique number for each block change we see
let pending = [];        // block changes seen since the last moment was written
let statusTimer = null;
let followTimer = null;
let done = false;

// --- live dashboard: a 3D page plus a server-sent-event stream ---
const dashClients = new Set();
function broadcast(obj) {
  const line = `data: ${JSON.stringify(obj)}\n\n`;
  for (const res of dashClients) {
    try { res.write(line); } catch (err) { /* client went away */ }
  }
}

// The page avoids backticks and ${} so it can live safely inside this template.
const DASHBOARD_HTML = `<!doctype html><html><head><meta charset="utf-8">
<title>MICA-Recorder</title><style>
  body{margin:0;background:#0d1117;color:#c9d1d9;font:13px ui-monospace,Consolas,monospace;overflow:hidden}
  #wrap{display:flex;flex-direction:row;height:100vh}
  #left{display:flex;flex-direction:column;padding:10px;flex:1;min-width:0}
  h1{font-size:14px;margin:0 0 8px 0;color:#58a6ff}
  #view{flex:1;min-height:0;border:1px solid #30363d;background:#010409}
  #meta{padding:8px 0 0 0;color:#8b949e}
  #right{flex:0 0 360px;display:flex;flex-direction:column;border-left:1px solid #30363d}
  #lat{padding:8px 12px;color:#d29922;border-bottom:1px solid #30363d}
  #log{flex:1;overflow:auto;padding:10px 12px;white-space:pre-wrap}
  .place{color:#3fb950}.break{color:#f85149}.status{color:#8b949e}
</style></head><body><div id="wrap">
  <div id="left"><h1>MICA-Recorder — 3D live view</h1><div id="view"></div><div id="meta">connecting…</div></div>
  <div id="right"><div id="lat">latency: waiting…</div><div id="log"></div></div>
</div>
<script src="/static/three.min.js"></script>
<script>
  var view=document.getElementById('view'),log=document.getElementById('log'),meta=document.getElementById('meta'),lat=document.getElementById('lat');
  var scene=new THREE.Scene();scene.background=new THREE.Color(0x010409);
  var camera=new THREE.PerspectiveCamera(60,1,0.1,4000);
  var renderer=new THREE.WebGLRenderer({antialias:true});view.appendChild(renderer.domElement);
  function fit(){var w=view.clientWidth||560,h=view.clientHeight||560;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
  window.addEventListener('resize',fit);fit();
  scene.add(new THREE.AmbientLight(0xffffff,0.9));
  var dl=new THREE.DirectionalLight(0xffffff,0.5);dl.position.set(1,2,1);scene.add(dl);
  // Open Minecraft-style space: fog fades the far edges so it never looks boxed.
  scene.fog=new THREE.Fog(0x010409,80,300);
  // A solid floor plane plus a large grid, both locked to the ground (y=floorY).
  var GRID=256,floorY=null;
  var ground=new THREE.Mesh(new THREE.PlaneGeometry(GRID,GRID),new THREE.MeshLambertMaterial({color:0x0b1622}));
  ground.rotation.x=-Math.PI/2;scene.add(ground);
  var grid=new THREE.GridHelper(GRID,GRID,0x2f4060,0x18222e);scene.add(grid);
  var axisLen=16,axes=new THREE.AxesHelper(axisLen);scene.add(axes);
  function mkLabel(t,c){var cv=document.createElement('canvas');cv.width=cv.height=64;var g=cv.getContext('2d');g.fillStyle=c;g.font='bold 44px monospace';g.textAlign='center';g.textBaseline='middle';g.fillText(t,32,32);var s=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv),depthTest:false}));s.scale.set(4,4,4);scene.add(s);return s;}
  var lblX=mkLabel('X','#f85149'),lblY=mkLabel('Y','#3fb950'),lblZ=mkLabel('Z','#58a6ff');
  var blocks={},cube=new THREE.BoxGeometry(1,1,1),green=new THREE.MeshLambertMaterial({color:0x3fb950});
  function setBlock(x,y,z,op){var k=x+','+y+','+z;
    if(op==='break'){if(blocks[k]){scene.remove(blocks[k]);delete blocks[k];}return;}
    if(blocks[k])return;var m=new THREE.Mesh(cube,green);m.position.set(x+0.5,y+0.5,z+0.5);scene.add(m);blocks[k]=m;}
  var player=null;
  var pmark=new THREE.Mesh(new THREE.SphereGeometry(0.4,12,12),new THREE.MeshBasicMaterial({color:0x58a6ff}));scene.add(pmark);
  var rg=new THREE.BufferGeometry(),rp=new Float32Array(6);rg.setAttribute('position',new THREE.BufferAttribute(rp,3));
  scene.add(new THREE.Line(rg,new THREE.LineBasicMaterial({color:0x58a6ff})));
  var target=new THREE.Vector3(0,64,0),az=0.7,el=0.45,dist=44;
  function cam(){camera.position.set(target.x+dist*Math.cos(el)*Math.sin(az),target.y+dist*Math.sin(el),target.z+dist*Math.cos(el)*Math.cos(az));camera.lookAt(target);}
  var drag=false,lx=0,ly=0;
  renderer.domElement.addEventListener('mousedown',function(e){drag=true;lx=e.clientX;ly=e.clientY;});
  window.addEventListener('mouseup',function(){drag=false;});
  window.addEventListener('mousemove',function(e){if(!drag)return;az-=(e.clientX-lx)*0.01;el+=(e.clientY-ly)*0.01;el=Math.max(-1.4,Math.min(1.4,el));lx=e.clientX;ly=e.clientY;});
  renderer.domElement.addEventListener('wheel',function(e){dist*=(e.deltaY>0?1.1:0.9);dist=Math.max(4,Math.min(400,dist));e.preventDefault();});
  function loop(){requestAnimationFrame(loop);
    if(player){pmark.position.set(player.x,player.y+1,player.z);
      var p=player.pitch||0,dx=-Math.sin(player.yaw)*Math.cos(p),dy=-Math.sin(p),dz=Math.cos(player.yaw)*Math.cos(p);
      rp[0]=player.x;rp[1]=player.y+1;rp[2]=player.z;rp[3]=player.x+dx*4;rp[4]=player.y+1+dy*4;rp[5]=player.z+dz*4;rg.attributes.position.needsUpdate=true;
      if(floorY===null)floorY=Math.floor(player.y);else floorY=Math.min(floorY,Math.floor(player.y));
      // Floor + grid stay on the ground plane (fixed y) and recenter on whole
      // blocks under the player, so the lines read as a stable, open floor.
      var gx=Math.round(player.x),gz=Math.round(player.z);
      grid.position.set(gx,floorY,gz);ground.position.set(gx,floorY-0.02,gz);axes.position.set(gx,floorY,gz);
      lblX.position.set(gx+axisLen+1.5,floorY,gz);lblY.position.set(gx,floorY+axisLen+1.5,gz);lblZ.position.set(gx,floorY,gz+axisLen+1.5);
      target.set(player.x,player.y+1,player.z);}
    cam();renderer.render(scene,camera);}
  loop();
  var lats=[];
  function lag(ms){lats.push(ms);if(lats.length>120)lats.shift();
    var mn=Math.min.apply(null,lats),mx=Math.max.apply(null,lats),av=lats.reduce(function(a,b){return a+b;},0)/lats.length;
    lat.textContent='capture\\u2192view latency: '+ms.toFixed(0)+' ms  (min '+mn.toFixed(0)+' / avg '+av.toFixed(0)+' / max '+mx.toFixed(0)+')';}
  function addLog(c,t){var d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);while(log.childNodes.length>250)log.removeChild(log.firstChild);log.scrollTop=log.scrollHeight;}
  var es=new EventSource('/events');
  es.onmessage=function(ev){var m=JSON.parse(ev.data);if(m.captureMs)lag(Date.now()-m.captureMs);
    if(m.type==='status'){player=m.player;
      meta.textContent='watching '+m.player.name+' @ ('+m.player.x.toFixed(1)+','+m.player.y.toFixed(1)+','+m.player.z.toFixed(1)+')  holding '+(m.player.held||'nothing')+'   | moments '+m.tick+'  changes '+m.changes;}
    else if(m.type==='block'){setBlock(m.x,m.y,m.z,m.op);addLog(m.op,'#'+m.id+' '+m.op+' '+m.block_type+' @ ('+m.x+','+m.y+','+m.z+')');}};
  es.onerror=function(){meta.textContent='disconnected (bot stopped?)';};
</script></body></html>`;

const httpServer = http.createServer((req, res) => {
  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(DASHBOARD_HTML);
  } else if (req.url === '/static/three.min.js') {
    if (threeJs) {
      res.writeHead(200, { 'content-type': 'application/javascript' });
      res.end(threeJs);
    } else {
      res.writeHead(404);
      res.end();
    }
  } else if (req.url === '/events') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.write('\n');
    dashClients.add(res);
    req.on('close', () => dashClients.delete(res));
  } else {
    res.writeHead(404);
    res.end();
  }
});
// Serve the page immediately so it can be opened before the bot finishes connecting.
httpServer.listen(VIEWER_PORT, () => console.log(`[MICA] live 3D screen + logs at http://localhost:${VIEWER_PORT}`));

function namespaced(name) {
  if (!name) return null;
  return name.includes(':') ? name : 'minecraft:' + name;
}

function watchedEntity() {
  // Prefer a human (any player but us); fall back to the bot itself.
  for (const name of Object.keys(bot.players)) {
    const player = bot.players[name];
    if (name !== bot.username && player && player.entity) return { name, entity: player.entity };
  }
  return { name: bot.username, entity: bot.entity };
}

bot.on('blockUpdate', (oldBlock, newBlock) => {
  if (!newBlock || !newBlock.position) return;
  const captureMs = Date.now();
  const becameAir = newBlock.name === 'air';
  const blockType = namespaced(becameAir ? (oldBlock && oldBlock.name) : newBlock.name) || 'minecraft:air';
  const op = becameAir ? 'break' : 'place';
  const p = newBlock.position;
  const id = eventCounter++;
  pending.push({ event_id: id, pos: [p.x, p.y, p.z], block_type: blockType, op, actor: watchedEntity().name });
  console.log(`[MICA] #${id} ${op} ${blockType} @ (${p.x},${p.y},${p.z})`);
  broadcast({ type: 'block', id, op, block_type: blockType, x: p.x, y: p.y, z: p.z, captureMs });
});

bot.once('spawn', () => {
  sessionStartMs = Date.now();
  console.log(`[MICA] connected to ${HOST}:${PORT} as ${bot.username} (mc ${VERSION})`);
  console.log(`[MICA] writing ${jsonlPath}`);

  statusTimer = setInterval(() => {
    const watched = watchedEntity();
    const e = watched.entity;
    if (!e || !e.position) return;
    const held = e.heldItem ? namespaced(e.heldItem.name) : null;
    console.log(`[MICA] moments=${tickIndex} changes=${eventCounter} | watching ${watched.name} @ (${e.position.x.toFixed(1)},${e.position.y.toFixed(1)},${e.position.z.toFixed(1)}) holding ${held || 'nothing'}`);
    broadcast({ type: 'status', tick: tickIndex, changes: eventCounter, captureMs: Date.now(), player: { name: watched.name, x: e.position.x, y: e.position.y, z: e.position.z, yaw: e.yaw, pitch: e.pitch, held } });
  }, 1000);

  // Crude follow: face the watched human and walk toward them so their builds
  // stay inside the chunks we have loaded. Fine on flat creative ground.
  followTimer = setInterval(() => {
    const watched = watchedEntity();
    if (watched.name === bot.username || !watched.entity) return;
    const target = watched.entity.position;
    bot.lookAt(target.offset(0, 1.6, 0));
    const far = bot.entity.position.distanceTo(target) > 4;
    bot.setControlState('forward', far);
    bot.setControlState('jump', far);
  }, 400);

  if (RUN_SECONDS > 0) setTimeout(shutdown, RUN_SECONDS * 1000);
});

bot.on('physicsTick', () => {
  if (sessionStartMs === null || done) return;
  const now = Date.now();
  const entity = watchedEntity().entity;
  const pos = entity && entity.position ? [entity.position.x, entity.position.y, entity.position.z] : null;
  const held = entity && entity.heldItem ? namespaced(entity.heldItem.name) : null;
  const changes = pending;
  pending = [];
  out.write(JSON.stringify({
    tick: tickIndex++,
    wallclock_ms: now,
    client: {
      capture_wallclock_ms: now,
      pov_frame: null,
      input_state: null,
      yaw: entity ? entity.yaw : 0.0,
      pitch: entity ? entity.pitch : 0.0,
      crosshair_target: null,
      held_item: held,
      hotbar: null,
      gui_open: null,
    },
    server: {
      player_pos: pos,
      block_events: changes,
      inventory_delta: [],
      dimension: bot.game && bot.game.dimension ? namespaced(bot.game.dimension) : 'minecraft:overworld',
      biome: 'unknown',
    },
  }) + '\n');
});

function shutdown() {
  if (done) return;
  done = true;
  if (statusTimer) clearInterval(statusTimer);
  if (followTimer) clearInterval(followTimer);
  fs.writeFileSync(manifestPath, JSON.stringify({
    session_id: sessionId,
    session_start_ms: sessionStartMs || Date.now(),
    mc_version: VERSION,
    mod_version: 'mineflayer-observer-0.0.1',
    declared_event_count: eventCounter,
  }, null, 2));
  out.end();
  console.log(`[MICA] wrote ${tickIndex} moments, ${eventCounter} block changes`);
  setTimeout(() => process.exit(0), 250);
}

process.on('SIGINT', shutdown);
bot.on('error', (err) => console.error('[MICA] error:', err.message));
bot.on('kicked', (reason) => console.error('[MICA] kicked:', reason));
bot.on('end', () => { if (!done) shutdown(); });
