# MICA — Code Audit & Issue Tracker

> Issues found auditing the implemented code (B0 capture + B1/D1). **Most were fixed and verified on
> 2026-06-29** (see RESOLVED below). What remains needs either your in-game rebuild check, or is
> deferred-by-design for a later phase — collected in REVIEW LATER. The vault mirror
> (`Implementation Log/Issues & Audit Tracker`) is a dated snapshot and is now behind this file.
>
> Status: `[ ]` open · `[x]` done · `[~]` deferred-by-design.

---

# ⚠️ REVIEW LATER — what still needs a look

## A2. Mod 0.0.3 (D2 prerequisites) — **ALL VERIFIED 2026-07-02 evening (session `194242`)**
Jar 1.1.0 built 19:41; verification session: 5185 ticks, 9 events, 4 snapshots. The strong check was a
mini Proposition-1 replay: events applied to snapshot 0 vs later snapshots — **117,647 of 117,649 cells
match**, and both divergences are the pre-documented classes, not capture faults.
- [x] **R-3 — authoritative place.** *Verified by replay: zero phantom or mislocated placements (a
  phantom would necessarily diverge; a mislocation would diverge in pairs — none). All 9 events land
  exactly where the world says.*
- [x] **R-4 — region snapshots.** *4 files (initial @0, post-burst @225/@536/@5024); palette+RLE decodes
  to exactly the region volume; manifest carries `snapshot_region`, `snapshot_quiet_ticks: 40`, 0.0.3.*
- [x] **R-5 — gate end-to-end.** *Full PASS incl. "region snapshots (D2): 4 snapshot files"; finalized
  count 9==9; old captures still PASS labeled "not D2-ready".*
- [x] **R-6 — door gap demonstrated.** *Replay flagged `(8,81,-105) air vs oak_door` — the un-evented
  top half, caught by the replay check exactly as documented. Bottom half recorded correctly. (The other
  divergence: water reflowed into a broken cell — the documented fluid class.)*

## A. Mod rebuild + in-game checks — **ALL VERIFIED 2026-07-02, B0 gate CLOSED (full GO for B1)**
Mod rebuilt (jar 07-02 01:03, `fabric-b0-0.0.2`); every check below verified live the same night:
- [x] **M-1 (mod) — `event_schema_version` in the manifest.** *Verified 2026-07-02 on
  `fabric-20260702-010354`: manifest carries `"event_schema_version": "1"`, ingest reads it back.*
- [x] **C-5 — live-stream thread exits cleanly.** *Verified 2026-07-02 live: probe attached
  (ESTABLISHED), game quit normally — process exited, port released, probe ended exit-0, and the
  session manifest (`011224`) was finalized with the consumer attached. Reconnect also verified.*
- [x] **U-5 — live raw-frame channel order.** *Verified 2026-07-02 live with `scripts/live_color_check.py`
  on a red test view: mean R=86 G=24 B=20 — RED dominates, channel order correct end-to-end.*
- [x] **R-1 (new 2026-07-02) — frame height floor + manifest frame settings.** *Verified 2026-07-02 on
  `fabric-20260702-010354`: manifest has `mod_version fabric-b0-0.0.2`, `frame_every: 1`,
  `frame_width_px: 320`; 738/738 frames on disk at 320×172 (aspect-preserved, ≥256×160); gate PASS
  including the new "manifest finalized" check.*
- [x] **R-2 (new 2026-07-02) — provisional-manifest rejection, end to end.** *Verified live: game
  hard-killed mid-session (`Stop-Process -Force`, session `012210`, 35 events) — manifest stayed
  `declared_event_count: -1`, ingest warned, gate FAILED on "manifest finalized (clean stop)" (exit 1)
  while every legacy check still passed — exactly the hole the check closes. Test capture deleted.*

## B. Deferred by design — future-phase work, not bugs (no action now)
- [x] **D-1 — RESOLVED in code (2026-07-02 late): place capture is authoritative.** `BlockItemMixin`
  records inside `BlockItem.place` after the game accepts, reading back the landed block; the predictive
  `UseBlockCallback` hook is deleted. Scope: BlockItem + breaks; doors/beds record one event (documented;
  D2 replay surfaces it). *In-game verification = R-3 above.*
- [x] **D-2 — RESOLVED by route (b) of the 06-27 review:** recording is now authoritative, declared stays
  recorder-derived (self-consistent), and D2's replay-vs-snapshot check is the final missed-change
  arbiter — now actually possible because snapshots exist (R-4).
- [~] **D-3 — alignment residual ≡ 0 in singleplayer** (one clock). Networked capture would need a server
  timestamp. (Now stated in the D0 contract itself, 2026-07-02 amendment.)
- [x] **D-4 — frame format RESOLVED (2026-07-02 decision):** 320-wide / 20 fps / aspect-preserved is the
  *canonical* contract format (not a deferral) — derived from the consumers: MineCLIP 256×160 + VPT 128×128,
  and D2 reads no frames. D0 amended; mod floors at 256×160. Native-res is no longer a target.
- [~] **D-5 — stub fields:** `biome="unknown"`, `mouse_dx/dy=0` (look is in yaw/pitch), `inventory_delta=[]`.
- [~] **D-6 — early-separation curve (D1 headline proof) needs labeled scripted-goal captures.** Code ready.
- [~] **D-7 — `SCAFFOLD` in `A` but folded into `PLACE` in v1.**
- [x] **D-8 — RETIRED 2026-07-04 late night:** the F9 pause was removed entirely in mod 0.0.5 (it
  silently dropped world changes — see the 2026-07-04 late-night entry); with no pause there is no
  straddle.

## C. Working-as-intended (documented, no action)
- [~] **L-2 — a tick-0 build action fails the B0 gate** (`_actions_have_prior_moment`, no tick −1). This is
  *correct* — the very first action has no "before" to score against, so the gate rightly rejects such a
  capture. (Made non-silent by U-1: `run_d1` now flags any unconsumed event.)

---

# ⚠️ tests/ DIRECTORY DELETED + FULLY RECOVERED (2026-07-05, morning)

Between the last green run (~01:15, 173 tests) and the morning sessions, `D:\2026projects\MICA\tests\`
vanished from disk — not via the Recycle Bin, cause UNKNOWN (nothing in the rig or pipeline deletes
it; user to confirm whether it was intentional). FlowViz and all source code untouched.
- [x] **Recovered to 173/173 green**: 17 tracked files restored additively from git HEAD blobs
  (`git show HEAD:path`), then a transcript-replay script reconstructed every post-commit
  Write/Edit from the session logs (104 ops across 30 files). Two files needed hand repair from
  working context (a duplicated test body in test_voxel_replay.py; a half-applied width edit in
  test_pixel_head.py — its pairing edit likely lived in a subagent transcript).
- [ ] **Exposure remains**: the last commit (f9cb9f1) is 2026-07-03 — two days of pipeline work
  exists ONLY in the working tree. Recommendation to the user: a commit checkpoint (their call;
  nothing is committed without their say-so).

# ✅ SESSION 112050 (BRIDGE) CLOSED — REGION v3 PROVEN + SCAN AT COSINE 0.98 (2026-07-05)

- [x] **Region v3's first real session verifies clean**: frame grew 5× mid-build, 28 snapshots,
  a=360 b=0 **c=0**, 0 escapes; D2/D1 PASS (272/272 events, 89% pixel-enriched — the pre-warm).
  Live quarantine was socket drops only (96 ticks, 11 events lost on the WIRE; disk complete —
  the documented lossy-socket limitation; drop trend 25→96 ticks under 3-model load, watch item).
- [x] **Label: infrastructure/bridge (builder) — matcher miss** (habitation/longhouse, fit 0.18
  noise): the real bridge dwarfs the 7×3 template, so the infrastructure TEMPLATE slot is still
  open; no pairs. Free-build value stands.
- [x] **SCAN MILESTONE — the earn-its-place curve now spans the full range**: session-paired scan
  file (run_agent_scan resolver fixed same morning: it had paired the NEXT session's scan — now
  pairs by wallclock window) shows **coverage 114/114 (100%), cosine(h3d_scan, exact h3d) 0.9837**.
  Across three real sessions: 29% → 0.55, 83% → 0.73, 100% → **0.98**. The agent-eye channel
  converges to the exact embedding as visibility completes — the h3d_scan signal is real.

---

# ✅ REGION v3 — GROW-ONLY CAPTURE FRAME (2026-07-05; mod 0.0.6 / jar 1.4.0 DEPLOYED)

The stray-first-block failure voided its second session tonight (004833: 16 escapes; after
205205's 67/75) and the user decided: the region must expand flexibly. Design pinned in the D2
doc (dated v3 supersession) + a D0 touch: **proactive, grow-only** — the frame extends toward
the player (margin 8, step 16, ≥5 quiet ticks, ≤1 growth/2 s, per-axis cap 145) BEFORE they can
build there, and each growth immediately snapshots the enlarged region: that file is both a
comparison point for the old cells and the adopted cells' pre-build BASE. Prop 1 survives
(monotone frame, base-before-event per adopted cell; features are absolute-coordinate, so
growth only ADMITS cells). **Escape redefined: outside the frame in force at ADJUDICATION**
(next comparison/drain) — growth absolves the chased builder; the cap keeps runaways honest.

- [x] Core: `Region.covers`, `ReplayWorld.grow` (superset-or-refuse) + `extend_base`
  (setdefault + changed-cell guard so a margin-race placement is never swallowed into base);
  monitor `on_snapshot(tick, cells, region)` adoption (grow → apply held ≤ tick → adopt base
  beyond previous coverage → compare), `notice_region`, adjudication-time escapes;
  `Evidence3DStream.notice_region`/`extend_world` (+ `build_evidence3d(growths=…)` so the
  OFFLINE feature pass adopts growth exactly like live — the put-the-base-block-back parity
  case is a test).
- [x] Runners: LivePipeline fans snapshots/regions to both worlds; run_live's 1 Hz manifest
  check now splits pre-freeze anchor-move (reseed, unchanged) from post-freeze growth
  (`notice_region`); seeding + catch-up use each snapshot file's OWN frame and walk growth
  interleaved. run_d2 walks per-file regions and hands growths to the feature pass. FlowViz
  refreshes the manifest region BEFORE counting each poll's events.
- [x] Mod 0.0.6 built (22 s) + jar 1.4.0 deployed. **The game must be relaunched to load it** —
  which also gives the next session a fresh region anyway.
- [x] Tests: MICA 166 → **173** (grow-only refusal, growth adoption + absolution, true escape
  past the cap, margin-race base guard, feature-world growth, offline/live parity, late-attach
  catch-up across growth), FlowViz 28 → **29** (health tracks a grown frame — caught a real
  ordering bug: the refresh ran after escape counting). Golden-equivalence replay on the real
  treehouse session still exact, record for record.
- [ ] In-game verification (next session, ON PURPOSE): place one block, walk 30+ blocks, build
  a template — expect "[MICA] region grew" log lines, zero escape warnings, clean offline PASS.
  The walk-to-your-site rule is hereby demoted to best practice.

---

# ✅ SESSION 002717 — FIRST FULLY-CLEAN LIVE RUN + THE h3d_scan CONVERGENCE ARTIFACT (2026-07-05)

The whole intended stack ran from tick zero for the first time: pre-warmed attach at tick 9,
pixels + h3d live from the first block, agent + camera + scanner shadowing, tick-aware monitor.
**run_live exited 0 with NO live quarantine** — the first session where the live path stayed clean
end to end (only 25 socket-dropped ticks keep it short of live proof-grade; the disk copy is
complete). Offline: replay a=125 b=0 **c=0**, 0 escapes; D2 PASS (724 records, h3d on 679);
D1 PASS (156/156 events).

- [x] **THE h3d_scan EARN-ITS-PLACE ARTIFACT — first real measurement, and it passes the shape
  test**: the agent (actively following this time) covered 105/126 built cells (83%) over 189
  sweeps, and cosine(h3d_scan, exact h3d) rose MONOTONICALLY with coverage: 0.14 @ 3% → 0.25 @
  21% → 0.48 @ 63% → **0.73 @ 83%**. The scanned-portion embedding converges toward the exact
  embedding as the agent sees more — the channel carries the same shape signal, gated only by
  visibility. Banked in `fabric-20260705-002717.agent_scan_report.json` + scan.png (report-graph
  x-axis fix landed same night: convergence samples now plot at their sweep's tick).
- [x] **Label: habitation (builder) — category agreed, style contested** (matcher: treehouse at
  fit 1.00 comp 0.87; builder: habitation, not a treehouse). Category-level pairs eligible (both
  sides attest habitation; the 225251 coarse-correct/fine-uncertain precedent). Counts toward the
  habitation slot of the five template captures with the style caveat; a quick cabin build would
  make it airtight. Template tally: production ✓, habitation (✓ with caveat) — infrastructure,
  defense, decorative remain.
- [ ] **Watch item — sparse frames**: only 522/3184 moments carried pov frames (→ 79/781 records
  pixel-enriched offline). Cause unknown: GUI-open time vs frame-capture throttling under the
  3-model GPU load. Check the next session's frame rate; if it recurs with the GUI closed, profile
  `capturePovFrame` under load.

---

# ✅ SESSION 232045 CLOSED — FIRST MATCHER-AGREES CAPTURE + FIRST SCAN ARTIFACT (2026-07-05)

- [x] **Verified clean under taxonomy v3**: 37 snapshots, a=3039 b=618 **c=0**, 0 escapes; D2 PASS
  (2359 records, h3d on 2184); D1 PASS (463/463 events consumed once, 2686/4421 pixel-enriched).
  The end-of-session disk gate ALSO passed live for the first time (the 15 s finalize wait).
- [x] **Label: production / crop_field (builder) — the matcher AGREED (comp 1.00, fit 0.44), and
  s_goal ranked production first on real play for the first time.** First real capture with
  builder+matcher agreement ⇒ Source B pairs eligible; counts as the PRODUCTION entry of the five
  Phase-E template captures. Four remain: habitation, infrastructure, defense, decorative.
- [x] **First real agent-scan artifact**: 65/221 built cells covered from ONE standing viewpoint
  (the user was AFK post-attach, so the follow loop had nothing to track and the agent never
  moved — correct occlusion behavior, a one-sweep scan). cosine(h3d_scan, exact h3d) = 0.55 at
  29% coverage. The convergence curve needs a session where the agent orbits — next capture.
- [x] **Report-tool hardening**: `_read_jsonl` skips unparseable lines instead of stopping at the
  first (an aborted earlier attach left junk at the head of the belief log and silently cost the
  belief graph). All 6 graphs render for 232045; suites 166 + 28 green.

---

# ✅ TAXONOMY v3 (SUPPORT-POPPED PLANTS) — LIVE DIAGNOSIS DURING SESSION 232045 (2026-07-05)

The pre-warm rig's first real outing attached MID-session (the user had been playing without the
rig since 23:20) — **late-attach catch-up recovered 463 human events from disk**, then honestly
flagged a quarantine. Live diagnosis from the disk copy: every class-c cell was
`replay grass vs world air` (1 → 7 over the session) — the short grass PLANT popping off when the
block under it is dug, a consequence with no break event of its own (the door-top family). No
wallclock jumps (F9 removal held), 0 escapes, mod 0.0.5 confirmed in the manifest.
- [x] **Taxonomy v3**: support-popped/regrowing short plants join class (a) — grass, ferns, dead
  bush, the 13 small flowers, the 6 saplings, lily pad, seagrass. Logs stay class (c) (unchanged
  reasoning). Verified offline against the live session's snapshots; D2 doc amended (dated);
  tests 166 green. The session's post-close rebuild is expected clean under v3.

---

# ✅ D1-LATE FIX (PRE-WARM) + SESSION REPORTS (2026-07-04/05, late night, later)

User observation confirmed: the D1 model channels showed up late on FlowViz. Root cause from the
223052 rig log: run_live only STARTED at Open-to-LAN, loaded Uni3D synchronously (~10 s) before
attaching, then VPT/MineCLIP in the background — `px:loading` for ~12 more status lines; the
session's first ~30-40 s of scored records passed unenriched.

- [x] **Pre-warm (`run_live --wait-session`)**: both GPU models load at RIG start (launcher/menu
  time); discovery waits without a deadline; the attach additionally holds until the manifest
  carries a region (a session file exists from game launch — attaching pre-world would lock
  D1-only). `lan_autostart` restructured into two lifecycles: run_live persistent (spawned at
  watcher start, restarted 10 s after each exit), agent per LAN announce with a 10 s re-arm.
  The old announce-triggered pair state machine is gone. D6 amended (dated).
- [x] **Session banking + proof graphs**: `scripts/make_session_report.py <session.jsonl>` —
  the per-session artifact set is now DEFINED in one place (raw + derived evidence + agent scan
  + report), and the report renders the recorded logs into graphs under `capture/reports/<sid>/`:
  belief marginals + entropy, per-goal comp + build growth, h2d/h3d norms + the five s_goal
  cosines, action timeline, scan coverage/convergence, exact-vs-scanned clouds, summary.json.
  Verified on the real 223052 session (5 graphs; belief visibly reacting ~tick 2700; defense
  comp locking at 0.89 as the build grows — the matcher's perimeter-wall read, as a curve).
- [x] Ordering question answered again: these are Phase-E support tooling, not a detour — the
  graphs are the probe/proof artifacts and the pre-warm makes template captures pixel-complete
  from the first block. Phase E's gate is still the user's five template captures.

---

# ✅ AGENT-SCAN 3D CHANNEL (h3d_scan) — SENSOR + MEASUREMENT BUILT (2026-07-04, late night)

Per user decision: the agent gets vision-derived 3D perception — an agent-eye raycast scan of
the build, Uni3D-embedded, entering evidence ONLY through the Phase-E adapter as a measured arm.
**Ordering pinned: Phase E first** (data-gated on the five template captures; the trained adapter
+ calibration harness are the floor every perception arm must beat). Vault note:
`Agent-Scan 3D Channel (h3d_scan)` + a dated cross-ref under the Research Plan's Phase E.

- [x] **Sensor (agent.js)**: 4 Hz sweep, 32×18 ray grid over the 70° first-person frustum from
  the live eye pose (the in-game-verified gaze convention), `bot.world.raycast`, range 32 —
  real occlusion. New cells append to `agent-<NAME>.scan.jsonl` (rotated aside on relaunch,
  never deleted); `scan_cells` in the status feed; `MICA_NO_SCAN=1` opt-out. Status cadence
  1 Hz → 4 Hz (the scan view renders from this pose).
- [x] **Mind side**: `mica/perception/agent_scan.py` (parse/accumulate; channel cloud =
  scanned ∩ BUILT — A7-clean, the note was refined same-sitting from "∩ non-base" for exactly
  this; coverage vs D2's exact built set) + `scripts/run_agent_scan.py` (per-session report:
  coverage curve; with --h3d the convergence curve cosine(exact h3d, h3d_scan) at ~deciles —
  the earn-its-place artifact).
- [x] **FlowViz "Agent scan" panel** under the agent camera: the scanned cloud rendered from
  the agent's live pose (70° pinhole), badge `N cells seen · covers K/M built`. Verified in a
  browser against a synthetic scene (wall + floor + agent pose): badge exact
  ("103 cells seen · covers 15/15 built") and the render geometry checked by pixel analysis —
  wall paints centered at eye height, floor fills the lower half.
- [x] **Tests: MICA 161 → 165** (torn-tail parse, first-seen accumulation, A7 channel selection,
  report end-to-end) **FlowViz 27 → 28** (scan fold, region filter, pose ride-along). All green.
- [ ] **Real-session smoke pending**: next capture doubles as the scan smoke — camera iframe vs
  scan panel silhouettes must match, coverage climbing as the agent orbits; then
  `run_agent_scan.py --h3d` for the first convergence artifact.

---

# ✅ SESSION 223052 — FIRST CLEAN FULL-RIG CAPTURE + TWO LIVE-PATH FIXES (2026-07-04, late night, after 0.0.5)

First session on mod 0.0.5 (F9 gone) with the full rig: camera live, h3d + pixels on from the start.
**The disk artifact is clean end to end**: replay a=0 b=10 (door tops — the documented gap) **c=0**,
0 escapes; D2 PASS (566 records, h3d on 409; matcher's finished-build read: **defense / perimeter
wall, fit 0.58, comp 0.89**); D1 PASS (566 scored, 68/68 events consumed once, 646/684 pixel-enriched).
Agent camera verified in a real session: prismarine-viewer bound :3007 and the status file advertised it.

- [x] **LIVE false quarantine root-caused + fixed (compare-time race).** The live monitor applied
  events the moment packets arrived, but a snapshot file reaches the comparison ~1 s after the world
  state it froze — a busy builder's newest blocks read as phantom class-c and the session banner
  screamed QUARANTINED all night while the recording was fine (offline replay of the same session:
  0 divergences at every snapshot). Fix: `SnapshotMonitor.apply_packet` now HOLDS in-region events
  with their tick; `on_snapshot` applies exactly those ≤ the snapshot's tick before comparing;
  `LivePipeline.finish` drains the hold. Escapes never wait (the alarm stays instant). Regression
  test added (a tick-120 event must not poison a @100 compare). Tests 160 → **161**.
- [x] **Disk-gate finalize race fixed.** run_live waited 5 s for the manifest to finalize, but the
  mod drains its frame-writer queue (up to 10 s) BEFORE writing the final manifest — the gate read
  a provisional manifest that finalized moments later and printed a spurious FAIL. `_FINALIZE_TIMEOUT_S`
  5 → 15 s. (The offline reruns saw the finalized manifest and passed.)
- [ ] **Socket drops under full load (open, monitor next session):** 87 of ~4700 ticks dropped in
  16-tick stretches with pixels + h3d + the game sharing the GPU (live-path only; the disk copy is
  complete by construction, and this session's dropped moments carried no events — ids 0..67 all
  seen). The live run is honestly marked not proof-grade when it happens. If it recurs, consider
  s_goal stride > 1 or frame thinning on the live wire.
- [x] **Session label: habitation/house (builder) — matcher miss #2.** The matcher read
  defense/perimeter wall (comp 0.89, fit 0.58) on a free-built HOUSE — after miss #1 read
  decorative/fountain on the 193059 house. A house's outer wall ring genuinely registers as a
  perimeter wall; free-built houses have now confidently defeated the matcher 2/2. Contest rule
  applied: no pairs. Standing conclusion sharpened: **Source B pairs come from TEMPLATE captures**;
  free builds serve behavior/pixel evidence and belief evaluation. (The five template captures —
  one per category, built to the templates.py shapes — remain the Phase-E data gate.)

---

# ⚠️ SESSION 210003 CLOSE-OUT + LIVE-VIEW FIXES (2026-07-04, late night)

## The F9 silent-pause discovery — a capture-integrity flaw, mod fix PENDING USER DECISION
- [x] **Root cause of 210003's quarantine found — and it was NOT (only) the cactus.** The offline
  rebuild under taxonomy v2 still quarantined: **385 class-c divergences, 130 distinct unexplained
  cells at session end** (45 glass, 46 dark_oak_slab, 30 oak_log, 4 missed breaks) — half the house
  has no events. Cause: **two F9 capture pauses** (43 s at tick 440, 290 s at tick 13095 — recovered
  from wallclock jumps between consecutive ticks; 43+290=333 s = exactly the wallclock-vs-tick-span
  discrepancy). `recordEvent` drops block changes while paused BY DESIGN, and the tick counter holds,
  so the recording looks gap-free while premise (i) is silently violated. The "PAUSED (F9)" banner
  only renders with the F8 HUD open (off by default) — F9 sits next to F8: an accidental, invisible
  press. D2 doc carries the dated danger callout; **interim rule: never press F9 mid-session.**
- [x] **Mod 0.0.5 — F9 REMOVED ENTIRELY (user decision, same night)**: no pause feature at all —
  recording runs whenever a world is open; to stop capturing, quit the world (a clean session end).
  Keybind, lang entry, `recording` flag, and the justPaused flush all deleted; HUD always shows
  RECORDING; jar 1.3.0. (This also retires deferred item D-8, the pause/resume straddle residual —
  there is no pause to straddle.) A live wallclock-jump detector was considered and dropped with the
  feature: with no pause, a client freeze stalls the integrated server too, so a jump can no longer
  hide unrecorded world changes.
- [x] **Session 210003 verdict**: structure evidence QUARANTINED, correctly and irrecoverably (the
  events don't exist). **D1 behavior evidence is green**: offline `run_d1 --pixels` PASS — 833
  scored + 580 context, 138/138 events consumed once, 870/1413 records carrying h2d+s_goal,
  fidelity + contract PASS. s_goal flat again (spread ~0.011 — consistent with the standing
  flatness finding). Source B matcher read skipped: the replayed world is missing 130 cells, so a
  finished-build read would be reading a half-house; no training pairs from this session either way.

## Fixes landed and verified tonight
- [x] **Taxonomy v2** (the cactus): growth-stage plants, spreading ground covers, leaf decay, and
  snow/ice are class (a); **logs deliberately stay class (c)** (indistinguishable from a missed
  placement). Verified on the incident session: its 5 grown-cactus cells read (a). D2 doc amended
  (dated old-vs-new). `TAXONOMY_VERSION="2"`.
- [x] **Agent camera root cause**: `prismarine-viewer` 1.33's `viewer/lib/entities.js` requires
  `canvas` top-level without declaring it — `require()` threw MODULE_NOT_FOUND all session, caught
  silently, `viewer_port` null throughout. Fixed: `npm install canvas` (now in package.json);
  agent.js publishes `viewer_error` in the status file, pre-probes ports 3007→3009 (prismarine-viewer
  has no listen error handler — a held port would crash the whole agent; a zombie once held it),
  publishes the ACTUALLY bound port; FlowViz cam badge shows `failed: <reason>` instead of "off"
  (verified in-browser end-to-end with a fake status line). In-game iframe check: next session.
- [x] **Late-attach catch-up** (the 22-of-100-cells point cloud): run_live now defines pre-attach
  history as "recorded before the first socket moment" and replays it from the disk jsonl —
  `Evidence3DStream.seed_history` (human-only, A7), `SnapshotMonitor.seed_history` (every event +
  pre-attach snapshot comparisons), dedup on every path (`events_consumed` stays 0; reseed guard
  intact). Attach-at-start is a no-op — golden equivalence preserved (the e2e equivalence test
  caught the first draft's "catch up everything on disk" overreach; the cutoff definition is the
  fix). `live_status.json` now carries uncapped `built_count` beside the 512-capped cell list.
  D6 amended (dated). **Rehearsal-verified on 210003**: "late attach: 110 human / 0 agent events
  caught up", built_count 22 → **100**, dashboard "100 cells · h3d ✓ in sync (100)".
- [x] **Rehearsal feeder**: `--grow-jsonl` (staged session file starts empty, fills as it plays —
  the mod's parallel disk write, so catch-up reads a true prefix) + `--late-ok` (feed plays on
  schedule; consumer may attach anytime — the real socket's semantics).
- [x] **FlowViz honesty**: point-cloud badge now names its empty states ("no live pipeline yet" /
  "live_status is for <sid> — not this session" via a one-shot `live_models_mismatch` hint / "0
  built cells" / "N cells (showing 512)").
- [x] **Region decision (user, 2026-07-04)**: capture region stays **as-is** (49-wide, provisional
  until first block event, then frozen). Dynamic/footprint-following region considered and declined —
  conflicts with the pinned fixed-frame proof design; the walk-to-your-site-first rule stands.
- [x] **Tests: MICA 152 → 160** (late-attach suite: seed splits actors, dedup everywhere, monitor
  snapshot interleave + corrupted-snapshot quarantine, `_seed_structure` both cases, socket e2e
  tail-attach, built_count cap; feeder grow-mode staging), **FlowViz 25 → 27** (mismatch hint
  once, built_count forwarded). All green.

---

# ✅ h3d LIVE — BOTH MODEL CHANNELS IN THE LIVE FUSED EVIDENCE (2026-07-04, night, later)

Per user decision: h3d must run live so it reaches the adapter alongside h2d and, through it,
the future action layer. Vault amended first (D3 h3d note + D6).

- [x] **`run_live --h3d` / `MICA_H3D=1`**: frozen Uni3D into `Evidence3DStream`'s existing
  `h3d_fn` seam (same provenance pins as run_d2 — encoder sha, cloud 4096, seed 0); loud
  degrade when assets missing; `--session <jsonl>` attaches without discovery;
  `--wait-models` holds the attach until the pixel head is warm (REHEARSALS ONLY — a real
  attach must never wait, it drops queued moments). lan_autostart passes MICA_H3D (default
  OFF — VRAM headroom while gaming).
- [x] **Live numbers, three surfaces**: the 1 Hz status line (`h2d|v|=25.8
  s_goal[production]=0.290 h3d|v|=1.0 (68 rec)`), `live_status.json` (norms + raw s_goal +
  h3d count), and FlowViz (s_goal chips + h2d|v| on the D1 read card, `h3d ✓ |v|` on the D2
  header — `_trim` now ships norms and the five raw cosines instead of presence-only).
- [x] **`scripts/replay_live_feed.py`** — full live-path rehearsal from a recorded session,
  no game needed: stages a clone under `capture/rehearsal/` (NEVER capture/raw; banked
  artifacts checksum-verified untouched), serves the mod's real wire protocol at 20 Hz×speed
  with PNG→RGBA frames, and **drips later snapshots in at their recorded ticks** (pre-staging
  them all made run_live seed from the end-of-session snapshot — the finished build read as
  terrain; found and fixed during verification).
- [x] **VERIFIED on session 194242 replayed over the socket**: 147/148 records
  pixel-enriched (0 past deadline with --wait-models), **h3d on 68/148 — exactly the banked
  offline artifact's coverage**, and **67 fused records carrying h2d(1024) + h3d(1024) +
  s_goal(5) simultaneously — the Phase-E adapter's input shape, produced live**. Numbers
  visibly changing (h2d 27.4→25.6→24.8→26.0→24.9→25.8; s_goal top rotating
  production→decorative→habitation; h3d count 41→68; belief following). Snapshot @5024
  dripped mid-feed and the monitor compared it. Gate PASS, proof-grade YES. Tests 151 + 23
  green; replay-equivalence still exit 0 (symbolic path untouched).

## ⛔ PHASE E READINESS VERDICT (2026-07-04): NO-GO — data-gated, not code-gated

Everything code-side that Phase E consumes now exists and runs live (fused records with both
model channels; Sources A/B/C machinery; calibration harness; architecture pins). What still
BLOCKS Phase E, all converging on one bottleneck:
1. **The real instrumented captures** (five template builds + free builds) — D1's headline
   early-separation probe (B1-F3), Source B's real pairs, the 12-F3 fused-arm rerun, and
   both h3d probe reruns all need them.
2. ~~The mod jar rebuild (region fix) — blocked on a firewall allowance~~ **CLEARED
   2026-07-04 evening — and the "firewall" diagnosis was WRONG (record corrected below).**
3. In-game verifications parked on the next smoke session: ~~region anchoring~~ (verified —
   see below), A7 with the agent placing, D2 recompute latency, and classifying session
   110234's 5,542 divergences (it quarantined with crop_escapes 0 — needs the class
   breakdown from a run_d2 pass or the next live session).
**Do not open Phase E until 1 + 3 clear.** When they do: label captures (Source B one
command), rerun the probes, then train A_φ + heads against the calibration harness.

---

# ✅ AGENT TRACKING AT 4 Hz + THE FOUR LIVE PERCEPTION VIEWS (2026-07-04, night)

Per user feedback after the capture session (tracking latency too high; wants to SEE the
perception stack working live):

- [x] **Label truth for 193059**: the builder made a HOUSE — labels.json records habitation
  with the matcher-miss note. **The matcher's kept verdict (decorative/fountain, margin
  0.33) was WRONG on a real free build**; the local VLM cross-check said habitation (with
  the builder). Contest rule now hard: matcher-contests-builder ⇒ **no pairs written** (the
  535 wrongly-labeled pairs were deleted); vault D3 pinned. Free builds need the
  builder+VLM triangle; template captures stay the trustworthy Source B input.
- [x] **Tracking latency**: follow/gaze loops 1–1.5 s → **250 ms (4 Hz)**, and gaze no longer
  routes through the 1 Hz status file — it mirrors the human's live entity yaw/pitch
  (projected ~4 blocks), the real-time crosshair proxy. Pathfinder churn-guarded (750 ms
  back-off throttle); the status file remains belief/advisory only.
- [x] **The four live views (FlowViz "Perception stack" panel + plumbing)**:
  (1) **agent camera** — prismarine-viewer first-person render served by agent.js (:3007,
  advertised in the agent status file; MICA_NO_VIEWER=1 skips), embedded as an iframe;
  (2) **point cloud** — the player-built cells (the exact Uni3D cloud source) streamed via
  live_status.json (≤512 cells + region) and drawn isometrically, height-colored, ~1 Hz;
  (3) **MineCLIP live** — the display readout's five cosines as a multi-line sparkline +
  current leader; (4) **VPT live** — the display h2d as a 32-bucket profile strip + norm
  sparkline (pixel head's display job now computes BOTH models each second).
  New `live_models` feed: FlowViz polls live_status.json, publishes latest-value messages
  (session-id-guarded). Tests: FlowViz 23 → **25**, MICA 152 (display-test updated for the
  both-models readout).
- [x] **Rig launch deadlock, root-caused live (2026-07-04, session 210003)**: after the
  205205 restart, the rig silently failed to attach to the new LAN — a **zombie agent**
  from the previous session never exited ('end' never fired), which held the watcher in
  'running'; and even after the zombie died, the old re-arm rule (wait for announcement
  QUIET) would deadlock whenever the player is already in the next world, still announcing.
  Two fixes, both effective next launch: (a) **agent watchdog** — the server's 1 Hz time
  packets are the heartbeat; 30 s of silence in the 'present' state ⇒ hard exit (the
  watcher needs both children dead); (b) **fixed 10 s re-arm cooldown** replaces the
  quiet-wait (attaching to a still-open world is correct, not a race). Manual recovery this
  time: killed zombie + watcher, relaunched; pipeline attached mid-session to 210003
  (region locked ON the build — first block at the site, 110/110 events in-region,
  0 escapes, no quarantine).
- [x] **Session-start footgun, observed live (2026-07-04, session 205205)**: ANY early block
  interaction freezes the capture region — the user touched a block near their start point
  (tick 340, pre-LAN-open), the region locked there, and the real build 55 blocks away fell
  outside it (67/75 events escaped; D2/h3d correctly blind; quarantine + escape alarms fired
  loudly and LIVE — the alarm system built after the original silent-blindness incident paid
  for itself). NOT a pipeline bug; the fixed-frame rule is deliberate (a moving region would
  silently change the evidence frame). **Session-start procedure pinned: walk to the build
  site FIRST; the first placed block locks the 49-wide box.** Session 205205's structure
  evidence is void (behavior/pixels valid); recovered by restarting the world session.
- [x] **Layout by modality + rename (2026-07-04, user request)**: the dashboard is now
  **MICA_Flow_View** with the left column = VISUAL (player view, agent camera, point cloud)
  and the right column = READOUTS (B0 live, agent, D1 read, model channels, D2 read, belief,
  GPU); canvases hidden until data arrives, so the idle page stays compact. **Point-cloud ↔
  h3d sync CONFIRMED**: both come from the same D2 cache (`world.built()`); `h3d_cells` (the
  last embedding's input size) now rides live_status and the panel says "h3d ✓ in sync (N)"
  when cloud == embedding. Empirical cross-check on session 193059: last h3d built_count 19
  == independently replayed final built cells 19, zero pending events — **EXACT**. Zero
  console errors; suites 152 + 25.

---

# ✅ FIRST FULL-RIG CAPTURE SESSION (2026-07-04, 19:30) — A7 verified in-game; first proof-grade real capture with BOTH model channels

Session `fabric-20260704-193059` (mod 0.0.4, full rig: HumanBuilder + MICA_AI + A7 test +
FlowViz + run_live with pixels AND h3d live):

- [x] **Proof-grade capture**: gate PASS, manifest finalized (99 events), 0 gaps / dups /
  unconsumed / crop escapes. Offline replay: **15 snapshots, divergences a=0 b=0 c=0 — the
  cleanest real-session replay on record.**
- [x] **A7 VERIFIED IN-GAME (closes the D5 §9(b) precondition)**: the agent placed blocks in
  a real session — `MICA_AI 2` events recorded under its name, run_d1 reports "2 agent
  (excluded by A7)" with all 97 human events consumed exactly once, and the replay proof
  (which keeps agent blocks) still matched reality perfectly.
- [x] **Both model channels ran live on real play** (h2d/s_goal enriching in-session, h3d
  first embedding at tick 2168, 295 h3d records live); offline rebuilds banked the canonical
  artifacts: `run_d2 --h3d` (535 records, 295 with h3d, final read decorative/fountain
  fit 0.57 comp 1.00) + `run_d1 --pixels` (594/625 with pixels; fidelity + contract PASS).
  Mean s_goal spread on real play: 0.012 — the flatness risk is REAL on free play; the
  template-capture probe (B1-F3) remains the decisive measurement.
- [x] **Live false-quarantine root-caused + FIXED**: mod 0.0.4's PROVISIONAL region can move
  (rewriting/pruning the base snapshot) after run_live seeds — the stale-seeded monitor then
  mass-diverges against every new snapshot (126,595 divergences here; 5,542 in 110234).
  run_live now re-reads the manifest region ~1 Hz and RE-SEEDS the structure stream while
  that is still exact (zero events consumed — guaranteed by the freeze rule);
  `LivePipeline.reseed_structure` asserts the guard. Test reproduces the bug shape
  (provisional seed → reseed at freeze → 0 escapes, no quarantine). Tests 151 → **152**.
- [x] **Source B labeler verdict on the session**: decorative/fountain, score 0.5652,
  margin 0.33 over habitation — KEPT. Awaiting the builder's own label for labels.json.
- [ ] Remaining from the smoke-session checklist: D2 recompute latency held (511 live
  corrections, 0 drops) — formally fine; **four more category captures wanted** (this
  session likely covers decorative) + free builds, then the real-capture reruns
  (B1-F3 probe, 12-F3, h3d probes) and Phase E opens.

---

# ✅ MOD BUILD UNBLOCKED + REGION FIX VERIFIED IN-GAME (2026-07-04, evening) — the "firewall block" was a misdiagnosis

- [x] **The real cause of "Unable to establish loopback connection", isolated by experiment**:
  plain sockets from the toolchain JVM work fine; `java.nio.channels.Selector.open()` (NIO's
  internal loopback socket-pair self-connect) fails — but ONLY inside the assistant tool
  environment's process tree (sandbox on or off). The same probe run via Task Scheduler,
  outside that tree: `SELECTOR OK`. **The machine was never blocked; no firewall rule was
  ever needed or added.** Gradle builds from the assistant must run out-of-tree
  (`schtasks /create ... /run` pattern) or from the user's own terminal.
- [x] **Second, previously masked failure — loom SNAPSHOT drift**: `loom_version=1.17-SNAPSHOT`
  silently moved to builds requiring a **Java 21 build JVM** (all concrete 1.17.x on the maven
  now declare jvm 21; the 07-02 jar caught the last 17-compatible snapshot). Fixed by pinning
  `loom_version=1.17.13` and adding **Temurin JDK 21.0.11+10 to `capture/.toolchain/`** as the
  BUILD JVM (`build-mod.bat` updated; the mod's MC 1.16.5 bytecode target and the game's
  runtime JVM are untouched). Also `build-mod.bat` now calls the wrapper by explicit path
  (hardened-PATH shells refuse bare .bat names from CWD).
- [x] **`mica-b0-capture-1.2.0.jar` BUILDS (17 s)** — and the mods folder already carried a
  1.2.0 deployed at 10:42 (built out-of-band this morning): **session `fabric-20260704-110234`
  ran mod `fabric-b0-0.0.4` in-game with a BUILD-anchored region (`[-34,59,-8 → 14,107,40]`,
  events inside, crop_escapes 0)** — the region-anchoring fix is verified live; D2 read real
  structure during play. Open thread: that session still quarantined via 5,542 divergences
  (class breakdown pending — next session's item).

---

# ✅ D3 DATA STAGE CLOSED (2026-07-04, night) — Source B implemented + proven; calibration harness; Source C mined

Per user decision ("finish D3's incomplete parts; machinery now, train when data lands" — the
Phase-E boundary kept). Vault amended FIRST (D3 note: λ fork resolved to **grid search**;
Source C section; Source B implemented-note; A_φ/head/recipe pins).

- [x] **Source B — the completed-build hindsight labeler** (`mica/data/source_b.py` +
  `scripts/label_finished_builds.py`, one command): replays all HUMAN events (A7-filtered)
  to completion, labels via D2's own matcher (`read_finished_build` — the fit×comp pose
  search, new public seam in evidence3d.py), keep/discard = score ≥ 0.20 + margin ≥ 0.05
  (pinned, printed). Pair builder reuses the runtime streams (snapshot rule + A7 by
  construction), verifies every join via `fuse_dicts`, writes `<session>.source_b.jsonl`
  with the label + build-progress fraction. **Measured on the scripted corpus (truth known):
  30/30 kept, kept-label accuracy 1.0, subtype accuracy 1.0 — 3,098 replay pairs banked.**
- [x] **The treehouse DISCARD is the system working**: the matcher reads the real free build
  as decorative/fountain 0.4156 in a **near tie with habitation (margin 0.002 < 0.05)** →
  discarded rather than labeled. The builder's own label says habitation; the anti-poisoning
  rule refused to guess on a genuinely off-template build — exactly its job. (VLM cross-check
  only runs on KEPT real builds, so it did not fire; harness is in place, Anthropic API when
  a key is present, manual sample sheet otherwise.)
- [x] **Calibration harness** (`scripts/calibration_report.py`): reliability bins + ECE,
  pooled/finals/per-goal, recomputed through the verified join; single-builder caveat printed.
  **First baseline numbers (v0 heads, scripted corpus): pooled ECE 0.23 — overconfident
  mid-build (the 0.7–0.9 confidence bins are 0.35/0.00 accurate) and underconfident at build
  end (finals ECE 0.50 at 0.93 accuracy).** This quantifies why D5 §9 refuses to freeze
  thresholds on v0 heads; the Phase-E pass criterion reuses this exact artifact.
- [x] **Source C — MineDojo YouTube miner** (`scripts/mine_minedojo_youtube.py`): index from
  Zenodo (CC BY 4.0; `youtube_tutorial.json` 34,472 videos cached at `capture/youtube/`,
  `--full` adds the 174 MB gameplay index) → transparent title scoring (2,218 candidates) →
  optional transcript refinement (youtube-transcript-api; **coverage is partial — 2022 index,
  many captions disabled — missing transcripts never discard, status recorded per clip**) →
  optional MineCLIP visual stage (`--visual`, needs ffmpeg — not installed yet). Output
  `capture/youtube/house_subset.json` + report + sample sheet; top-20 spot-checked, uniformly
  real house tutorials. **Every artifact carries the scope line: pixels-only, never fused
  evidence, never Source B pairs, never head training** (vault Source C section pins why —
  no block events; MineCLIP shares selection and measurement, so readouts are diagnostic).
- [x] **Tests 143 → 149** (`tests/test_source_b.py`: correct label with margin on a finished
  template build; too-little-built/near-tie/escaped/nothing-built all discard with reasons;
  agent blocks never reach the labeled world; pair progress strictly pre-action).
- [ ] **Still open in D3 (Phase E, data-gated — unchanged by design)**: train A_φ + heads
  (needs real pixel captures), real calibration numbers, the 12-F3 fused-arm rerun, both h3d
  probe reruns, λ grid sweep. All stage one-command off the user's captures.
- [x] **Source C human-review gate (2026-07-04, late — user decision: no mined label is used
  unreviewed)**: `scripts/make_review_sheet.py` extracts 1–5 key frames per kept clip AT the
  transcript's build moments (evenly spaced without captions; low-res video cached once in
  `capture/youtube/video/`, frames via imageio-ffmpeg's bundled binary — no system ffmpeg
  needed) and assembles proposed_label / evidence windows / rule+score per clip;
  `scripts/review_house_labels.py` (localhost :8330) shows the cards and takes the verdict —
  HOUSE / INTERIOR / ROOF_BUILD / WALL_BUILD keep, NOT_HOUSE rejects. Decisions persist to
  `review_decisions.json`; **`house_subset_approved.json` (human labels only) is the sole
  Source C artifact downstream may read** — pinned in the vault Source C section.
  *Label scheme revised same day (user caught the first cut mixing levels in review): a
  whole-clip VERDICT (HOUSE / NOT_HOUSE / UNSURE — the gate) is now separate from optional
  per-frame STAGE TAGS (EXTERIOR / WALL_BUILD / ROOF_BUILD / INTERIOR / OTHER — annotation,
  rides along as `frame_stages`), mirroring D2 §3's "a wall or roof is a stage, never a goal
  hypothesis". Pre-split decisions migrated (stage picks → HOUSE + a refine-me note).*
- [x] **Auto-label tier (2026-07-04, latest — after the user's 5/5 manual pass)**:
  `scripts/auto_label_clips.py` — auto-approve needs TWO signals agreeing (title score ≥ 6
  AND a VLM shown the frames saying HOUSE); auto-reject needs VLM NOT_HOUSE against a weak
  title; disagreements/UNSURE queue for the human with the VLM's opinion as a hint; a random
  10% audit slice of every would-be-auto batch stays human so auto precision is MEASURED
  (`auto_label_report.json`; <90% agreement ⇒ tool says tighten). Overrides graded too.
  `decided_by: human|auto` provenance flows into `house_subset_approved.json` (downstream can
  filter to human-only). Review sheet extended to 16 live clips (14 of top 30 videos dead —
  ~50% dataset rot holds). Console prints made UTF-8-safe (a fullwidth ｜ in a title crashed
  the first batch).
- [x] **VLM tier UNBLOCKED with a LOCAL model (2026-07-04, latest — user decision: open-source,
  on the 5090, no API key)**: Ollama 0.31.1 installed (winget) + **qwen2.5vl:7b** pulled;
  `mica/data/vlm.py` is the shared local-first backend (temperature 0, JSON-constrained;
  Anthropic API only as fallback; provenance recorded per verdict) — both `auto_label_clips.py`
  and Source B's cross-check in `label_finished_builds.py` now route through it.
  **Earn-its-place validation vs the user's 16 completed manual labels: 94% verdict agreement,
  PASS (≥90% bar)** — with the honest caveat that the single NOT_HOUSE (titled "2x2 Starter
  House Tutorial", score 15) was missed: negatives are near-unmeasured (n=1) and the two-signal
  rule shows ~6% measured false-approve on this batch; the 10% audit slice keeps grading it.
  Stage-tag prompt fixed to an explicit JSON key shape (the model omitted stages otherwise).
- [x] **Stage scheme + taxonomy bridge (2026-07-04, after the user's sub-label review)**:
  `INVENTORY_UI` added to the frame stages (user finding: inventory/menu frames got NO tag —
  they matter as an exclusion set for pixel statistics); `--retag` mode fills only-missing tags
  (human tags never touched) with a per-frame one-word fallback for frames the local model
  skips when batch-enumerating (it returned 2 JSON entries for 3 images) — **coverage now
  118/118**, distribution WALL 35 / EXTERIOR 29 / INTERIOR 25 / OTHER 15 / ROOF 14. The
  approved subset now carries the taxonomy bridge explicitly: every HOUSE clip has
  `category: "habitation"`; extending mining to the other four categories (per-category
  vocabularies) is the recorded follow-up; frame stages never map to categories (stages ≠
  goals, D2 §3); nothing from Source C ever enters fused evidence.
- [x] **ALL FIVE CATEGORIES MINED + LABELED (2026-07-04, latest — user request)**: the whole
  Source C pipeline is category-generic — miner vocabularies + verdict labels + VLM definitions
  per category (`mined_<category>.json`; retrieval counts: habitation 2218, production 702,
  defense 161, decorative 51, infrastructure 36 — the tutorial index is house-heavy, thin
  categories may need `--full`/wider vocabularies); review sheet merges category-tagged items
  (decided items never rebuilt); auto tier judges each clip against ITS category's definition
  (verdict `<LABEL>`/`NOT_<LABEL>`/`UNSURE`); audit-reserved clips excluded from later auto
  passes (they grade the machine). **Approved subset now: 77 clips — habitation 35,
  production 11, defense 11, decorative 10, infrastructure 10 (17 human, 62 auto,
  provenance per clip); 6 audit cards queued for the human; auto-vs-human agreement 1/1 so
  far.** The model even auto-REJECTED one infrastructure clip against its own title signal.
- [x] **Stage definitions made CONCRETE (2026-07-04, user decision after the audit cards)**:
  the sub-label scheme is now geometric — where the CAMERA is and which part of the build is
  worked (EXTERIOR = outside looking at it; INTERIOR = inside; WALL_BUILD = the side, up
  close; ROOF_BUILD = on top; INVENTORY_UI = overlay wins, checked first; OTHER = strict last
  resort). One text, two consumers: the model's prompt guide and the review chips' tooltips.
  `--restage` re-tagged the model's own 91 tags under the new definitions while **all 171
  human-set tags stayed untouched** (a tag is provably human when its clip has no VLM record
  or differs from the model's proposal). Distribution shifted plausibly (ROOF 27→38,
  INTERIOR 40→37, INVENTORY_UI 10→14, WALL 88→83).
- [x] **Automation readiness measured + three guardrails (2026-07-04, latest)**: graded audits
  came back **5/8** (misses all decorative/infrastructure) while fresh verdict validation under
  the new scheme hit **97% over all 36 human-decided clips** and stage validation only **54%**
  (72 frames). Verdict → GO with rails; stages → annotation-only. Rails implemented:
  (1) per-category audit fractions (habitation/production 10%, decorative/infrastructure/
  defense 40%; `--audit-fraction` overrides all); (2) the 18 old-prompt decorative/
  infrastructure auto-approvals carry a RECHECK note in the UI until confirmed/overridden
  (any human verdict clears it and grades the tier); (3) `frame_stages` in
  `house_subset_approved.json` carry per-tag provenance `{stage, by: human|auto}` (154 human
  / 92 auto today) so stage-dependent uses filter to human tags.
- [x] **Recheck pass graded (2026-07-04, latest)**: the first recheck review never registered
  (UI defect — recheck cards LOOKED decided, so there was nothing to click; fixed with an
  explicit CONFIRM button). User re-clicked all 18 — and re-verdicted most of the corpus by
  hand along the way (approved subset now 62 human / 22 auto). **Auto-tier agreement:
  43/46 = 93% overall — habitation 1/1 (+97% validation), production 10/10,
  infrastructure 11/12 (92%), defense 11/11, decorative 10/12 (83%)**; the three misses
  are the known decorative/infrastructure trio, unchanged. Audit fractions per the earn-down
  rule: infrastructure and defense → 25%, decorative stays 40% (only category under the 90%
  bar), habitation/production 10%. **Final state (all 85 cards decided, queue EMPTY):
  auto-tier agreement 62/65 = 95% — habitation 20/20, production 10/10, defense 11/11,
  infrastructure 11/12, decorative 10/12; the same three misses, no new ones. Approved
  corpus: 82 clips (81 human-verified / 4 auto) — effectively a fully human-verified
  calibration set for grading all future auto batches (`--validate`).**
- [x] **LARGER BATCH mined + auto-labeled (2026-07-04, latest — 30 clips/category target)**:
  sheet grown to **157 clips** (habitation 37 + 30 each of production/defense/infrastructure/
  decorative); auto-label passes with the finalized per-category audit rates + concrete stage
  scheme. **Approved corpus 82 → 130 clips (81 human / 55 auto), now category-balanced**
  (habitation 36, production 27, defense 25, decorative 21, infrastructure 21). Guardrails
  fired correctly on the more-diverse batch: heavier audit routing in thin categories (17 in
  the human audit queue). Two robustness/correctness fixes landed this run: (a) the sheet
  builder now writes review_items.json **incrementally** (an interruption keeps progress —
  the prior two runs lost everything); (b) `--validate` was only grading HOUSE clips and
  didn't exclude auto-labels — **fixed to grade human-decided clips across all five
  categories**. **Correctness proof — VLM vs 81 human clips: 93% overall (PASS ≥90%),
  per-category habitation 97% / production 100% / defense 91% / decorative 83% /
  infrastructure 83%**; frame-stage agreement 55% (annotation-only, human-filtered in the
  artifact). Visual spot-proof done by reading frames directly (Claude Preview + browser MCPs
  were disconnected this session): strong categories show unmistakable correct labels, the two
  83% categories show genuinely ambiguous frames — consistent with their higher audit rates.
- [ ] Optional Source C follow-ups: deeper transcript pass (`--transcripts 500`); the
  `--visual` MineCLIP stage; widen decorative/infrastructure vocabularies or add `--full` to
  lift them past 90%; clear the 17-clip human audit queue at the review server.

---

# ✅ STAGE-2 PIXEL HEAD LIVE + AGENT ACTIVE SHADOWING (2026-07-04, evening)

Per user decision: the live pipeline now runs VPT + MineCLIP in-game (D6 §8 stage 2,
vault-amended first), and the in-game agent actively traces the player (D5 presence
v1.1 pin). **Confirmed before building: the pixel channels were OFFLINE-ONLY** —
`run_d1 --pixels` worked and s_goal reached the belief (deliberative bump, weight 1.5),
but live sessions carried h2d/s_goal on 0/151 records (session `110234`), exactly as
D6 §8 documented ("live v1 runs the symbolic path only"). Not a bug — an unimplemented
documented seam. Now implemented:

- [x] **`mica/perception/pixel_head.py` — `LivePixelHead`** on the reserved seam
  (`PipelineConfig.pixel_head`, the one-method `enrich` contract unchanged): one GPU
  worker thread loads frozen VPT + MineCLIP (with a CUDA warmup before "loaded" flips);
  `enrich` submits the record's EXACT frame window (same rule as batch
  `_enrich_pixels`; s_goal clip honors `--sgoal-stride`) and waits a bounded 400 ms —
  not loaded / deadline missed / no frames ⇒ record unchanged, both channels None
  together. Records displace queued display jobs (records outrank readouts). The
  worker also computes a ~1 Hz **display-only** s_goal readout (status line,
  `live_status.json`, agent view — never written into records). Frames arrive raw
  RGBA off the wire; the head keeps its own ring (FrameRef width/height decode).
  `write_d1_provenance` moved here, shared by run_d1 + run_live.
- [x] **run_live wiring**: `--pixels` or `MICA_PIXELS=1` (lan_autostart defaults it ON);
  missing torch/checkpoints degrade LOUDLY to the symbolic run; per-moment
  `head.observe()` fed by the runner outside the pipeline seam; d1 provenance sidecar
  written when any record was enriched; status line shows `px:<live top goal>`.
  **GPU smoke-verified**: load+warmup 32 s (background), enrich 45–107 ms/record,
  h2d(1024) + s_goal(5) filled, display readout live.
- [x] **`live_status.json` — the agent bridge**: run_live atomically rewrites a ~1 Hz
  snapshot next to the session (top goal + p, P(z=1), entropy, current_behavior,
  player_pos, focus_block, pixel status). Anything local polls this; the
  single-consumer socket stays run_live's alone.
- [x] **Agent active shadowing (`agent.js`, D5 presence v1.1)**: mineflayer-pathfinder
  (dig/scaffold/tower all disabled — the body moves, never edits the world); follows
  the human in a 4–8 block band, backs off inside 4, yields the workspace (never
  stands within 3 of the live focus_block); gazes at the pipeline's focus_block, else
  the human; one throttled advisory chat line ("I think you're building a … build
  (NN%)") only when the top goal holds ≥5 consecutive reads at p ≥ 0.5, ≥30 s apart,
  `MICA_QUIET=1` disables — the pre-B6 SUGGEST exercise the D5 §9 pin licenses, not
  the gate. Agent status file gains `follow` + `believes` (FlowViz-compatible).
- [x] **Vault first (source-of-truth rule)**: D6 §8 stage-2 amendment (exact-window
  + deadline revision of sketch 3, dated) and the D5 presence-v1.1 pin were written
  before the code.
- [x] **Tests**: MICA 135 → **143** (`tests/test_pixel_head.py`, torch-free stub
  encoders: exact window strictly pre-action, stride widens clip only, empty window ⇒
  both None, not-loaded/load-failure/deadline ⇒ unchanged, display readout advances
  without touching records, serialization); run_live integration asserts
  `live_status.json`. FlowViz 23 still green.
- [ ] **In-game verification at the next smoke session**: follow band + workspace
  yield feel right; advisory line fires once and reads sensibly; `px:` on the status
  line with real POV frames; s_goal separation on a real build (the D1 flatness
  question now measurable live). Note: MineCLIP's text side still touches the HF Hub
  on first load (standing ISSUES item — pin/cache for offline runs).

---

# ✅ D2 STRUCTURE-READ BLINDNESS — DIAGNOSED AT THE ROOT + FIXED (2026-07-04, later)

**The user's question ("why is the D2 structure read not working — D2 bug or viz bug?") is answered:
neither.** It was a capture/session-lifecycle flaw, and the session self-reported it — silently.

- [x] **Diagnosis (data-complete, session `fabric-20260704-022944`)**: the mod froze `snapshot_region`
  at the player's **spawn** (`[405,50,438]→[453,98,486]`, first in-world tick), the player teleported
  to their build site (~`(-12,71,12)`) — **92/92 events landed outside the region** → `ReplayWorld`
  marked each `escaped`, never applied → `built() = ∅` → `fit = comp = built_count = 0` in all 477 B2
  records (mathematically forced: fit > 0 iff any built cell). FlowViz rendered the zeros faithfully.
  The monitor kept comparing the abandoned spawn box → 823,543 divergences → quarantine (correct, but
  invisible while playing). Ingest/gate were clean throughout. **D2's algorithm and the viz were both
  working as designed on blind input.**
- [x] **Root fix (mod 0.0.4 / jar 1.2.0, code done)**: the region is **provisional until the first
  block event** — re-centers on the player when they move > 8 blocks (base snapshot rewritten at quiet
  moments per re-anchor, so it stays strictly pre-event), **freezes at the first recorded block event**,
  and prunes stale provisional snapshots so `snapshots/` holds only final-region frames. Build-where-
  you-spawn behaves exactly as before. HUD (F8) shows `region: follows you @(x,z)` → `region: fixed`.
  D2 doc §2 amended (superseded-callout, 2026-07-04).
- [ ] **Jar build BLOCKED by a machine-level Java loopback denial** — Gradle dies with
  "Unable to establish loopback connection" (persists with `--no-daemon`; `gradlew --version` works,
  python binds loopback fine → firewall/AV is blocking the toolchain JVM specifically). **Fix on the
  user's side**: allow `capture\.toolchain\jdk-17.0.19+10\bin\java.exe` (and `javaw.exe`) through the
  firewall/AV, then `capture\fabric-mod\build-mod.bat build`. In-game verification of the anchoring
  (teleport-then-build must yield escapes = 0 and a live D2 read) belongs to the next smoke session.
- [x] **Failure made loud (so this class can never be silent again)**:
  `LivePipeline.status()`/`summary()` carry `crop_escapes` + `quarantined`; `run_live`'s 1 Hz line
  appends `!! D2 BLIND: N events outside region !!` / `** QUARANTINED **` with a 30 s-throttled loud
  block; `SnapshotMonitor` memory capped (exact `divergence_count` + `class_counts`, first 200
  examples — it held 823k tuples); `run_d2` reports exact counts. FlowViz raises a red
  **capture-health banner** ("⚠ build outside capture region — structure read is blind (N/M events
  escaped)") computed from data it already tails (manifest region vs event positions), plus a
  quarantine flag read from `<session>.live_run.json` — verified live against a real bad session
  (`fabric-20260704-025338`, 49/49 escaped).
- [x] **Agent view shipped (the planned second perspective)**: FlowViz is now two labeled columns —
  **player perspective** (POV, B0, and the D2 structure read: analysis OF the human's build) and
  **agent perspective** (presence card fed by `agent.js`'s new 1 Hz flushed
  `agent-<NAME>.status.jsonl` — state/pos/distance-to-human/watching/last action; a D1 behavior-read
  card — the human's behavior as MICA perceives it; belief; GPU). Agent block events get their own
  token color + "A7-excluded" counter, never evidence-edge tokens. Per the user's framing decision:
  D1 + belief = the agent's perception, D2 + POV = the player's point of view (presentation only;
  pipeline semantics untouched).
- [x] **Live GPU panel**: `nvidia-smi` polled every 2 s (utilization/memory + compute apps), pipeline
  processes named (Minecraft javaw / run_live / FlowViz / agent node), non-pipeline WDDM graphics
  clients collapsed to "+ N other graphics apps"; graceful "n/a" without nvidia-smi.
- [x] **Tests**: MICA 131 → **135** (`tests/test_region_blindness.py`: tonight's failure shape
  reproduced — outside-region events → built()=∅ + escapes counted; monitor cap exactness;
  class-c-past-cap still quarantines). FlowViz 16 → **23** (`tests/test_panels.py`: blind banner,
  quarantine → warn, agent-event split, agent status file → snapshot, gpu fold, helpers).

---

# ✅ A7 ACTOR FILTER IMPLEMENTED (2026-07-04) — agent blocks can never become intent evidence

Per the D0–D4 readiness review (vault, 2026-07-03) F1–F3, user-approved as concrete implementation:

- [x] **Contract** *(actor strings re-pinned 2026-07-04 by user decision — supersedes the first-cut
  `mica:` namespace)*: the human plays as **`HumanBuilder`**, the agent joins as a real player named
  **`MICA_AI`** (replicas `MICA_AI_1`, `MICA_AI_2`, …; `is_agent_actor()` = exact or `MICA_AI_`
  prefix, so `MICA_AIX` never matches). Agent-as-player is the design's point: the capture tags
  events by username, so the agent's blocks are attributed for free — and the same identity is the
  D5 embodiment. Older captures with other human usernames (`Player940`) stay valid (non-agent =
  human). Filtering happens at consumption, never capture — raw logs stay complete.
- [x] **AI player + live visualization (2026-07-04)**: `capture/mineflayer-bot/agent.js`
  (`npm run agent`) joins the offline server as `MICA_AI` (env `MICA_AGENT_NAME` for replicas) —
  presence-only per the D5 authority pin (looks at the human, never moves into the workspace,
  placement disabled). On spawn it auto-opens **FlowViz** in the browser (reuses a running instance
  on :8321, else launches `python flowviz.py --raw-dir capture/raw` from `D:\2026projects\mica-flowviz`;
  `MICA_NO_FLOWVIZ=1` to skip). `MICA_A7_TEST=1` places three stone blocks after 10 s for the
  in-game A7 verification.
- [x] **Topology finding (corrected after reading the capture source — no mod work needed)**: the
  mod's place/break hooks run SERVER-side in the client JVM's integrated server (`world.isClientSide`
  guards; actor = the placing player's name), so in the singleplayer rig, **Open to LAN + MICA_AI
  joining is fully attributed by the existing capture** — the agent's events carry actor "MICA_AI",
  the replay check passes (events exist for its blocks), and mixed-actor sessions are proof-grade
  as-is. Procedure: play singleplayer → Open to LAN (cheats ON if running the A7 test) → `npm run
  agent` (auto-discovers the LAN port via the multicast announcement; `MICA_PORT` overrides). The
  one topology needing future mod work is a dedicated-server rig (client mod hosts nothing there).
  The observer bot's attribute-everything-to-the-watched-player heuristic (`bot.js`) remains a
  known limitation of that supplementary path only.
- [x] **Evidence side (human events only)**: D1 `_human_events()` feeds both `_tick_behavior` and run
  event-consumption (an agent-only tick classifies from the human's inputs/motion, as if no event
  occurred); D2's feature world filters at `Evidence3DStream.remember` — the single entry point —
  so `built()`/comp/fit never see agent cells and no agent id parks as "unconsumed".
- [x] **Verification side (all events)**: `SnapshotMonitor`'s shadow world, the B0 gate, and event-id
  continuity span every actor (reality contains the agent's blocks; ids stay dense).
- [x] **Fidelity accounting**: run_d1 scopes unconsumed/duplicates to human events; agent events
  reported separately, never failures. Synthetic generator supports mixed actors
  (`ScriptedPlacement.actor`; agent placements never move the pretend human's hand/crosshair);
  `assisted_build()` fixture added.
- [x] **Tests** (`tests/test_actor_filter.py`, 5): namespace predicate; never-scored/never-consumed;
  feature world excludes vs monitor keeps; mixed-session human evidence ≡ human-only session's;
  live-pipeline end-to-end. **Suite 126 → 131.**
- [x] **Gate-feature exposure (F2)**: `LivePipeline.status()` now carries `current_behavior` (D1 open
  run), `player_pos`, `focus_block`/`focus_dwell_ticks`, `belief_snapshot_id` (:= correction tick,
  pinned in vault D5 §7).
- [x] **Authority pins (F3, vault D5 §9)**: build/counterfactual on v0 heads; production thresholds
  frozen only on Phase E calibrated heads; first live demo OBSERVE/SUGGEST/PREVIEW with
  PLACE_LOW_RISK config-disabled until Phase E + in-game A7 verification.

---

# ✅ H3D CHANNEL WIRED (2026-07-03, later) — Uni3D structure evidence flows B2→B3; belief-side readout GATED OFF (null reported)

- [x] **Producer**: `Evidence3DStream(world, h3d_fn=...)` — with `run_d2 --h3d`, every B2 record
  carries the frozen Uni3D-B embedding of the strictly PRE-action player-built cells (shares the
  feature cache: recomputed only when events changed the region). `Uni3DShapeHead.h3d()` returns
  the contract's plain tuple so pipeline code stays torch-free. Provenance pins encoder sha /
  cloud points / seed. **Real artifact: `194242.evidence3d.jsonl` carries h3d(1024) on 68/148
  records** (the 80 before the first placement are None by contract); replay + contract + join PASS.
- [x] **Transport fixed** (was the "conceptual only" gap): `evidence3d_to_dict` no longer hardcodes
  `"h3d": None`; `fuse_dicts` parses it back — `FusedEvidence.h3d` is real offline and live.
  Overwrite guard mirrors run_d1's pixel guard (a plain run refuses to erase an h3d-enriched file).
- [x] **Consumer implemented, adoption-gated**: `mica/intent/h3d_readout.py` (stdlib 5×1024 linear
  readout; explicit activate()/use_default() for probes) + a centered, goal-free-gated term in the
  deliberative bump only (zero-sum across goals — balance law; ramp-in below ~8 built cells where
  the linear probe is near-chance; fade-out as best fit×comp anchors, where templates take over).
  Trained by `scripts/train_h3d_readout.py` (probe recipe, all 30 sessions, train acc 0.837).
- [x] **Fusion gate run** (`scripts/probe_h3d_fusion.py`, session-held-out, 6 folds, three arms on
  identical records): **+0.024…+0.030 mean earliness in every swept config** (w=1.0, w=0.5,
  anchor-gated, ramp+gate) **but −0.033 accuracy in all of them — exactly one session** (16,
  production, a 0.98-sustained coin flip in the plain arm; diagnosis: early 2–3-block clouds
  entrench a wrong lead; at the final record the readout favors the truth and the flip persists by
  path dependence). Pre-stated criterion (earliness at no accuracy cost) → **FAIL** →
  **weights file NOT shipped, term default-off**; `train_h3d_readout.py` refuses to ship while the
  gate says FAIL (`--force` overrides). Sweep stopped at the pre-stated cap — no forking paths.
- [x] Torch-free CI for the channel via a stub head: producer cache semantics, JSONL round trip,
  typed fusion carry, deliberative-only movement, heuristic untouched, and the balance-law
  exactness check (uniform readout ⇒ bit-identical likelihoods). Tests 122 → **126**.
- [ ] **Next**: rerun `probe_h3d_fusion.py` with the real captures in the corpus mix (the channel's
  unique value should appear where templates mismatch free builds — scripted template-exact data is
  the symbolic channel's home turf); if PASS, `python scripts/train_h3d_readout.py` ships the
  weights and the term turns on. Phase E's trained adapter supersedes this hand-coded term either way.

---

# ✅ LIVE RUNTIME CONVERSION (2026-07-03) — B1/B2/B3 now run in-game (spec: vault D6)

- [x] **U-2 actually closed** (the 06-29 entry only fixed the docstring): D1 is now
  `Evidence2DStream` — a tick-fed FSM (rolling 21-tick buffer, deferred scored emission for
  build runs so event ids finalize, held contexts preserving batch order); `evidence_stream()`
  is the recording driver over the same machine. D2 is `Evidence3DStream` (harvest/on_correction,
  bounded delta_comp state); `build_evidence3d()` is its wrapper. **One incremental core, two
  drivers — the batch algorithms are deleted, every old entry point unchanged.**
- [x] **Golden equivalence proven, serialized-bit-exact** (`tests/test_stream_equivalence.py`,
  `tests/test_live_pipeline.py`): streaming == batch over every sample build, all corpus plans,
  every gate-passing real capture, and end-to-end through mod-shaped wire bytes (in-order AND
  out-of-order) — B1, B2, fused, and belief-at-corrections. Regenerated real artifacts
  byte-identical to HEAD code; `belief_summary.json` reproduced byte-identical (0.933 / 0.416).
  *(Found while proving it: the banked `194242.evidence3d.jsonl` predated the 07-03 F1
  fit×comp fix — its production style still read "crop field". Refreshed; now current.)*
- [x] **Live front half**: `capture/live_ingest.py` (bounded tick reorder over the lossy socket,
  depth 32 > the reader's 16-slot frame wait; duplicates/stale/gaps counted),
  `validation/live_checks.py` (`LiveGate` — per-packet tick/alignment/event-id-continuity/
  prior-moment checks; full `b0_gate` still runs on the disk copy at session end),
  `capture/wire_synthetic.py` (mod-side wire encoder for tests; `test_live_stream` helpers moved in).
- [x] **`mica/live_pipeline.py` + `scripts/run_live.py`**: per-moment chain D1→D2→fuse→tracker
  (join verified live, run_tracker's invariants asserted every correction), authoritative belief
  advances only at corrections (bit-exact vs offline) with ~1 Hz display-only `drift` lines,
  per-record flushed logs (`.evidence2d/.evidence3d/.fused/.belief.jsonl` + `.live_run.json`),
  `SnapshotMonitor` polled mid-session, symbolic-first with the `pixel_head` hook reserved.
  Replay mode (`run_live.py <session.jsonl>`) is the one-command equivalence proof — PASS on
  `194242` (378/148/148/378 lines). Live mode integration-tested over a real localhost socket
  (`tests/test_run_live.py`), including a dropped-moment run (gap counted, not proof-grade, disk intact).
- [x] **Latent discovery bug fixed**: `newest_capture` only excluded `.evidence2d.jsonl`, so a
  fresh `.evidence3d.jsonl` (or the new `.fused/.belief` logs) could be picked up as "the newest
  capture". Now excludes all derived suffixes.
- [x] **Corpus default corrected**: `make_scripted_corpus.py` defaulted to `--per-goal 4` while
  the banked corpus is 6/goal — the no-flag command now reproduces the banked 30 sessions
  byte-for-byte (labels.json + report verified against pre-run hashes).
- [x] Serializers centralized in `contracts/serialize.py` (+ `packet_to_dict`, `fused_to_dict`,
  `belief_to_dict`); `jsonl_ingest.packet_from_dict` promoted (shared by disk + live readers).
  **Tests 91 → 120.**
- [ ] **Remaining**: the in-game smoke session on the capture rig (human at the modded client —
  place/break/walk/F9 while `run_live.py` shows the belief moving, then quit and replay-verify);
  `mica-review` pass on the stream cores.

---

# ✅ GOAL TAXONOMY (2026-07-02, late night) — G is now hierarchical; reviewed math preserved

- [x] **G = 5 categories × 2 style subtypes** (`mica/contracts/goals.py`, `TAXONOMY_VERSION 1`;
  subtypes ↔ template instances, `TEMPLATE_SET_VERSION 2` — incl. the post-and-rail pen the real
  free-build session motivated). First proof target unchanged at category level (all reviewed
  thresholds stand). Leaf-level belief licensed by the **nested resample kernel** (03 amendment —
  exact Chapman–Kolmogorov composition verified; all structural theorems carry).
- [x] **Coarse-to-fine via the streams, not wider vectors**: s_goal stays 5-dim category-level
  (D1 amendments); style evidence = B2's new `subtype` field (winning instance) + state_feats cues.
  Goal-symmetry rule extends: leaf-indexed features are deliberative-only (D3 constraint 4).
- [x] **Corpus regenerated under the taxonomy**: 30 sessions, all 10 styles exercised, style accuracy
  at build end **1.0**; category identifiability curve keeps its shape (0.2 → 0.43 first 30%, 0.97
  end). Docs amended: 02, 03, 07 Q1, D0 (B2 box), D1 design+spec, D2, D3. Tests 74 → 75.

---

# ✅ D3 WIRING AUDIT + FIX (2026-07-03) — B1 and B2 now actually fuse, verified

- [x] **FLAW FOUND: the D1 wire dead-ended at the D3 boundary.** The first tracker run consumed
  structure + action label only — `state_feats`, `focus`, `idle` were serialized then dropped at load;
  `global_feats`/`pose`/`subtype` dropped too; s_goal was None corpus-wide. No B3 contract existed;
  the handoff was an ad-hoc dict zipped by order.
- [x] **FIXED: `mica/contracts/b3.py`** — `FusedEvidence` (every channel, deliberative-only fields
  marked) + `fuse()`/`fuse_dicts()` that VERIFY each join (tick + event ids; mismatch refuses).
  Heads consume the full bundle: held-item style match (g-bearing, delib-only), streak as a gain
  gate on the bump, dwell → INSPECT; `strip_behavior()` = the D1 with/without probe, first-class.
  Synthetic builders now equip what they're about to place (pre-action windows see the real hand).
- [x] **Two more balance-law lessons, measured**: streak in the base PLACE rate broke the law exactly
  when placements arrived — fused arm scored WORSE than stripped (0.800 vs 0.933) until streak moved
  to bump-gain. Goal-free context gates the goal-bearing term; it never shifts base rates.
- [x] **VERIFIED, two-arm run (30 sessions, no pixels)**: fused **0.416** sustained-from vs stripped
  0.455 vs floor 0.428 — **D1 behavior buys +0.039 earliness at equal accuracy (0.933), and the fused
  tracker beats the structure-only floor for the first time.** *(12-F3 caveat, 2026-07-03: the
  synthetic held-item cue is noiseless by construction — the generator equips the exactly-right block
  20 ticks ahead — so this number is an upper bound pending the real-capture rerun.)* Handoff verified
  per record; invariants every step; tests 82 → **86** (channel-arrival, held-item-reaches-evidence,
  mismatch-refusal, load-bearing-wire tests).
- [x] **Three-arm stream ablation added (2026-07-03)** — `strip_structure()` completes the triangle.
  Measured over 30 sessions: **3D structure contributes +0.500 accuracy and +0.152 earliness over
  2D-only** (0.933/0.416 fused vs 0.433/0.568 2D-only); 2D behavior contributes +0.039 earliness at
  equal accuracy. Fusion beats both single streams and the structure-only floor. The "does 3D help
  intent inference" question now has a standing measured artifact (`belief_summary.json`).
- [ ] Not yet consumed by v0 heads: `global_feats` (flows, unused), `h2d`/`h3d` (None until
  pixels/Phase E — h3d now has a licensed producer, below), focus dwell (inert on synthetic —
  real captures will exercise it).
- [x] **Pretrained-3D arm DECIDED + measured (2026-07-03): Uni3D-B**, chosen over OpenShape
  (vanilla ViT via timm — no MinkowskiEngine on Windows — and stronger zero-shot: 55.3 vs 46.8
  LVIS, 88.2 vs 85.3 MN40; both saved papers read in full). Adaptation: player-built cells →
  surface point cloud (`mica/perception/shape3d.py`), vendored repo behind a pure-torch FPS shim
  (`vendor/uni3d/mica_uni3d_loader.py`, verified against brute force), prompts embedded with the
  PAIRED teacher (EVA02-E-14-plus). **Two probe verdicts on the corpus (chance 0.2):** zero-shot
  text cosines FAIL the earn-its-place gate (≈chance while building, 0.50 at completion vs floor
  0.967; y-up and prompt-phrasing artifacts ruled out) → **no s_shape field in B2**; the raw
  1024-dim embedding PASSES a session-held-out linear-readout probe (beats the symbolic floor at
  every bin through 70% progress, trails only late) → **`embed()` is the licensed h3d producer
  for Phase E's trained adapter** (`capture/scripted/shape3d_probe.json`,
  `h3d_linear_probe.json`; setup: `scripts/setup_uni3d.py`). Rerun both probes on real captures.
- [ ] **Remaining pretrained-3D comparator arms (open)**: (2) AssistanceZero's released voxel
  goal head as the in-domain pretrained comparator (the amortized-belief baseline the eval plan
  wants); (3) SpatialLM strictly as a measured syn2real transfer arm; (4) multi-view MineCLIP
  orbit as a depth-via-views proxy. All framed as ablation arms against the symbolic structure
  channel, which is the floor they must beat.

---

# ✅ D2 GATE FIXES + D3 KICKOFF (2026-07-03) — tracker runs end to end

- [x] **D2-gate F1 fixed**: instance-selection key = fit × comp (each factor alone fails: fit is blind
  to missing upper layers, comp is terrain-inflatable — both observed). Corpus style accuracy 0.967 →
  **1.0**; real-terrain style reads de-skewed (production: crop field → bordered plot). D2 doc aligned
  (one coherent winning instance, not per-feature maxima).
- [x] **D2-gate F2 fixed**: `scripts/d2_progress_report.py` — the one-command phase artifact. First
  run: comp monotone **10/10**, edit non-increasing **10/10** (scoped: within an instance's reign;
  dips exactly at style switches are the fine layer updating — verified, both dips sat at switches),
  rotation recovery 2/2, separation curve persisted. **D2 phase criterion: PASS.**
- [x] **D3 OPENED — the belief tracker is implemented and runs end to end.**
  `mica/intent/tracker.py` (Algorithm 1, faithful: resample kernels, ε-floor, predict/correct) +
  `mica/intent/heads_v0.py` (hand-coded baseline heads honoring all four D3 constraints) +
  `scripts/run_tracker.py`. Property tests for every theorem: mass preservation, exact lazy
  prediction (CK), normalizer floor, humility bound, goal-freeze under z=1. Tests 75 → **82**.
  First full belief run over the 30-session corpus: **final category accuracy 0.933**, sustained-
  correct from 0.465 of build (floor: 1.0 / 0.368 — clean scripted templates are the floor's home
  turf; the tracker's edge is designed to come from behavior+pixel evidence and noisy/free builds).
  Two modeling lessons banked in code comments: the softmax balance law (base must under-predict
  placement vs realized build fraction or pauses invert the evidence — first cut scored 0/30), and
  progress-gain saturation erasing discrimination.
- [ ] **D3 next**: adapter + trained heads need the data recipe run (Source B labeling over real
  builds) and calibration protocol; mode-semantics separation is weak under hand-coded heads
  (P(z=1): 0.73 deliberate vs 0.78 shortcut — directionally right, validation deferred to trained
  heads). **The four real captures remain the standing data item.**

---

# ✅ GOAL TAXONOMY (2026-07-02, night 2) — G is now five functional categories + style subtypes

- [x] **G restructured** per user decision: habitation / infrastructure / production / defense /
  decorative-civic, each realized by style subtypes (15 in v1; `contracts/goals.py`, TAXONOMY v2).
  Belief stays over the 5 categories — all reviewed math + |G|=5 contracts untouched; the subtype is
  an evidence-side readout (`PerGoalStructure.subtype` = winning template instance). Templates v3
  (15 instances incl. treehouse/road/perimeter-wall/flower-garden/fountain); MineCLIP category
  anchors pool their subtypes' prompts; the nested two-rate kernel for a future leaf-level belief is
  recorded in 03 (verified: composes exactly; U corrected to the category-uniform product measure so
  the flat filter is the exact marginal). Docs amended across 02/03/07/D0/D1/D2/D3/06/overview.
- [x] **Corpus regenerated under the taxonomy**: 30 sessions, 15 styles, 30/30 unique sequences;
  category identifiability at/below chance through 20% of build (0.13–0.17), 0.97 at end; style
  accuracy at end 0.967. Coarse-to-fine, measured. Tests 74 → **75**.
- [x] **The motivating treehouse session validates the change**: fit-first structure ranking now
  recovers its CATEGORY (habitation, fit 0.76) with the style honestly uncertain — labeled in
  `capture/raw/labels.json` as the first real capture (habitation). **Four real captures remain**
  (infrastructure, production, defense, decorative — any subtype each).

---

# ✅ B2 IMPLEMENTED (2026-07-02, late night) — v1 symbolic core, verified on real data

- [x] **B2 v1 built per the amended D2 spec**: `contracts/b2.py`, `perception/templates.py` (versioned
  set, 5 placement-buildable instances), `perception/voxel_replay.py` (event sourcing + taxonomy),
  `perception/evidence3d.py` (rotation-aware pose search, world-read comp / player-read fit),
  `validation/evidence3d_check.py` (+ B1↔B2 join check), `scripts/run_d2.py` (gate-coupled; quarantines
  on class-c divergence or crop escape). Tests 55 → **70**.
- [x] **Real-data verification (session `194242`)**: replay a=3 b=1 **c=0** PASS; 148 records; join
  one-to-one; honest free-build readout (farm comp 0.85 / fit 0.08 — terrain vs player split working).
- [x] **Source-A scripted corpus generator built + corpus generated (2026-07-02, late night).**
  `mica/capture/scripted_goals.py` + `scripts/make_scripted_corpus.py`: 30 seeded sessions (6/goal),
  variance across pose (offsets + all 4 rotations), order (4 strategies), pacing (gap sd > mean),
  mode (deliberate/shortcut ground truth for z), mistakes (93 break events), completion (0.85–1.0),
  materials. 30/30 unique action sequences. **Early-identifiability curve: structure-only accuracy
  0.23 → 0.37 over the first 30% of build (chance 0.2), 0.63 at 40%, 0.93 at end** — early prefixes
  genuinely ambiguous, resolution learnable. Artifacts in `capture/scripted/` (regenerable, seeded;
  gitignored). Tests 70 → 74.
- [ ] **Still needed from the user: the five REAL in-game template captures** — scripted sessions have
  no pixels, so the s_goal early-separation probe (stride 1/6/20 A/B) and the VPT sanity probe require
  real play. Build each `templates.py` shape once; the corpus's labels.json convention applies.

---

# ✅ RESOLVED (2026-07-02, night) — VPT/MineCLIP alignment review fixes, verified by real reruns

- [x] **G-10 (AL-F1) — s_goal now computed through MineCLIP's TRAINED reward path.** The checkpoint's
  video adapter + learned residual gate (σ(w)=0.9456 — it trained) were being bypassed; `score()` now
  applies them, keeping the raw cosine (no logit_scale — D3 owns temperature). *Verified:* both real
  artifacts regenerated (means shifted as the A/B predicted), fidelity + contract PASS, provenance pins
  `s_goal_path` + `s_goal_clip_stride`.
- [x] **G-11 (AL-F2) — clip-span mismatch verified + probe axis landed.** MineDojo paper confirms:
  MineCLIP trained on 16-SECOND snippets carrying 16 frames (≈1 fps); our 0.8 s consecutive clip is a
  20× span shift. `run_d1 --sgoal-stride N` implements the A/B (end of clip stays strictly pre-action).
  *Verified end-to-end at stride 6.* D1 design + spec amended. **Probe must A/B stride 1 / 6 / 20.**
- [x] **G-12 (AL-F3) — goal-symmetry rule in the contract.** D0 B3 + D3 head stub now forbid s_goal
  (or any per-goal B2 feature) from reaching the heuristic head — protects 09-F4 identifiability.
- [x] **G-13 (AL-N1/N2) — D3 Phase-E constraints recorded:** `a_hat_conf` never multiplies into
  likelihoods; s_goal arrives unscaled (~1e-2 spreads pre-temperature).

---

# ✅ RESOLVED (2026-07-02, evening) — B1 gate review fixes, verified by tests + real reruns

- [x] **G-5 (B1-F1) — `run_d1` refuses B0-gate-failing sessions.** Gate checks run before evidence is
  built; failing checks printed; `--allow-ungated` escape hatch for debugging. *Verified:* end-to-end
  subprocess test + the crash-truncated real session is refused.
- [x] **G-6 (B1-F2) — periodic context records implemented** (decision: implement now). `evidence_stream`
  emits unscored ~1 Hz records inside runs longer than a second (`scored=False`, no event consumption);
  validator updated (unscored ⇒ empty `event_ids`; scored PLACE/BREAK ⇔ events); contract docstrings
  updated. *Verified:* real reruns — `161017`: 285 scored + 14 context; `010831`: 164 scored + 10
  context; fidelity + contract PASS on both; scored counts unchanged from the pre-change runs.
- [x] **G-7 (B1-F4) — h2d window decided: the 1 s pre-action window is intended.** D1 design + spec
  amended (dated); native-context is the recorded upgrade path if D3's probes need it.
- [x] **G-8 (B1-F5) — overwrite guard.** A symbolic-only `run_d1` refuses to erase a pixel-enriched
  `evidence2d.jsonl` (`--overwrite` to force). *Verified on the real artifact.*
- [x] **G-9 (B1-F6/F8) — provenance pins `goals` + `prompt_templates`** (they define s_goal);
  `PROMPT_TEMPLATES` public; `run_d1` imports `GOALS` from the contract. *Verified: sidecar regenerated.*

**Tests: 54 passing** (48 + 6 new). Still open from the B1 review (data, not code):
- [ ] **B1-F3 — scripted-goal captures (5, one per goal) + early-separation probe + idle-precision
  sample, BEFORE corpus collection.** Real-data flatness (spread ≈ 0.036) says this de-risk is urgent.
- [~] **B1-F7 — idle/inspect granularity + confidence** — revisit after the idle-validity measurement.
- [~] **HF tokenizer** — MineCLIP's text side hits the HF Hub on first load; pin/cache for offline runs.

---

# ✅ RESOLVED (2026-07-02) — B0 gate review fixes, verified by tests

- [x] **G-1 — crash-truncated sessions hard-fail the gate.** `CapturedSession.declared_is_provisional`
  set by `jsonl_ingest` when the manifest count is missing/`-1` (still loads, logs a warning); new gate
  check "manifest finalized (clean stop)" rejects it. *Verified:* truncated real capture with a
  provisional manifest now FAILS (was: PASS certifying "no dropped events" with 5 events lost).
- [x] **G-2 — live stream never drops whole moments.** `read_moments` waiting-buffer eviction now yields
  the packet frameless instead of discarding it; docstring documents the out-of-order property.
  *Verified:* new overflow test — 18 framed packets, oldest two yield `(packet, None)`.
- [x] **G-3 — manifest frame provenance.** `SessionManifest` carries `frame_every` / `frame_width_px`
  (0 = older recording); ingest reads them; the mod writes them (R-1 checks in-game).
- [x] **G-4 — D0 amended (2026-07-02)** — frame contract (≥320w/≥160h aspect-preserved as-available),
  `tick` semantics, `declared_event_count` semantics, predictive place, truncated-session rejection,
  live-stream scope (B1 + B2). The contract text is the source of truth again.

**Tests: 48 passing** (44 + 4 new: provisional-load, provisional-gate-reject, manifest frame fields,
live-stream overflow). Both real captures still PASS the gate.

---

# ✅ RESOLVED (2026-06-29) — fixed + verified

- [x] **B-1 / I-1 — one shared `newest_capture()`** (`mica/capture/discovery.py`), used by `b0_gate.py`,
  `inspect_capture.py`, `run_d1.py`; excludes `*.evidence2d.jsonl`. *Verified:* with a D1 output on disk,
  all three pick the real capture and `b0_gate` no longer crashes.
- [x] **I-2 — VPT `load_state_dict(strict=True)`** (matches `mineclip_head`). *Verified:* loads, h2d = 1024-d.
- [x] **M-1 — checkpoint provenance recorded.** `run_d1 --pixels` writes `<session>.d1_provenance.json` with
  sha256 of `vpt-1x.weights` + `mineclip_attn.pth`; the B0 `SessionManifest` was trimmed to B0-only fields
  (the VPT/MineCLIP hashes were misplaced there) and now carries `event_schema_version`, which the mod
  writes and `jsonl_ingest` reads. *Verified:* sidecar written with real hashes.
- [x] **M-2 — `pyproject.toml [perception]` pins the real stack** (numpy/scipy/Pillow/transformers/kornia/
  dm-tree/gym3/imageio/gdown + `x_transformers==0.27.1`), with torch + mineclip/vpt out-of-band steps documented.
- [x] **U-1 — unconsumed events no longer silent.** `run_d1` reports `unconsumed` and fails fidelity if any
  block event lands in no record. *Verified:* 0 unconsumed on the real session.
- [x] **U-2 — `evidence_stream` docstring corrected** to "offline-batch (sorts the whole session); a live
  source needs a rolling-window variant."
- [x] **U-3 — preprocessing verified + matched.** VPT's own `resize_image` is a plain `cv2.INTER_LINEAR` squish
  to 128×128; both wrappers now resize with `Image.BILINEAR` to match (MineCLIP normalizes internally).
- [x] **U-4 — PIL handles closed** in `_enrich_pixels` (load into memory inside a `with`).
- [x] **U-6 — `_CONFIDENCE` lookup defaults** (`.get(a_hat, 0.7)`) instead of `KeyError` on an unlisted action.
- [x] **U-7 — MineCLIP prompt ensemble + grammar.** Three templates per goal averaged at load, article
  picked per goal ("an animal pen"). *Verified:* loads, s_goal = 5 values.
- [x] **C-1 — `mica/capture/source.py` deleted** (dead `CaptureSource` Protocol).
- [x] **C-2 — `scripts/validate_capture.py` deleted** (subsumed by `b0_gate.py`).
- [x] **C-3 — `make_proof_logs.py` kept** (synthetic proof, distinct from the real-data gate).
- [x] **C-4 — large binaries git-ignored** (already in `.gitignore`: `models/`, `vendor/`, `capture/raw/*`).
- [x] **C-6 — `timebase.ends_before_action` kept** as the documented contract snapshot-helper.
- [x] **L-1 — `sync_report` `dropped`** now counts expected ids that never showed up (`len(expected - present)`),
  immune to out-of-range ids.

**Tests:** 44 passing throughout. `run_d1 --pixels` on the real 285-record session: fidelity PASS, B1 contract PASS.
