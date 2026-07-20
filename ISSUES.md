# MICA — Code Audit & Issue Tracker

> Issues found auditing the implemented code (B0 capture + B1/D1). **Most were fixed and verified on
> 2026-06-29** (see RESOLVED below). What remains needs either your in-game rebuild check, or is
> deferred-by-design for a later phase — collected in REVIEW LATER. The vault mirror
> (`Implementation Log/Issues & Audit Tracker`) is a dated snapshot and is now behind this file.
>
> Status: `[ ]` open · `[x]` done · `[~]` deferred-by-design.

---

# 🎯 D9 ACTING LAYER + SUGGESTION-ACCEPTANCE SIGNAL (2026-07-17)

Vault note: `MICA System Design/D9 - Acting Layer - Calibrated Autonomy and the Acceptance Signal.md`
(user direction: real-life assistance, not a supervisor -- calibrated uncertainty becomes the
internal supervisor; three stages, each measurement-gated; intention recognition unchanged as thesis).

- [x] **Suggestion-acceptance logging landed** (D9 section 3, pre-registered definitions v1):
  gate trace rows carry `proposal_first` on every proposal-bearing read (the unvoiced CONTROL
  group); agent.js logs each actually-voiced advisory to `agent-<NAME>.voiced.jsonl` (the 30 s
  throttle lives body-side); `mica/validation/suggestion_acceptance.py` + script compute
  followed exact/near/type, contradicted, ignored, and the LIFT vs the unvoiced base rate.
  Tests cover every tier + window edges + negative lift; pre-D9 sessions report n=0 gracefully.
- [x] **run_decoder_eval re-adjudication launched** (user decision: re-adjudicate the 07-13
  stage-A pin after tonight's retrain clobbered it) -- verdict lands in decoder_report.json.
- [x] **Re-adjudication verdict (2026-07-17): OQ1 PASSES for the first time** on the
  leak-free models + regenerated corpus (B 1.6372 < A 1.66; coherence kept; h4 tau-acc
  0.7432) -- stage B (MTP) earns the live slot by the pre-registered rule. The stage
  decision is now EXECUTABLE: models/decoder_stage_pin.json (with full history incl.
  the 07-16 silent clobber) + train_decoder.py refuses to flip stages without it.
- [x] **Gate re-frozen against tonight's models** (run_decoder_eval + run_gate):
  delta_hat 0.2295 s, theta_1 0.45, c_min 0.9, theta_suggest 0.338 / theta_place 0.438
  (review predicted ~0.34/0.44); counterfactual criteria ALL PASS. Honest consequence:
  theta_place (0.438) > validation conf max (0.427) -- placement is a-priori
  UNREACHABLE until calibration improves (D9 Stage 1), exactly as the review's
  question 2 anticipated. Advisory suggestions now fire at a calibrated-percentile
  rate instead of the stale 37%.
- [x] **Advisory-safe body fixes committed** (ecec637): A7 name refusal, stop-anywhere
  + stopDigging, A7-test interlock, farmland avoidance, nearestHuman.
- [ ] **Still TODO before any --place session**: the freeze's monotonicity gate
  (reject thetas whose selected-reads accuracy is below rejected-reads accuracy),
  gate_v1 provenance stamps + LiveGateRunner refusal, --place/--heads coupling,
  persistent stop across relaunch (F6), and the gather/materials majors (F4, F10,
  F11, F16, F17).

---

# 🧠 BELIEF-TRACKER FULL-STACK REVIEW FIX PASS (2026-07-16, night)

Vault note: `MICA Intent Belief Tracker/13 - Belief Tracker Full-Stack Review (mica-review).md`
(93-agent adversarially-verified review; 39 findings, 0 refuted). User decisions: bundled refit;
07-13 arms table RETRACTED outright; falsification = the vault's stricter criterion; idle DROPPED.

- [x] **F1 BLOCKER — idle target leak.** `idle == 1[a_hat=IDLE]` sat in `shared_dense` and both
  trained heads read it (flip test on shipped weights: P(IDLE) 0.000→0.997; idle corrections'
  L-table collapsed to ~constant). Removed (SHARED_DIM 14→13); featurizer LEAK TEST added
  (`tests/test_features_leak.py`: no head feature may be a function of a_hat — sabotage-proven);
  `b3.py` flow comment now names a_hat/event_ids/idle as target-not-feature.
- [x] **F2 BLOCKER — stale exposure grouping.** Trainers now pin `train_sessions_real` +
  `validation_sessions` in the model jsons; `run_arms` derives its groups from them and asserts
  arm1/heads_v1 agree; hand-maintained `_REAL_TRAIN_SEEN` deleted. **The 07-13 arms_report is
  RETRACTED** (user decision) → `capture/raw/retracted/RETRACTED-20260716-*`.
- [x] **F7 — falsification criterion reconciled to the vault's words** (user decision): a session
  succeeds iff top-1 locks on AND STAYS by 40% of the build; the arm passes iff successes beat
  1/|G| chance (exact one-sided binomial, α=0.05). `intent_metrics.falsification_verdict` + 06
  dated amendment; mean early margin demoted to diagnostic.
- [x] **F8 — pooled-then-binned ECE** per (group, arm) cell in run_arms (mean-of-per-session
  ECEs kept only as `per_session_ece_mean_DISPERSION_ONLY`).
- [x] **F9 — Arm-0 floor swap recorded**: 06 amended (fit-first (fit, comp) is the floor;
  match−waste unreconstructable at B3 — needs template cell counts).
- [x] **F6 — held-out calibration drops tuning-validation sessions** too; verified live: the
  refit's held-out run drops 11 trained + 1 tuning session.
- [x] **Fusion-review fixes folded in** (same refit): renormalized sweep objective
  (Z(a_obs)/Σ_a Z(a)) + uniform p̂ for the class-balanced backbone. **Honest re-sweep verdict:
  (w2, w3, γ) flipped from (0.0, 0.5, 1.0) → (1.0, 0.5, 0.0)** — γ collapsed to zero and the 2D
  stream is no longer dropped; the 07-13 knobs are confirmed scale artifacts.
- [x] **Guards landed**: correct() positivity assert + floored() L∈[0,1] assert (F19/F31);
  run_tracker/run_arms/session_metrics skip-and-record degenerate sessions (F17/F21);
  step() rejects negative mass (F22); pivot seam guards (F23); heuristic goal-symmetry assert
  in the 12-F8 margin readout (F35); v0 SCAFFOLD folds to PLACE (F28); six-fold duplicated
  guard tests deduped in test_tracker.py + test_b3_fusion.py (F18).
- [x] **Vault batch**: Foundations gains A3′ (evidence exogeneity) + λ>0 + log-space fix +
  TV constant + Lyapunov phrasing + A7 evidence-attribution sentence (the 06-17 review's four
  fixes, finally applied) + Π rename + Seneta §3.4 pin + CLIPS authors filled; 03 gloss/flooring/
  α,β; 02+07 taxonomy v3 amendments + τ(k) onset anchor; 01/00 decoder scoping + provenance
  banners; 06 criterion + floor amendments.
- [x] **Coordinated refit executed**: arm1 + heads_v1 retrained leak-free with pinned exposure;
  honest held-out calibration: **ECE 0.2645** (the leak's free idle wins no longer mask
  overconfidence; 12-F8 mode margin now POSITIVE +0.1232, was −0.084); scripted tracker re-bank:
  fused 0.833/0.472 vs floor 1.0/0.425 (01 amended). heads_v2 re-swept honestly (γ=0).
- [x] **Four-arm table regenerated** (honest grouping: real_never_seen = 20 sessions, was ~5
  contaminated): **every arm FAILS the reconciled pre-registered criterion in every group**
  (closest: arm0 scripted 3/5, p=0.058) — the old mean-margin test would have passed several
  cells; the registered criterion passes none. The load-bearing finding: on real never-seen
  sessions the structure floor collapses to 0.050 while both tracker arms reach 0.300 (6× —
  the belief earns its keep exactly where templates stop being exact); v0 pooled-ECE 0.0877
  is the best-calibrated arm. Earliness claim = OPEN pending the richer corpus, not passed.
- [x] **decoder_v1 retrained** on the 130-dim leak-free context (corpus + Stage A/B; stage B
  best holdout token NLL 1.7679). **Full suite: 353/353** — the fix pass is closed end to end.

---

# 📐 PERCEPTION-LAYER REVIEW FIX PASS — SYMMETRY + VERTICAL ANCHOR (2026-07-16, later)

Vault note: `Implementation Reviews/2026-07-16 - Perception Layer Review (mica-review).md`.
Two D2 symbolic features were computing wrong values that passed every range check and test
— confirmed by execution, fixed same day, both sabotage-proven, corpus regenerated.

- [x] **F1 — `_symmetry` axis mirrors were algebraically wrong** (computed the negated
  centroid offset, not the reflection; score was translation-VARIANT — the longhouse's own
  footprint read 0.71). Fixed with exact doubled-coordinate reflections; new property test
  pins each mirror individually (a doubly-symmetric fixture lets `max()` hide one broken
  mirror — how the bug shipped).
- [x] **F2 — vertical anchor was min-built-y, unstated in D2** — one stray block four
  levels below a perfect cabin sank the template into the terrain: comp stayed 1.0 via
  underground stone while edit_distance exploded 0 → 57. Anchor now registered by
  built-cell alignment voting; `Pose` carries `dy` (defaulted — old records parse);
  stray-block + platform-first regression tests.
- [x] **Symmetry gated** (user decision): `d2_progress_report` criterion 4 — clean
  full-completion builds must read ≥ 0.9. Monotonicity criterion scoped per
  (instance, POSE), logging `registration_moves` — dy in the registration surfaced a
  near-tie argmax flip the old criterion had no vocabulary for.
- [x] **D2 doc amended** (dated §3 block): vertical-anchor rule, pose-scoped
  monotonicity, symmetry correction + gate.
- [x] **Corpus regenerated** (user decision: immediately): scripted 30/30 via no-flag
  `make_scripted_corpus.py`, progress report PASS (9/9 monotone, symmetry 1/1,
  separation 0.167→0.967); real labeled sessions 33/33 via per-session
  `after_game.py --redo-evidence` (the one nonzero exit is 210003's documented
  correct quarantine). **19/25 matcher verdicts changed** — the terrain/crop_farm
  pattern dissolved (7 pre-fix crop_farm reads → 1), 223829 flipped contests→AGREES
  (its pairs now train), 022406 + 001126 flipped agrees→contests (their pairs now
  correctly withheld). Regenerated records carry `pose.dy`; suite 350/350.
- Still open from the same review: F3 (pixel-head ring vs stride-20), F4 (h3d width
  check), F5 (two doc one-liners).

---

# 📦 B0 OBSERVATION-PACKET REVIEW FIX PASS (2026-07-16)

Vault note: `Implementation Reviews/2026-07-16 - B0 Observation Packet Review (mica-review).md`.
The D7 `inventory` field's production path was verified correct; every finding was in the
untested seam around it.

- [x] **F1 — `packet_to_dict` dropped `inventory`** (round-trip invariant silently false).
  Fixed in `contracts/serialize.py`; `test_packet_round_trip_preserves_inventory` added
  (proven able to fail by reverting the fix).
- [x] **F2 — D0's B0 box never listed `inventory`.** Dated 2026-07-16 amendment added to D0
  (schema box + leakage rule + field-map row + B1 state_feats note).
- [x] **F3 — context-record leak boundary untested.** `test_context_record_inventory_pinned_before_run_start`
  in `tests/test_d1.py`: a mid-run sample must never reach a context record (`run.t0 - 1` pin).
- [x] **F4 — inventory presence invisible.** `"inventory channel (D7)"` gate line, REPORT-ONLY
  (user decision 2026-07-16): sample count, or "not D7-ready" — never fails a capture; the
  banked corpus predates the channel. NOT a coverage.py probe — every coverage reader hard-gates
  in `b0_gate.gate_checks`, which would have failed all pre-07-10 captures.
- [x] **Symmetry tripwire** (review Q1): `test_maximal_packet_round_trip_covers_every_field` —
  dataclass-introspected maximal fixture; a field added to `ObservationPacket` without both
  serializers now breaks a test instead of landing silently.
- [x] **`scripts/tests/` deleted** (user decision 2026-07-16): a stale git-tracked snapshot of
  `tests/` from the 07-05 recovery commit (`7e8c126`), 15+ files behind, never collected
  (`testpaths = ["tests"]`), zero references repo-wide.

---

# 🧪 HEADS V2 BUILT + EVALUATED — PRE-REGISTERED CRITERIA FAIL; BANKED AS THE NEGATIVE RESULT (2026-07-13/14, night)

D8 implemented exactly as reviewed (all seven checklist fixes): per-stream late
fusion (`adapter_v2.py`/`heads_v2.py`, channel partition structural + tested),
event-gated GM-normalized C_γ-clipped goal readout on Arm 1's backbone (retrained
class-balanced per F7c), held-item dropout + the pooled inventory channel (D7 §3),
`train_heads_v2.py` with the full pinned fit (4 temperatures, offsets, 4,800-combo
(w2,w3,γ,ε,λ) sweep on held-out filter NLPD). `--heads v2` wired into
run_tracker/calibration_report (routing generalized to any heads_<version>).
Checklist tests: `tests/test_heads_v2.py` (8).

**Verdict (held-out real, 18 never-seen sessions): FAIL — banked as the
pre-registered negative result.**
- ECE **0.6888** vs v1's 0.2624 (criterion 2): 14,157 corrections stated ~0.99
  confidence at 0.232 accuracy — the event-gate residual (D8 §3's own recorded
  risk) measured biting: real sessions are mostly event corrections, and γ=1.0
  let the readout saturate the recursion.
- Fused final accuracy 0.179 vs floor 0.393 (criterion 4): FAIL. Chosen fusion
  w2=0.0 (the dial measurably dropped the 2D stream), w3=0.5.
- **The offsets fit degenerated and was auto-disabled**: the pinned split holds
  out ONE real session (single-category) — the fit memorized its label (+6.1
  decorative, NLL 0.002). New diversity guard in the trainer: offsets activate
  only when the fit spans ≥3 categories / ≥2 sessions; shipped as zeros with the
  reason printed. The same thin holdout is what let γ=1.0 win the sweep.
- **The D8 build-order pin was vindicated**: "code only at the 16–20-agreed
  milestone" — built early at 13 agreed (user decision), and the data said not
  yet. Everything stands ready for the retry: one command against a richer
  corpus; the increment-scored readout is the first design change to consider.
- **Fallback clause adjudicated, not applied**: "3D-only becomes live default"
  was written against pre-cascade numbers; post-cascade v1-fused (0.321) beats
  3D-only (0.214), so the live default STAYS v1-fused (dated note in D8).
- Artifacts: `belief_summary_real_v2.json`, `calibration_report_real_v2_heldout
  .json`, `heads_v2_training_report.json`, `models/heads_v2.*` (opt-in only).
- [ ] **mica-review of the v2 implementation math — owed next session.**

# 🎯 PROOF-GRADE FIX: GATE READS OFF THE INGEST THREAD + MOD 0.0.9 BUFFER (2026-07-13, late night)

**The measured cause of every gapped session** (stall meters across the last 10 live
runs): the gate read costs **205–361 ms MEAN** (worst 989 ms) and ran INSIDE the
socket-draining loop — six-plus blocked ticks every second, all session. `stale ≈
gaps` everywhere (review F6's watch condition has fired: late arrivals were being
discarded, most of them late because of our own stalls).

**Fix 1 — `AsyncGateRunner`** (`mica/gate/live_loop.py`): the same LiveGateRunner on
one worker thread. The ingest loop hands over the freshest (belief, fused, status)
and keeps draining; freshest-wins coalescing (no stale queue); block events buffer
through a deque so the runner's cell sets have exactly one mutating thread; a gate
crash degrades to an OBSERVE block instead of killing the worker; a bounded wait
makes lost wakeups impossible to hang on. run_live wires it around the runner; the
stall meter now records from the worker's own clock (expect `gate` timings unchanged
but `moment` worst to COLLAPSE — that is the number that makes sessions proof-grade).
4 new concurrency tests. Torch releases the GIL during the decoder forward, so the
read now genuinely overlaps ingest.

**Fix 2 — mod 0.0.9**: the live-stream drop-oldest buffer grows 64 → 300 messages
(~3.2 s → ~15 s at 20 Hz, ~9 MB worst case) — sized before three GPU models shared
the machine; now it absorbs the rare long stall instead of losing moments. Jar
**mica-b0-capture-1.4.0 rebuilt (19 s, out-of-tree per the recorded schtasks
pattern — the assistant-tree NIO limitation still stands) and verified to carry
0.0.9**. ► USER STEP: install `capture/fabric-mod/build/libs/mica-b0-capture-1.4.0.jar`
into the launcher's mods folder; the next manifest must say `fabric-b0-0.0.9`.

**Not changed, on purpose:** the reorderer's stale policy (F6) — with the stalls
gone, lateness should collapse; re-measure before adding a hold-back window. The
1 Hz snapshot/manifest check (~50–65 ms mean) stays on-thread for now — one tick of
occasional stall sits inside the new buffer's tolerance; revisit only if the next
sessions still show gaps.

**Acceptance bar (post-cascade plan §2): three consecutive real sessions with
`proof_grade: true`.** Sight `worst-stall` in the 1 Hz line dropping to ~tens of ms.

# 🧾 CASCADE #3 VERDICT + STAGE-A DECODER PIN + DEBTS BATCH (2026-07-13, night)

**Cascade A #3 completed** (pair pool 9,881 → 14,202; 13 agreed sessions). Verdict —
accept heads + gate, decoder flagged:
- Held-out ECE 0.225 → 0.2624 (mild; the rejected cascade hit 0.417). Per-goal ECE
  exposes the diversity gap: defense 0.124 vs production 0.5745 / decorative 0.442.
- Gate freeze: θ₁ 0.35 → **0.50**, θ_suggest 0.295, θ_place **0.395**, validation
  conf max **0.4138** — placement reachable on validation for the FIRST time, by the
  pinned rule. Safety criteria ALL PASS (5 train-seen commits).
- **Decoder OQ1 FAILED** (stage B NLL 1.6484 > stage A 1.6035; coherence 0.034 vs
  0.094 — collapsed from 0.213 on the v3 corpus). ► USER DECISION (same night):
  **stage A is the live decoder** — `decoder_v1.pt` = stage-A weights, stage B
  banked as `decoder_v1_stage_b.pt`, pin + caveat recorded in `decoder_v1.json`.
  Known consequence: the horizon≥2 confidence head is stage-B-trained, so K_commit
  effectively caps at 1 live — congruent with the 1-block-per-read authority.
  Retrofit investigation is its own workstream (post-cascade plan §1).

**Debts batch (same night):**
- [x] **Yaw wrap (F5/N-3) — RESOLVED BY MEASUREMENT, premise corrected.** The mod
  records ACCUMULATING yaw (12 sessions scanned: ranges like −1848°..559°, exactly
  ONE single-tick discontinuity — a client re-anchor, not systematic ±180 wrapping).
  The recorded wrapped-difference fix would have CORRUPTED legitimate >180° window
  turns. Implemented instead: a reset guard (`_YAW_RESET_DEGREES` 300) — the tick
  classifier ignores re-anchor jumps, and the window `yaw_delta` subtracts them;
  windows without a reset are bit-identical (no re-summation; golden equivalence
  re-verified). 5 tests (`tests/test_yaw_reset.py`). Corpus regen still rides the
  heads-v2 cascade as pinned.
- [x] **P-1 — auditor ↔ B0 cross-check**: `audit_gate_trace.py` now reads the
  session's raw recording; every MICA_AI place event must match a committed
  directive (HARD fail: unauthorized placement), unexecuted directives are counted
  as warns (body refusal is legitimate). All 36 banked traces pass.
- [x] **P-2 — watcher placement passthrough**: `MICA_PLACE=1` makes lan_autostart
  spawn run_live with `--place`; logged in rig_log's `runlive_start`. Default off.
- [x] **P-3 — gather walk timeout**: shared `gotoWithTimeout` (20 s, timer cleaned)
  now guards BOTH errands; gather also remembers unreachable cells per errand so a
  timeout can't retry the same block forever.
- [~] **N-1 / N-2** fold into the heads-v2 feature work (post-cascade plan §3), not
  patched on v1.
Suite 331 green; node --check clean on agent.js + lan_autostart.js.

# 🧱 D5 §9 PIN LIFTED — PLACEMENT ENABLED BEHIND `--place`: DECLARED-TARGET ROUTE, F4 AGENT-POSITION VETO, BODY EXECUTOR (2026-07-13, user decision)

The pin's two preconditions were long met (calibrated heads 07-05; A7 in-game 07-04,
session 193059). User decisions: declared-target route for the confidence bar (the
GATHER precedent — under the current heads conf never reaches θ_place, D7 review F2),
one block per gate read, human hands the agent its materials. Vault first: D5 §9 dated
amendment + §10 Q1 live-half note; D7 §2 consumer table gains "placement authority"
(quarantine intact — the declaration gates AUTHORITY, never content).

**Change log:**
- `mica/gate/fsm.py` — `FsmConfig.declared_place_enabled` (default OFF: banked
  behavior bit-unchanged), `GateRead.agent_pos` + `place_declared`, the proximity
  veto's third half (human within 4 blocks of the AGENT's body → YIELD immediately —
  review F4 closed), PLACE branch: conf ≥ θ_place OR declared target (config-armed).
- `mica/gate/live_loop.py` — `place=True` lifts the pin (`demo` follows); one
  placement directive per read (`gate.place: {id, cell, block, route}`) — first
  committed action only, Place only, cell not already agent-filled (`agent_cells`,
  optimistic + fed by A7-tagged events: the re-propose guard, since agent placements
  never enter evidence); trace rows carry `authority: "demo"|"place"` and real
  `committed_actions` (route `declared`/`confidence`); materials report's
  `committed_places` counts directives. Directive ids carry a per-run token (a body
  outliving a re-attached mind must never skip a fresh run's ids — review F2).
- `scripts/run_live.py` — `--place` (loud banner; default stays demo; replay_gate
  and counterfactual untouched).
- `capture/mineflayer-bot/placement_math.js` (+ node tests) — support-face search
  (below > sides > above) and eye-to-cell reach; `agent.js` — `runPlaceErrand`
  mirroring gather: one errand at a time, walk with a 20 s timeout (review F3),
  re-check fresh gate state after walking + cell empty + support face, equip, place,
  verify, report (`placing`/`last_place` ride the status file); "stop" halts
  placement AND gather, "go on" re-arms; spawn message updated.
- `scripts/audit_gate_trace.py` — per-segment authority from the trace's own rows;
  place criteria: ≤1 committed action/read, place_low_risk state only, reversible,
  traceable, route verified (confidence → conf ≥ θ_place from the row's own snapshot;
  declared → session present in declared_targets.json); GATHER now a legal state;
  declared-route candidates re-derived via their reason. All 29 banked traces PASS.
- Tests: suite 312 → **326 green** (declared route × config/veto/hysteresis, agent-
  body veto, directive emission + suppression + id uniqueness, materials count,
  demo-unchanged regressions); `node --check` + placement_math node tests clean.

**Review** (`2026-07-13 - D5 Placement Enablement Review (mica-review)`): 1 BLOCKER
fixed same sitting (`isSolidAt` tested `boundingBox === 'solid'`; mineflayer says
`'block'` — every placement would have silently failed "no support face"), 2 MAJOR
fixed (directive-id collision across mind restarts; walk timeout), F5–F7 accepted and
documented. **Wire rehearsal** (194242 @10x, `--heads v1 --place`): proof-grade YES,
zero gaps, 259 reads all `authority: "place"`, ZERO directives (no declaration for
that session — the dormant case behaves exactly as frozen), audit PASS.

**Addendum (2026-07-13, after the first two live attempts — user decision "agent
asks, you provide"):** session 013710 proved the mind side beyond expectation —
**4 placement directives committed via the CONFIDENCE route** (conf 0.43–0.50 ≥
θ_place 0.4, the first time the live belief has ever cleared it; no declaration
needed) — but no body was in-game (runbook step 4 skipped). The next session
(014619) had the body, and the hand-over gave logs while the decoder proposed
stone_bricks: the exact-block constraint reported `missing: {stone_bricks: 8}`
SILENTLY, because the substitution ask only fires when a same-family stand-in is in
stock. Fix shipped: the status gate block carries `authority`; when placement is
armed and a shortage has no substitute ask, the agent SAYS what it needs in chat
(once per distinct missing set) and confirms receipt the moment the block lands in
its inventory. Creative-mode self-provisioning (`/give`) was considered and
REJECTED — it would void the D5 §4 scarcity pin and the D7 sufficiency/gather
layer; the dated note lives in D5 §4. Runbook step 6 below becomes: hand it the
build's block, or just wait for it to ask.

**Addendum 2 (2026-07-13 evening, session 165730 — 510 reads, the closest miss
yet):** conf peaked **0.916**, K reached 3, materials FEASIBLE on ~120 reads (the
shortage request + hand-over worked), PLACE_LOW_RISK was the candidate on 9 reads —
and STILL zero commits, because no candidate survived M=3 consecutive reads: the
human hovered near the agent/target all session (120 YIELD + 189 human-active
reads), and every approach drops the lattice instantly while re-earning takes 3
clean seconds. Two fixes shipped, one lesson stands:
- **Receiving posture** (agent.js): while a shortage request is outstanding, the
  agent walks to nearby dropped items and stands still for an approaching human
  instead of backing away — the band retreat had made hand-overs physically
  impossible (it fled every delivery). Workspace rule still wins; posture ends
  when the shortage clears. Dated note in D5 §4.
- **NOT a heads problem — no cascade**: conf 0.916 live is far past anything
  validation ever showed; retraining now would also repeat the rejected 07-10
  cascade's mistake (readiness still shows 0 new agreed captures).
- **The lesson for the demo**: declare the target in the tool (the declared route
  makes PLACE the candidate on every safe K≥1 read instead of only conf≥0.4 ones)
  and give the agent a real window — walk 4+ blocks away and stand still for ~5
  seconds. Shepherding it resets authority every time.
  *(The declare half of this lesson is WITHDRAWN by Addendum 4 below — the window
  half stands.)*

**Addendum 4 (2026-07-13, late — user decision: the declared-place route is
RETIRED for live sessions).** "No declaring in live session": live behavior must
come from inference alone; declarations are post-session labels. The user caught a
real drift — the original D7 §2 posture was tool-side/before-or-after-play with the
honest note that it is "not available live mid-session," and both the gather
amendment and yesterday's placement route had bent that toward live inputs. As
built now: nothing arms `declared_place_enabled` (run_live `--place` = confidence
route only, banner and docs updated); a mid-session declaration entered by mistake
changes NOTHING (regression-tested); D7's consumer table strikes "placement
authority" with the dated reversal; GATHER's declared branch is dormant under this
rule (per D7 F2, gathering waits on heads v2 / a θ_gather freeze). Field basis that
confidence-only suffices: 013710's 4 commits (conf 0.43–0.50) and 165730's 0.916
peak. Runbook step 5 amended.

**Still open before the first placement session:**
- [x] **P-1 — auditor B0 cross-check (review F4)** *(DONE 2026-07-13 night — see the debts-batch entry at top)*: committed directives vs the
  session's actual MICA_AI place events, both directions. Required before any
  UNsupervised placement session; the first supervised one can precede it.
- [x] **P-2 — watcher vs --place (review F8)** *(DONE 2026-07-13 night — MICA_PLACE=1 passthrough)*: `lan_autostart.js` spawns run_live
  WITHOUT `--place` and owns the single-consumer socket. For the supervised session:
  stop the watcher, run `python scripts/run_live.py --heads v1 --place` by hand, and
  confirm the "PLACEMENT ENABLED" banner. A `MICA_PLACE=1` passthrough is the later
  unattended option (deliberately not wired yet).
- [x] **P-3 — gather walk timeout** *(DONE 2026-07-13 night — shared gotoWithTimeout)*: the gather errand shares F3's unguarded goto;
  give it the same 20 s race when gather next gets touched.

**PLACEMENT RUNBOOK (supersedes the 07-06 demo runbook's placement half):** 1) stop
the rig watcher; 2) launch the game, singleplayer + Open to LAN; 3) `python
scripts/run_live.py --heads v1 --place` (sight the banner); 4) `node
capture/mineflayer-bot/agent.js`; 5) do NOT declare anything during the session
(Addendum 4 — declarations are post-session labels; placement is licensed by the
inferred conf ≥ θ_place alone); 6) hand MICA_AI a stack of the build's block when
it asks — walk right up to it, it holds still while a request is open; 7) build, then
step ≥4 blocks away and idle. Expect SUGGEST/PREVIEW first (M=3 hysteresis), then ONE
block per read while the window stays safe; walking toward the agent or its target
must drop it to YIELD the same read; "stop" halts everything. Afterward:
`audit_gate_trace.py` on the session (PASS required), B0 events show the placements
as MICA_AI, D1/D2 evidence excludes them (A7), materials report shows
`committed_places` > 0.

# 🔬 D5 GATE VALIDATED END-TO-END + DATA LIFECYCLE FIXED + STORAGE MADE AUDITABLE (2026-07-09)

The D5-gate confirmation pass the user asked for, plus the after-session data audit.
**The gate's decision logic is sound**: 30/30 gate tests green; a new independent auditor
(`scripts/audit_gate_trace.py`) replays hysteresis, the K_commit staircase, vetoes, and the
D5 §7 pass criteria over every banked trace — 13 files, ~13,000 reads, **all pass** (zero
placement states under the demo pin, zero irreversible/under-threshold/untraceable commits;
the pooled counterfactual's 13 committed actions all check out). Fresh `--gate` replay of
033350 under the current freeze: clean.

**Real defects found and fixed (all in the data lifecycle, not the gate math):**
- [x] **Run-mixing on re-attach (the audit's one hard FAIL).** run_live opened the belief/
  fused sinks `"w"` but the gate trace appended — a second attach truncated the belief log
  while stacking trace rows, so run 1's gate rows pointed at belief snapshots that no longer
  existed (broke §7 replayability in 165142 + 212650). Fixed at the source: run_live banks a
  previous run's 9 outputs into `previous_runs/run-NN/` before opening its own
  (`_bank_previous_run`), and `LiveGateRunner` now opens its trace `"w"`. The two damaged
  historical traces were split by run (`organize_raw.py`); every trace now audits clean.
- [x] **Agent scans were session-blind.** `agent-*.scan.<epoch>.jsonl` piled up at the raw
  root (23 files) with wallclock-guess pairing. Now the agent tags every sweep with the
  session id from live_status and rotates by session name; `run_agent_scan` pairs exactly by
  name before falling back to wallclock; `organize_raw.py --apply` adopted all 24 historical
  scans into their session folders (wallclock-matched, move log in organize_log.jsonl);
  after_game adopts scans automatically right after relocation.
- [x] **after_session_tool audit (chain: GUI → after_game → labels.json → matcher → global
  report → session_report).** Pair-level linkage is verified-by-construction (fuse_dicts
  refuses mismatched joins) — but: concurrent label jobs raced on labels.json and the global
  matcher outputs (now: one job at a time, GUI says so); a failed/skipped matcher recorded a
  fake or limbo state (now: exit code checked, session stays awaiting_label with the reason);
  a provisional manifest was processed anyway (now: refused with "quit the session first");
  labels.json / session_report.json / source_b_report.json / skipped.json were non-atomic
  writes (now: write-then-rename); builder subtypes normalize to lowercase ("House" vs
  "house" polluted the distribution).
- [x] **Storage made auditable.** `scripts/audit_sessions.py`: one table over all 47
  sessions — filing (id↔folder↔date↔manifest), stage completeness, per-session vs global
  report coherence, and trainability by the cascade's own rule (7 READY = readiness()'s 7
  agreed, cross-checked). Flags unrelocated sessions/unadopted scans; `--json` banks the
  full audit.
- [~] **Threshold provenance drift, surfaced for the vault (user decision).** Three records
  disagree on the FSM freeze: the D5 note §10 says θ_suggest 0.277 (first freeze), this
  file's cascade entry says 0.38/0.48, and `models/gate_v1.json` (regenerated by the Jul-8
  01:10 cascade via run_gate.py's pinned 80th-percentile rule) says **0.3/0.4** — the live
  traces confirm 0.38 through 07-07 and 0.3 on 07-08. The rule is being followed; the vault
  note's literal value is stale. The auditor judges each trace by its own recorded config
  and warns on drift. ► USER: amend D5 §10's resolved note to state the value follows the
  freeze rule (or pin one value and stop re-freezing per cascade).

**Behavior read-out (live traces, 1,284 reads):** SUGGEST fired only 11× (all in 215002) —
expected: those sessions ran under θ 0.38 while conf p90 sat 0.2–0.57; under the current
0.3 freeze the next live session should suggest materially more. Yield 10–43% (veto works;
counterfactual holdout 86% with walking builders). Hysteresis holds 0–18%/session, candidate
flicker ≤0.3/read — the damping is doing its job. decode_ms p50 ~350–400. Degraded reads
concentrated in the truncated-run session (fixed class). **Data-side next steps (retrain
scope, thresholds untouched):** label the 4 awaiting sessions (002202, 004538, 190631,
033350) in the after-session tool, review the 6 contested/discarded in History, then
`run_cascade_a.py` — readiness currently shows 0 new agreed since the 07-08 cascade.

Suite 302 green (+6 organize_raw) + 11 tool tests; `node --check agent.js` ok.

---

# 🧊 ADAPTIVE REGION (MOD 0.0.7) + COVERAGE PATROL + THE LEGIBLE VOXEL VIEWS (2026-07-07, small hours)

User report: "scan region too rigid; point cloud misaligned; no ground/axes; expected a
3D voxel scan." The data separated real gaps from misconceptions — **frames were provably
consistent all along** (103/104 cell-exact scan∩world intersection, Uni3D cosine 0.945;
one block-integer world frame everywhere, gravity axis = y by the probe-pinned
convention); ground-vs-structure was already split in the EVIDENCE (built = changed ∧
non-air). The real gaps, all fixed:

- [x] **Mod 0.0.7 — growth also fires off the EVENTS.** 0.0.6 growth was position-only and
  quiet-gated; bridging outward outran it (185218: 47/172 escapes → structure quarantine).
  Now a block change near/past a face grows the box IMMEDIATELY (no quiet, no cooldown);
  position path retuned (margin 12, cooldown 20t, quiet 1t). Grow-only/step/cap/opt-out
  unchanged — **shrink/move stays refused by design** (re-framing would silently change
  built-vs-terrain; the D2 note carries the dated 0.0.7 amendment).
  **Jar rebuilt out-of-tree ✓ (carries 0.0.7 + event-path growth — verified inside the
  jar). ► USER STEP: install `capture/fabric-mod/build/libs/mica-b0-capture-1.4.0.jar`
  into the launcher's mods folder before the next session; the first manifest must say
  `fabric-b0-0.0.7`, and bridging outward fast should log "region grew (build reached a
  face)" with zero escapes.**
- [x] **Agent coverage patrol** (user-approved walk+sweep): when the human is idle or
  beyond the follow band, MICA_AI walks to a vantage ~7 blocks off the UNSEEN side of the
  build (unseen = built_cells − scanSeen) and sweeps its gaze; workspace veto, personal
  band, and gate YIELD all outrank it; aborts the moment the human acts. The scan sensor
  itself is untouched — patrol only moves the body. `lastAction: "patrol: N unseen cells"`.
- [x] **ONE fixed display frame, pinned** (user requirement): every 3D view (FlowViz scan,
  FlowViz point cloud, clouds.png) shows x,y = the horizontal world axes, z = height above
  the DETECTED GROUND (min built y — the D2 templates' own rule). Display-only projection;
  the evidence path (Uni3D's probe-pinned input frame) is untouched; no per-view axis
  flips anywhere.
- [x] **Structurally legible voxel renderer** (FlowViz, shared by scan + point cloud):
  perspective orbit camera (drag), lambert-lit cube faces (ground reads horizontal, walls
  vertical, roofs cohere), DPR anti-aliasing, semantic colors (terrain = desaturated
  earth; structure by material family — block NAMES now ride the scan feed), boundary-edge
  strokes where surfaces break, ground grid + region wireframe + labeled x/y/z gizmo.
  **No interpolation anywhere — only observed cells, ever** (invented voxels = fabricated
  evidence; Uni3D's input was never gappy anyway — surface sampling is dense).
  *(Amended same night, user preference: the scan panel's DEFAULT is the agent's
  FIRST-PERSON view — the same voxel renderer driven by the live eye pose at game FOV —
  with the free 3D orbit behind the toggle; the old flat square painter is deleted. When
  the agent faces away from everything scanned, the panel says so instead of blanking.)*
- [x] **clouds.png rebuilt** in the same fixed frame with lit mpl voxels + edges, both
  subplots identical orientation — exact (104) vs scanned (103) now compare face-to-face
  (the alignment proof, visible). Also fixed: the report's scan-file pairing now uses the
  WALLCLOCK rule (newest-by-mtime once paired the NEXT session's world → "0 scanned").
- Suites: MICA 296 green, FlowViz 29 green; node --check clean; renderer verified in a
  live preview (orbit + toggle exercised).

# 📺 FLOWVIZ SYNCED TO THE DESIGN + EAGER MODEL PRE-WARM (2026-07-06, late night)

- [x] **FlowViz caught up with the system design** (its repo, `D:\2026projects\mica-flowviz`):
  the dashboard predated the whole D5 layer. Now: a **B6 · gate node** ends the pipeline
  map (live state under it); a **D5 gate panel** (state chip in lattice colors, reason,
  discounted conf + p*, K_commit, proposal, target cell); a **Materials panel** (agent
  inventory + grants from the agent status file, shortages + the pending ask from the
  gate); stale badges fixed ("belief in progress — provisional" → "v1 heads · gate:
  advisory"); `gate`/`p_z1`/`entropy`/`current_behavior` now ride the live_models message.
  **Picker bug fixed**: session discovery used a fixed derived-suffix list, so it picked
  `<sid>.gate_trace.jsonl` as "the session" — replaced with the one-dot rule
  (`<sid>.jsonl` has exactly one dot), which survives future artifacts. Verified in a
  live preview against 185218's real data (gate OBSERVE, missing 1× spruce_planks
  rendered). FlowViz tests 29 green.
- [x] **Models now pre-warm eagerly (the fix for "models don't work")**: the lazy
  manifest-triggered start left a mid-game watcher start attaching with COLD models —
  185218 passed its first 14 scored records unenriched. `MICA_PREWARM=eager` is the
  default (run_live + both GPU models load at watcher start and stay warm);
  `MICA_PREWARM=lazy` restores GPU-free idling. **HF_HUB_OFFLINE is now process-wide**
  in the watcher (children inherit — the chain's run_d1 load once pinged the Hub
  mid-chain) and defaulted in after_game.py for manual runs.
- [x] **185218 post-mortem (the step_fail + "models don't work" session)**: the chain's
  evidence step failed HONESTLY — the session is **structure-quarantined on the disk
  copy: 47/172 block events escaped the capture region** (the known region-anchor
  failure: quit the GAME fully between worlds so the region re-anchors at the new build
  site). Evidence regenerated for inspection, report at capture/reports/185218, but the
  session is not labelable/proof-grade by design. run_live + the agent both shout region
  escapes live — watch for the in-game warning next time.
- [~] **Placement still config-disabled — BY PINNED DESIGN (D5 §9)**, not a bug: the
  first live demo is observe/suggest/preview only until the A7 actor filter is verified
  in-game with the agent actually placing (a supervised session), and the execution path
  (gate → mineflayer placement) is deliberately unbuilt until the user lifts the pin.
  Prereqs already met: v1 heads calibrated + thresholds frozen ✓, counterfactual
  PLACE_LOW_RISK commits ALL PASS ✓, materials constraint ready ✓. Recommended before
  enabling: gate-side agent-position proximity (review F4).

# 🤖 RIG AUTOMATION — LAUNCH-TO-REPORT, EVENT-DRIVEN, DEDUPED, LOGGED (2026-07-06, night)

`lan_autostart.js` is now the full automation watcher; `start-rig.bat` still starts it
manually, and `scripts/install_rig_task.ps1` (optional, validated dry-run, NOT registered)
makes it a logon task for fully-hands-off operation.

**Event flow:**
1. **Minecraft launched** = a new `fabric-*.manifest.json` appears in a capture root (the
   mod writes it at game launch — fires only for the modded instance). fs.watch + 5 s poll
   fallback. Only THEN does run_live start → **VRAM is free while no game is up** (before
   this, the rig held both GPU models 24/7). A watcher started mid-game arms via a javaw
   check instead.
2. **Open to LAN** = the existing multicast announcement → agent.js (unchanged).
3. **World closed** = run_live exits (session id parsed from its own "live session:" line)
   → the POST-SESSION CHAIN runs: `after_game --session <sid> --evidence-only` (gate +
   run_d1 --pixels + run_d2 --h3d, NO label) → `make_session_report <sid>` →
   `after_game --status` (readiness LOGGED — **the cascade retrain is never auto-fired;
   pinned user decision**). Steps are data (`POST_SESSION_STEPS`), sequential; a failure
   logs `step_fail` and stops that session's chain without killing the watcher.
4. run_live re-arms only while javaw is still alive (second world); otherwise the GPU is
   released until the next launch.

**Dedupe:** one watcher per machine (exclusive localhost lock port 45677 — a second copy
exits with a message); one run_live / one agent at a time (slot guards); one chain per
session ever (in-memory set + `chain_done` scan of the log at startup; a failed chain MAY
retry after a watcher restart, deliberately).

**Log:** `capture/raw/rig_log.jsonl` — one JSON line per event (`watcher_start`,
`minecraft_detected`, `runlive_start/attached/exit/rearm`, `lan_announced`,
`agent_start/exit`, `step_start/done/fail`, `chain_done/skipped`, `minecraft_closed`,
`error`) with ts, session, exit codes — trigger time, run status, completed steps,
failures.

**after_game additions (additive):** `--evidence-only` (the unattended heavy half; writes
`awaiting_label` + the `evidence_ready` marker into session_report.json and never clobbers
a recorded verdict); labeled runs SKIP gate/d1/d2 when the marker (or a legacy verdict —
proof of a past regen) + evidence files exist, so **your label completes in seconds, not
GPU-minutes** (`--redo-evidence` forces); `--all` now surfaces evidence-ready sessions
awaiting labels and completes them once the label lands in labels.json.

**Your flow now:** play (watcher running or logon task installed) → quit → evidence +
report generate themselves → run `after_game.py --session <sid> --goal <cat> --subtype
<style>` once, whenever — it's fast. `--status` says when the batch is cascade-worthy.

**Drills run:** manifest touch → `minecraft_detected`+`runlive_start` once, re-touch → no
duplicate; second watcher → lock exit; evidence-only on labeled 165142 → skip + verdict
preserved; stubbed labeled run → only `label_finished_builds.py` invoked. Suite 296 green.
End-to-end proof = the next real play session.

- [x] **h3d_scan JOINED THE CHAIN (2026-07-06, night — user caught the gap: "is the scan
  actually using the model?")**: the scan sensor was recording but `run_agent_scan.py --h3d`
  (Uni3D over the agent's OWN scanned cloud vs the exact build's embedding — the
  earn-its-place measurement) was never run by the automation, and it had the layout
  straggler (scan files live at the raw root; relocated sessions found none). Fixed:
  bare-id input via session_store, root-fallback scan search (session-dir scans always
  win — a test fixture must never lose to a real root scan), no-scan/D1-only = skip not
  fail, and a `scan` step in POST_SESSION_STEPS between evidence and report (so scan.png
  renders). First automated verdict, on 212650: **coverage 103/104 built cells (99%),
  cosine(h3d_scan, exact h3d) = 0.9449** — the agent's own eyes carry nearly the same
  shape signal as the server-truth cloud. Suite 296 green.
- [x] **FIELD BUG (caught live on 212650, 2026-07-06 21:34, FIXED same night): a menu exit
  is not a session end.** Exiting to the world menu closes the live socket (runlive_exit)
  but re-entering CONTINUES the same session — the chain fired mid-session, failed on the
  provisional manifest (30 s wait → gate FAIL), and its dedupe mark then SKIPPED the real
  end at 21:36. Fix: `runChain` now requires a **FINALIZED manifest** (declared_event_count
  ≥ 0 — the only true "session over" signal) before marking processed; an unfinalized
  session defers with a 30 s recheck loop (one loop per session, `chain_deferred` logged;
  after ~20 min of deferrals a crash is assumed and the chain runs anyway — `chain_forced`
  — letting the B0 gate speak). 212650's skipped chain was run manually; the watcher was
  restarted onto the fixed code.

# 🩺 NEW-WORLD SESSION DIAGNOSED — H3D WAS OFF AT THE RIG, THE PLOT COULDN'T SHOW IT ANYWAY; + THE MATERIALS CONSTRAINT SHIPPED (2026-07-06, evening)

The user's chest-house session (`fabric-20260706-165142`, 16:51) came back with "H3D flat"
and an HF Hub warning. The explicit did-it-run verdict, from the session's own artifacts:

| step | status | evidence |
|---|---|---|
| agent camera scan | **RAN** | agent-MICA_AI.scan.jsonl: 349 rows through tick 5560 |
| pixels (h2d/s_goal) | **RAN** | 573 records enriched; d1_provenance written |
| Uni3D cloud + h3d | **DID NOT RUN** | d2_provenance `h3d_source: null`; h3d null on all records |
| scan/cloud report panels | not generated | after_game/make_session_report had not run |
| belief + gate | ran | 573 corrections, 132 gate reads — but on the **v0** belief |
| agent placement | disabled BY DESIGN (§9) | **not a cause**: h3d reads HUMAN cells (A7); 128+ human events |

**Three causes, all fixed:**
- [x] **The rig never enabled h3d and never passed the v1 heads** — `lan_autostart.js`
  spawned run_live with MICA_PIXELS=1 only. Every rig session to date ran the shape channel
  dark and the gate on v0 beliefs against v1-frozen thresholds. Now the rig runs the FULL
  demo config: `--heads v1` + `MICA_H3D=1` default (user env still overrides).
- [x] **channels.png could never show h3d** — it plotted the L2 norm of a UNIT-NORMALIZED
  embedding: constant 1.0 by construction. Now plots cosine(k, k−1) drift +
  cosine(k, final) convergence. The regenerated chest-house report shows convergence
  climbing 0.05 → 1.0 as the build goes up — h3d was never "flat", the plot was.
- [x] **HF warning root-caused + silenced**: reproduced — the ping fires inside the
  VPT/MineCLIP load (the only repo HF caller, setup_uni3d, was skipped: assets exist).
  All weights are local, so the rig now sets `HF_HUB_OFFLINE=1` (verified: full GPU load
  clean under it; downloads run outside the rig).
- [x] **Session processed**: after_game (goal habitation/**cabin** — the matcher CONTESTS
  with production/crop-field 0.57, VLM sides with the builder; NOT in the agreed set),
  live-quarantined artifacts banked (late attach @3048 + 4×16-tick gaps + 2 genuinely lost
  block events → honest quarantine; disk copy gate-PASS), evidence regenerated offline:
  **1146 corrections (vs 573 live), h3d on 1086**. Report at capture/reports/165142.
- [x] **make_session_report layout stragglers fixed** (found processing it): accepts a bare
  session id via session_store; reports root = `reports/` next to the raw tree (the old
  two-dirs-up climb landed inside the date folder); scan search includes the raw root
  (agent files are root aggregates).
- [x] **Rehearsal proof of the fixed rig config** (5x wire, pixels+h3d+gate+v1+offline):
  proof-grade YES, 0 gaps, h3d on 68/148 records, d2_provenance pins uni3d-b.pt, no HF
  warning, divergences == offline verifier. Budget: gate 153 ms mean / 501 worst.

**🧱 MATERIALS CONSTRAINT SHIPPED (D5 §4 amendment, user decisions pinned: agent's own
inventory · decoder continuations only · ask-in-chat substitution):**
- `mica/gate/materials.py` — `feasibility_flags` (True = can't place; stock depletes along
  the prefix; move/look/say free; namespace-normalized), `propose_substitute` (same-family,
  in stock, never denied), `read_agent_snapshot` (newest agent status line; stale >10 s =
  None), `MaterialsAccount` (per-read maxima, not sums).
- `LiveGateRunner`: reads the snapshot per gate read; the committed prefix is CAPPED at the
  first out-of-stock placement (shortage shrinks the build); trace rows gain
  `feasible_prefix`+`missing`; the gate block gains `materials` (incl. the pending ask);
  `close()` writes `<session>.materials_report.json` (proposed/committed/remaining/missing/
  compromises). **No inventory snapshot (replays, counterfactual, agent gone) = constraint
  INACTIVE, behavior bit-unchanged.**
- `agent.js`: inventory + material_grants ride the 4 Hz status file; the ask is one
  throttled chat question ("short N ⟨block⟩ — use ⟨substitute⟩? yes/no"); yes → session
  grant, no → never re-asked, silence times out (60 s); end-of-session console one-liner.
- Placement stays §9-disabled: v1 = the ask + capped prefix + report; the hard block
  engages automatically when PLACE_LOW_RISK enables. Tests: `tests/test_gate_materials.py`
  (10; incl. grant-flips-feasibility and constraint-inactive). **Suite 296 green.**

# 🔬 D6 CLOSE-OUT — LIVE LOOP REVIEWED, FIVE MAJOR FIXES, FIRST PROOF-GRADE LIVE RUN (2026-07-06)

The "implement D6" request, honestly scoped: D6 shipped 2026-07-03 and was amended through
07-05, so this pass audited the loop as-built against the vault, closed BOTH owed reviews
(stream cores, open since 07-03; live gate wiring, owed from D5), instrumented the read
budget, and finished what §9's proof-log actually still lacked. Full review note (F1–F12,
what's solid, the checklist): vault `2026-07-06 - D6 Live Runtime Loop Review`.

**Findings fixed (all live-only; every replay path bit-unchanged, verified):**
- [x] **F1 (MAJOR)** — the gate read a frozen belief between corrections (no CTMC decay for
  elapsed time; D6 §11 Q1 has been open since 07-03). Now: `LiveGateRunner._materialized`
  advances a COPY via `predict` over (status tick − belief_snapshot_id) — drift-line rule,
  never fed back. Test: peaked belief decays; dt=0 bit-identical.
- [x] **F2 (MAJOR, demo-mitigated)** — late-attach catch-up never reached the gate: its
  reversibility read started blind to pre-attach human blocks. `_catch_up_structure` now
  returns the history and the runner ingests it (the gate = the THIRD world consumer of
  catch-up). Origin drift on late attach recorded, not fixed (demo attaches at start).
- [x] **F10 (MAJOR)** — the first gate read imported torch + loaded the decoder ON the
  pipeline thread: seconds of stall vs the 3.2 s socket buffer = a guaranteed gap burst at
  every gated session's first correction. Now `preload=True` at construction, pre-attach.
- [x] **F11 (MAJOR, rehearsal catch)** — `run_live.py --heads v1` CRASHED at startup:
  `--heads` was missing from the value-flag list, so "v1" routed into replay mode as a
  session path. The documented demo command never worked. Fixed (`_VALUE_FLAGS` +
  `_positionals()` + routing test over every documented invocation).
- [x] **F12 (MAJOR, rehearsal catch — false-quarantine source)** — snapshots were judged at
  FILE ARRIVAL; §4's race-free argument only holds while consumer lag < the 40-tick quiet
  window. Reproduced at 10x: live quarantined (8 divergences, class c) while the offline
  verifier on the SAME data reads a=3 b=1 c=0 clean. Now snapshots queue until
  `pipeline.last_tick` reaches their tick (judging later is exact; earlier was the broken
  direction); end-of-stream drains the queue. Plausible contributor to the 07-04/05 era's
  quarantined-with-zero-gaps verdicts.
- [x] **F3 (MINOR)** — degradation reads now write their trace row (D5 §7 one-row-per-read).
- [x] **F7/F8/F9 (NOTES)** — dead `use_v1` param removed; `proof_grade` persisted into
  live_run.json; test_run_live.py's duplicated test definitions (committed in 7e8c126,
  shadowing each other) deduplicated.
- [ ] **F4 (MINOR, user decision)** — D5 §10's "live gate adds the agent's own position" is
  NOT in the gate (the embodiment's follow band covers the demo). Recommend wiring agent pos
  into GateRead when PLACE_LOW_RISK is first enabled (agent status file already has the pose).
- [x] **F5 (MINOR) — RESOLVED BY MEASUREMENT 2026-07-13 (yaw accumulates; reset guard shipped instead — top entry).** — yaw compared unwrapped across ±180° (`_camera_moved`,
  `yaw_delta`): wrap-crossing windows carry ±360-corrupted yaw_delta into training pairs.
  Fix changes evidence bit-for-bit → must ride a corpus-regenerating cascade; first check
  what yaw range the mod reports.
- [ ] **F6 (MINOR, watch)** — the reorderer releases the head-of-stream tick with no hold;
  earlier late arrivals count `stale`. Revisit only if real live_run.json shows stale > 0.

**Read-budget instrumentation (new):** `StallMeter` in live_pipeline (moment / correction /
enrich) + runner meters (gate_read / snapshot_check) → worst+mean in the ~1 Hz status line
(now also shows `gate:<state>`) and a `timings` block in live_run.json. Gate trace rows carry
per-read `decode_ms`. Measured (10x rehearsal, gate live, v1 heads): moment 0.03 ms mean /
10 ms worst; correction 0.6 ms; gate read 143 ms mean / 379 ms worst — single thread holds.

**First proof-grade live run** (wire rehearsal, not yet in-game): `replay_live_feed.py
194242 --speed 10` + `run_live.py --session <staged> --heads v1` → zero gaps, zero drops,
divergences IDENTICAL to the offline verifier (a=3 b=1 c=0), B0 disk gate PASS,
`proof_grade: true`, 259 gate reads traced. Verification battery: **287 tests green**;
replay-equivalence exit 0 on 194242 (378/148/148/378) AND the treehouse
(1860/1730/1730/1860).

**The §9 row-3 honesty table** (all 17 live runs to date; none in-game proof-grade —
zero-gap runs are all false-quarantine-era, every run since the GPU heads joined has gaps):

| session | gaps | quarantined | disk gate | live clean |
|---|---|---|---|---|
| 0704-004034 / 022944 / 193059 / 232045 | 0 | yes (era bug) | pass | yes |
| 0704-104428 | 0 | yes (era bug) | FAIL | yes |
| 0704-025338 / 121023 / 205205 / 210003 | 16–58 | yes | pass | no |
| 0704-110234 / 223052 | 16–87 | yes | FAIL | no |
| 0705-002717 / 113204 / 134615 | 18–48 | no | pass | no |
| 0705-004833 / 112050 | 32–96 | yes | pass | no |
| 0706-001126 (treehouse) | 165 | yes | pass | no |

**Remaining (user-owned):** the in-game proof-grade session — the same session as the D5
demo (runbook below, updated); D6 §11 Q3 (quarantine gapped sessions outright?); F4/F5
decisions above.

# 📁 RAW FOLDER REORGANIZED BY DATE/SESSION + AFTER-GAME AUTOMATION + CASCADE-A WRAPPER (2026-07-06, user request)

`capture/raw/` was a flat pile (35 sessions' output files interleaved with ~30 global
reports). Now organized `capture/raw/<date>/<session_id>/` — one folder per game holding all
its logs; global aggregates (reports, figures, `labels.json`, `arm2_cache.jsonl`, `agent-*`,
`pre_cascade_a/`) stay at the root. **Nothing lost, reproduction intact**: after migrating all
35 sessions, `run_tracker --real --heads v1` and `run_gate` reproduce the banked cascade-A
numbers exactly (real acc 0.364; gate ALL PASS, the 13 PLACE_LOW_RISK commits, identical
blocking rates), and golden-equivalence over real captures still runs (test fixed to search
the nested tree).

**How it's built (small blast radius by design):**
- `mica/capture/session_store.py` — the one resolver for "where does session X live":
  `session_dir/session_file` return `<root>/<date>/<id>/` with a **legacy-flat fallback** so
  un-migrated + non-dated (test) sessions keep resolving. `resolve_frame` tolerates relocation
  (frame paths are baked ABSOLUTE in the raw jsonl; a moved session's go stale — resolved
  against the jsonl's current dir instead).
- Writers were already sibling-to-jsonl, and snapshots/frames move WITH the jsonl (siblinghood
  preserved), so only the fixed-root READERS changed (run_tracker/arms/gate/decoder_eval,
  training_pairs, label_finished_builds) + a recursive glob in training_pairs and the
  stream-equivalence test. `run_d1` got the relocation-tolerant frame resolver.
- `scripts/migrate_raw_layout.py` — dry-run-default, idempotent, writes `_migration_manifest.json`
  (from→to, reversible). Ran `--apply`: 35 sessions moved.
- **Future logs organize themselves**: `after_game.py` relocates each fresh capture into its
  dated dir as step 1, and all derived outputs land there as siblings — no mod change needed.

**`scripts/after_game.py` — run after each game (the manual treehouse dance, automated):**
`--goal <cat> --subtype <style> [--session <id>]` → wait for a finalized manifest → relocate →
bank `*.live-quarantined.*` if the live loop dropped socket events → B0 gate → `run_d1 --pixels`
→ `run_d2 --h3d` (a non-zero exit = structure quarantine → flag + skip label) → upsert
`labels.json` → `label_finished_builds` → write `<session>/inspect.md` + `session_report.json`.
**Stops before any retrain.** `--all` catches up a backlog (keys on "has evidence yet"; lists
sessions still needing a builder label); `--status` prints the readiness line.

**Stacking + readiness — `capture/raw/cascade_status.json`:** tracks the agreed-capture set +
pair pool at the last cascade, so `--status` aggregates across all stacked games: "since the
last cascade (2026-07-06): N new agreed captures, +M pairs — ready." Baseline stamped to the
cascade A already run (6 agreed, 9,881 pairs).

**`scripts/run_cascade_a.py` — the batch retrain, one command** (still user-invoked; the fire
decision stays manual): banks pre-cascade models, runs the pinned chain (`--unpin` corpus →
relabel → train_heads → calibration/tracker → train_arm1 → run_arms → `--unpin` decoder corpus
→ train_decoder → eval → gate), fails fast on any non-zero step, stamps the readiness baseline.
`--dry-run` lists the chain.

**Runbook:** play → `after_game.py --goal G --subtype S` (or `--all` for a backlog) → check
`inspect.md` per session → `after_game.py --status` → when it says READY, `run_cascade_a.py`.

# 🎮 D5 COMPLETE — THE LIVE GATE LOOP SHIPPED; DEMO-READY (2026-07-06, user instruction; cascade workflow untouched)

The last D5 pieces (Phase G item 4 + the §6 leftover). Change log:
- `mica/gate/live_loop.py` — `LiveGateRunner`: one gate read per ~1 Hz status tick (live
  belief → decoder proposal → frozen staircase + CommitHysteresis → FSM with
  **place_low_risk_enabled=False**, the §9 demo pin); flushes `<session>.gate_trace.jsonl`;
  returns the `gate` block for live_status.json; degradation = OBSERVE with the reason.
  Plus `replay_gate()` — the pre-demo proof driver (`run_live.py <jsonl> --gate`).
- `scripts/run_live.py` — gate ON by default when decoder+freeze exist (`--no-gate` off
  switch); **`--heads v1` opt-in** (run_tracker's rebind pattern; v0 stays default — golden
  equivalence re-verified exit 0); status writer carries the gate block.
- `mica/live_pipeline.py` — two read-only accessors (`belief`, `last_fused`); emissions
  untouched.
- `capture/mineflayer-bot/agent.js` — gate rendering subordinates the old advisory line (the
  vault-pinned handover): SUGGEST = throttled offer with the proposal summary; PREVIEW =
  gaze at target + throttled line (ghost blocks = client-mod work, recorded out of scope);
  YIELD = widened follow band for 4 s. Fallback to old advisory when no gate block.
- **Arm-0 legacy-mean neutral (§6/review F2, closed)**: `commit.legacy_gate_mean` +
  `derive_arm0_neutral` measured on validation reads at freeze time → `gate_v1.json`
  "arm0_neutral"; `arm_p_star`'s arm0 branch consumes it (uniform floor fallback).
- Tests: `tests/test_live_gate.py` (demo config blocks placement; degradation; actor-honest
  cell tracking; arm0 derivation). Suite 284 green.
- **Verified**: replay-equivalence exit 0 (v0 default untouched); treehouse gate replay
  through the LIVE runner: 1,730 reads, observe 56% / yield 44%, ZERO placement states,
  trace banked. `mica-review` pass on the live wiring: ~~owed next session~~ **done
  2026-07-06 — see D6 CLOSE-OUT above** (five MAJOR fixes landed on this wiring, incl. a
  startup crash in the `--heads v1` command below; runbook updated).

**DEMO RUNBOOK** (the human part — Phase G item 4's session, which doubles as D6 §9's
proof-grade smoke): 1) start the rig (singleplayer + Open to LAN, mod 0.0.6);
2) `python scripts/run_live.py --heads v1` (expect "D5 gate: LIVE, decoder pre-warmed
(demo config...)" — the ~1 Hz line now shows `gate:<state>` and `worst-stall`);
3) `node capture/mineflayer-bot/agent.js` — MICA_AI joins, shadows, and speaks/acts FROM
THE GATE: suggestions when conf ≥ 0.38 with nothing committed, previews (gaze + line) when
the staircase commits, stepping out when you get close to its would-be target; 4) play;
quit cleanly. **Proof-grade check afterward:** `live_run.json` must say
`"proof_grade": true` (zero gaps, no quarantine, disk gate PASS) — if false, its `timings`
block says which step ate the budget; then `run_live.py <session>.jsonl` (replay) must
exit 0. The session's `gate_trace.jsonl` is the demo's proof-log. PLACE_LOW_RISK stays
config-disabled per §9.

# 🌊 CASCADE A EXECUTED — MOTION REGENERATION + FULL RETRAIN CHAIN; FIRST PLACE_LOW_RISK COMMITS UNDER ALL SAFETY CRITERIA (2026-07-06, user instruction)

Fired on the treehouse capture per the pin. One chain, every step one command, pre-cascade
artifacts banked first (`models/pre_cascade_a/`, `capture/raw/pre_cascade_a/`): --unpin corpus
regen → relabel → train_heads → calibration + tracker reruns → train_arm1 → run_arms →
--unpin decoder corpus → train_decoder → run_decoder_eval → run_gate. Pins removed — the
no-flag reproduction rule is RESTORED (banks match current code). 272 tests green.
Exposure honesty: 001126 moved to _REAL_TRAIN_SEEN in run_arms/run_gate BEFORE any table.

**BEFORE → AFTER (static banks → motion banks + treehouse):**
- **Data**: scripted pairs 3,098 → 4,303 (labeler 1.0/1.0 both eras); real pairs 3,848 (5
  sessions) → 5,578 (6); decoder corpus 12,209 → 17,081 samples.
- **Heads**: 12-F8 mode margin **−0.084 INVERTED → +0.258 FIXED** (the motion gaze signature
  did exactly what it was built for); pooled real acc 0.400 (4/10) → 0.364 (4/11 — same
  correct calls, bigger population); held-out ECE 0.1199 (underconfident) → 0.1776
  (overconfident drift — recorded); treehouse: production-82% misread → correct from 11.3%
  (train-seen now).
- **Arms**: v0 scripted holdout 1.000/+0.016 PASS (stable); arm2 1.000/+0.039 PASS; **v1
  scripted holdout regressed 0.400 → 0.000** (the domain trade sharpened: v1 leans further
  into real evidence); real_never_seen — every arm still FAILS the pre-registered earliness
  bar (population unchanged; the provisional-negative verdict STANDS — held-out builders
  remain the bottleneck). Pivots (now motion): v1 3/5 (two near-instant: 0 and 4 steps),
  v0 1/5. Arm2 provenance: 1,815 fresh queries + 1,634 cache hits.
- **Decoder**: Stage-B holdout NLL 1.2219 → 1.1115; OQ1 PASS with a wider margin (retrofit
  gain 0.9343→0.8885; its coherence contribution 0.048→0.213 = 4.4×); coherence holdout
  0.107 → 0.213, banked transfer 0.216 → 0.347; horizon-4 τ-acc 0.849 → 0.887;
  intervenability flip 0.483 → 0.583 (truth-alignment 0.38 → 0.25, recorded).
- **Gate**: validation belief reach conf-max 0.328 → **0.557**; staircase θ₁ 0.30 → **0.50**
  (a-priori reachable, p*max 0.9963, δ̂ 0.2284 s); FSM θ_suggest 0.277 → 0.38, θ_place
  0.377 → 0.48; mean K_commit 0.68 → ~0.79; **PLACE_LOW_RISK fired for the first time — 13
  committed actions on real train-seen reads, pass criteria ALL PASS (0 under-threshold /
  irreversible / untraceable / low-confidence)**. Synthetic motion sessions now exercise the
  proximity veto for real (walking builders: decoder_holdout blocking 0.29 → 0.78).

**Honest ledger**: the headline never-seen earliness claim is unchanged (still fails, still
data-limited on BUILDERS, not sessions); v1 gave up scripted-holdout accuracy; calibration
drifted overconfident and should be watched at the next retrain. What the cascade bought:
a working mode latent, a decoder twice as coherent, a gate strong enough to place — and one
corpus story consistent with the code again.

# 🌳 SESSION 001126 (TREEHOUSE) CLOSED — SIXTH MATCHER-AGREED CAPTURE, +1,730 PAIRS; CASCADE-A TRIGGER ARMED (2026-07-06)

The user's free-build treehouse (~6.7 min, 336 events, 31 snapshots). Three-part outcome:
- **Capture: proof-grade.** Disk B0 gate PASS; manifest finalized; replay check PASS (all
  divergences in documented classes, 0 crop escapes). The LIVE loop quarantined itself
  (socket drops: 165 ticks / 11 stretches, **4 block events lost in flight**) — correct
  behavior, disk authoritative; live artifacts banked as `*.live-quarantined.*` and evidence
  REGENERATED OFFLINE (`run_d1 --pixels --overwrite`, `run_d2 --h3d --overwrite`: fidelity
  PASS, 336/336 consumed once, 96% pixel-enriched, 94% h3d).
- **Labeling: TRIPLE AGREEMENT** (second ever, after treehouse-002717): matcher habitation
  (style read: longhouse) score 0.2377 ≥ 0.2 with clean margin; VLM cross-check habitation;
  builder habitation/treehouse. **1,730 pairs written** — source_b total 6,946 → 8,676 across
  6 matcher-agreed real captures. Labeler rerun was bank-safe as predicted: scripted 30/30
  kept at accuracy 1.0/1.0 (motion changes no placements; pairs rewritten from banked
  evidence).
- **Recognition: both head configurations misread it — which is exactly its training value.**
  Live (v0, quarantined evidence): pinned at uniform (H≈2.30) the whole session. Offline v1
  on clean evidence: called PRODUCTION for 82% of corrections (p_top to 0.43), habitation
  only closing at the end (0.249 vs 0.272) — the fresh pairs teach precisely this mistake.

**Cascade-A trigger status**: the pinned trigger is "the next batch of matcher-agreed
captures." One new capture (+1,730 pairs, +45% real-pair volume) is now banked; whether it
alone constitutes the batch — firing --unpin regen → retrain heads/arm1 → rerun arms →
decoder corpus → retrain decoder → eval + gate — is the user's call.

# ⚠️ REVIEW LATER — what still needs a look

## D6. Naturalized synthetic motion — follow-ups (2026-07-05)
The scripted builder now walks, aims before acting, lingers on fresh blocks, stares at broken
spots, sweeps the structure during pauses, and glances at a virtual companion
(`mica/capture/motion.py`, opt-in via `ScriptedBuild.motion`; `scripted_goals` sets a per-mode
gaze style). Knock-ons to revisit:
- [ ] **N-1 — `DELTA_COMP_WINDOW` counts corrections, not events.** With motion on, one placement
  carries ~4-5 corrections (walk/aim/linger segments), so the delta_comp baseline shrinks from
  ~5 events ago to ~1 — the `heads_v0` progress bump weakens on regenerated corpora. Decide:
  baseline against event-carrying corrections only, or retune `_PROGRESS_GAIN`.
- [ ] **N-2 — heads tuned on the robotic corpus.** `recent_actions` now interleaves
  navigate/inspect/idle (the placing-streak gate drops mid-burst) and `focus.dwell_ticks` is
  nonzero on build corrections (the intended dwell→INSPECT signal). Re-check `_STREAK_WEIGHT` /
  `_DWELL_WEIGHT` / `_PROGRESS_GAIN` against the regenerated corpus.
- [x] **N-3 — RESOLVED BY MEASUREMENT 2026-07-13 (premise corrected — see the debts-batch entry at top).** Raw `now.yaw - before.yaw` misreads a
  ±180° wrap as a huge one-tick INSPECT spike on real captures. The choreographer sidesteps it
  (it emits continuous unwrapped yaw), but the mod records raw yRot — fix with a wrapped-difference
  helper.
- [~] **N-4 — shortcut pacing outruns vanilla sprint.** Scattered shortcut plans place blocks
  ~9 cells apart every 3-4 ticks; no walk covers that at 0.28 blocks/tick. Reach stays the hard
  invariant (the builder always arrives before acting), so hasty sessions carry a declared
  scramble allowance (`MotionStyle.dash_cap` 0.75 vs. sprint) — deliberate sessions stay fully
  vanilla (measured worst step 0.246). Deferred-by-design: fixing it for real would mean slowing
  the frozen placement schedules, which would change every seeded corpus.

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
- [x] **Exposure CLOSED (same day)**: cause confirmed — the parallel agent-movement chat ran
  `git stash -u` ("epitaxy: pre-switch from main"), which also explained a second full working-
  tree revert hours later. Stash popped, 173 green re-verified, and per user instruction the
  checkpoint was **committed and pushed (7e8c126, 101 files)**. Excluded from git with new
  ignore rules: capture/youtube (11 GB mined clips), researchpapers/ (264 MB PDFs),
  capture/rehearsal/, capture/reports/, .idea/. STANDING RULE: parallel chats work in their
  own branch/worktree, never main's working tree.

# ⚠️ VERIFIED BROKEN — NO-FLAG CORPUS REGENERATION NO LONGER REPRODUCES THE BANKS (2026-07-06, verification on user request; finding recorded, NO regeneration performed)

Measured in memory, banks untouched. `make_scripted_corpus.py` no-flag vs the banked
`capture/scripted/`: **30/30 sessions mismatch** (e.g. cabin-7000: 292 fresh B1 records vs 217
banked; longhouse-7004: 381 vs 261). Decoder corpus same: fountain-11023 regenerates to 80
corrections vs 64 banked (+25%).

**Root cause**: the merged motion PR (aeee7c7, "choreographed builder motion") keeps the
engine opt-in (`ScriptedBuild.motion=None` = byte-identical static) but `build_from_plan` now
opts EVERY scripted plan in — so the default generation path changed under the pinned rule
"no-flag reproduces the banked corpus." The banks on this machine were never regenerated
(evidence mtimes 07-02/07-04 predate the merge): both `capture/scripted/` AND
`capture/decoder/` are STATIC-generator data. (Archaeology note: static evidence SHOWS short
inspect runs and 1-tick focus dwells — the pitch-snap artifact around each placement — so
those are NOT motion signatures; absent walking / all-zero pos_delta is the reliable tell.)

**What this means**: every Phase E–G artifact (source_b pairs, heads_v1, arm1, arms_report,
decoder corpus, decoder_v1, gate traces) is mutually consistent on static data, and the banks
still reproduce under pre-merge code (the 7e8c126 era generator). Only the forward property is
broken: today's no-flag command generates richer motion data that matches no bank.

**DECIDED (2026-07-06, user): B now, A scheduled.** Executed same day:
- **Load-bearing pins**: `capture/scripted/PROVENANCE_PIN.json` + `capture/decoder/
  PROVENANCE_PIN.json` record the static-generator provenance, the verified mismatch, and
  every model trained on each bank.
- **Refuse-loudly guards**: `make_scripted_corpus.py` and `make_decoder_corpus.py` now exit 1
  while a pin stands, naming what a regeneration would orphan; `--unpin` is the explicit
  escape and deletes the pin (cascade-A entry point). 3 new tests
  (`tests/test_provenance_pin.py`); the pins can't rot silently.
- **CASCADE A scheduled**, bundled with the retrain the next batch of matcher-agreed captures
  forces anyway (one cascade pays for both improvements): --unpin regenerate perception
  corpus → relabel (label_finished_builds) → train_heads → train_arm1 → run_arms →
  --unpin make_decoder_corpus (fresh arm2 LLM queries, hours) → train_decoder →
  run_decoder_eval → run_gate — every step one command; run as its own recorded phase step
  with before/after tables so the motion effect stays attributable.
- **Standing caveat until cascade A**: `run_arms.py` (pivot sessions) and `run_gate.py`
  (decoder-holdout branch) regenerate sessions IN MEMORY — on current code those come out
  motion-choreographed and inconsistent with the static banks. Don't rerun them expecting
  banked-identical results; the pins guard disk regeneration only.

# 🔁 HYSTERESIS CORRECTED — D5 §3 PSEUDOCODE REWRITTEN, CODE ALIGNED, GATE REGENERATED (2026-07-06, user instruction; review P2-F2 closed, scope upgraded)

The directed re-review found THREE defects where P2-F2 had recorded one: (1) the note's bare
streak counter let MIXED-candidate flicker accumulate into authority (A,B,A,B reached M);
(2) the hard veto was hysteresis-delayed — contradicting §4's "YIELD fires whenever the human
is proximal" — and the code shared this (symmetric hold); (3) a read with conf below θ_place
could keep PLACE_LOW_RISK standing for M−1 reads, committing under a failed check (latent in
code; never fired only because PLACE never fired). One rule fixes all three, and it is the
project's existing doctrine (D4 commit gate: "caution never waits, only authority does"):
**downward lattice moves are immediate; upward moves need the SAME candidate on M consecutive
reads.**

Changed: D5 §3 pseudocode rewritten (lattice-ranked ApplyHysteresis + pending variable; dated
old-vs-new amendment in the note); D4's hysteresis bullet amended to match; `LatticeHysteresis`
replaces the symmetric `StreakHysteresis` (`mica/gate/fsm.py`; the old class removed from
`commit.py`); 3 new tests (immediate veto, immediate confidence-dip drop, flicker never
accumulates) — suite 257 green. Gate regenerated: **pass criteria ALL PASS**, blocking rates
per group UNCHANGED (0.286/0.217/0.136/0.099 — the veto fires the same reads; only state
persistence moved), YIELD shares up (validation 0.03→0.22: leaving YIELD is now an upward
move, so the agent re-approaches only after M clean reads — boundary jitter gone), scattered
PREVIEW holds collapsed (train-seen 0.03→0.00: mixed flicker no longer accumulates), mean
K_commit unchanged (staircase untouched), committed actions still 0.

# ✅ SEMANTICS RESOLVED — D5'S DISCOUNTED CONF IS THE CANONICAL OBSERVE/SUGGEST RULE (2026-07-06, user decision; review P2-F1 closed)

The user picked D5 §3's form: SUGGEST iff conf = p*·(1−P_z1) ≥ θ_suggest. The FSM already
implemented it and its frozen 0.277 was swept under it, so **gate behavior is unchanged** —
verified by regenerating both artifact sets and comparing: decoder_report.json numbers
identical (OQ1 1.0479/1.0574, coherence 0.1073/0.2163, parity, intervenability), gate rerun
identical (same state shares, blocking rates, mean K_commit per group, ALL PASS). What
changed is the bookkeeping, so θ_suggest now has exactly ONE home:
- `mica/gate/commit.py`: `GateThresholds.theta_suggest` + its validator REMOVED — the
  staircase owns K_commit only; the docstring points to fsm.py for state selection.
- `scripts/run_decoder_eval.py`: the θ_suggest formula pin dropped from the sweep; the
  raw-semantics `"suggest"` field dropped from decoder_commit_trace.jsonl rows —
  gate_trace.jsonl (FSM) is the sole SUGGEST record.
- `models/gate_v1.json` regenerated: "thresholds" without theta_suggest; "fsm" semantics
  note records the resolution. `scripts/run_gate.py` note text updated (no logic change).
- Tests: θ_suggest removed from staircase constructions; the old cross-quantity validator
  test deleted (FsmConfig's θ_place > θ_suggest is the surviving constraint, already
  tested). Suite 256 → 255 green, no failures.
- Vault: D4's commit-gate bullet amended (dated old-vs-new; the old θ_suggest < θ(1)
  inequality superseded STRUCTURALLY — the SUGGEST branch only runs when the staircase
  committed nothing); D5 §10 conflict block closed; review P2-F1 marked resolved.

# 🛡️ D5 EXECUTED — PHASE G PART 2: FSM + COUNTERFACTUAL GATE, ALL SAFETY CRITERIA PASS OVER 4,412 READS (2026-07-06, small hours)

Opened by fixing the one outstanding defect first (review F6, root-caused before touching
anything): `tokenizer.decode_chunk`'s "no actions" rejection traveled in the SAME shape as
garbage rejections, so "the decoder says the build is done" was indistinguishable from
malformed output. Smallest fix: `model.propose` finalization extracted into a pure
`_finalize_chunk` helper returning a named `NOTHING_TO_DO` signal for the immediate-close
case; B5 untouched (still no empty chunks); eval counts it separately (measured 0 on the
banked eval — the current model over-proposes at build end instead, a data note, 250/12k
empty-target training samples). Affected files: `mica/decoder/model.py`,
`scripts/run_decoder_eval.py` (counter only — banked report numbers unchanged, provably),
`tests/test_decoder_model.py` (three-outcome test).

**Part 2 change log:**
- `mica/gate/fsm.py` — D5 §3 made executable: hard proximity veto first (Euclidean 4-block
  radius around the human OR target within 2 blocks of their focus — the §10 Q1 combination
  pin, amended into D5), conf = p*·(1−P_z1) as the soft mode discount, safety lattice
  OBSERVE/SUGGEST/PREVIEW/PLACE_LOW_RISK/YIELD, StreakHysteresis(M), EXECUTE_CHUNK
  config-disabled (proven unreachable by test), PLACE_LOW_RISK config flag (counterfactual
  on / first live demo off per D5 §9).
- `scripts/run_gate.py` — the counterfactual run (D5 §8): per correction replays the v1
  belief, asks the decoder, runs the staircase, lets the FSM choose; real train-seen thinned
  ×5 (cost, recorded), every other group full-coverage. Player position + human standing
  cells joined from the RAW session recordings (banked evidence doesn't carry them).
- `tests/test_gate_fsm.py` — 11 tests (veto precedence, lattice, discount, hysteresis hold,
  config flags, EXECUTE_CHUNK unreachability, snapshot fields). Suite 244 → 256 green.
- **FSM freeze** (pre-registered rules → `models/gate_v1.json` "fsm" section): θ_suggest
  0.277 (80th percentile of validation discounted conf), θ_place 0.377 (θ_suggest + 0.10),
  M 3, radii 4.0/2.0. **Surfaced, not silently picked**: D4 words the OBSERVE/SUGGEST split
  as raw p*, D5 §3 as discounted conf — the FSM follows D5 (owner note); both freezes are
  labeled with their semantics; the conflict is recorded in D5 §10 for reconciliation.
- Vault: D5 §10 amendments (proximity pin + semantics conflict); review-note Part 2 addendum
  (P2-F1 conflict, P2-F2 hysteresis stricter than D5's literal pseudocode — the note's
  version would commit on a flickering candidate, P2-F3 counterfactual proximity limits).

**The trace (capture/raw/gate_trace.jsonl, 4,412 reads · gate_report.json · gate_states.png):**
- **Pass criteria ALL PASS**: 0 under-threshold commits, 0 irreversible commits, 0
  untraceable commits, 0 low-confidence placements. Every read carries its
  belief_snapshot_id and its binding reason.
- Chosen states: OBSERVE 0.78–0.97 everywhere; YIELD (proximity veto) 0.03–0.22 = the
  blocking rate; PREVIEW 3% on real train-seen (mean K_commit 0.61 there); PLACE_LOW_RISK
  fired ZERO times — θ_place 0.377 sits above what the underconfident calibrated belief
  reaches (validation conf max 0.328), so placement authority is measured dormant, exactly
  the D5 §9 first-demo posture, now as a number instead of a pin.
- Candidate-vs-chosen gap = hysteresis at work: SUGGEST arises on 4–5% of real reads and
  PLACE_LOW_RISK on up to 2.7% (train-seen), but no burst survived M=3 consecutive reads.
  The gate is conservative because the belief flickers — the same data lever as everything
  else (stronger heads → steadier candidates → states unlock; every artifact regenerates by
  one command).

Still open by design: Arm-0 legacy-mean neutral calibration (awaits the review question's
answer + a legacy-gate baseline), the D4/D5 semantics reconciliation (user pick), and the
live advisory demo (needs a human at the modded client; the gate now has everything it
reads exposed via live_status.json per readiness-review F2).

# 🔧 D4 EXECUTED — PHASE G PART 1: FROM-SCRATCH MTP DECODER + COMMIT GATE SHIPPED, GATE FROZEN AGAINST THE CALIBRATED HEADS (2026-07-05, late night)

Started on user instruction the same night Phase F banked (C6's ordering satisfied). Pinned
decisions honored: from-scratch decoder (17.5M as built, FastSceneScript regime), LOCAL-ONLY
corpus v1 (AssistanceZero data = recorded follow-up; D4 note amended, dated old-vs-new). The
backbone check (Phase G item one) is resolved-by-decision — nothing LLM-sized on disk, on
purpose. `mica-review` ran BEFORE numbers (vault: `2026-07-05 - D4 Decoder and Commit Gate
Review`); 244 tests green; no stream core touched (replay-equivalence re-run: PASS, exit 0).

**New code (the change log):**
- Contracts: `mica/contracts/b4.py` (ControlContext — the arm-swap slot, intervenable),
  `b5.py` (ProposalChunk, conf_1:=1 enforced), `b6.py` (GateDecision enum + inputs_snapshot).
- Decoder: `mica/decoder/grammar.py` (move/look/place/break/say, strict JSON + robust parser),
  `tokenizer.py` (87-token language, τ=0/τ=2 agreement), `context.py` (C_η: evidence vector +
  per-arm slot fills + belief-level `intervened_belief`), `model.py` (decoder-only trunk + MTP
  retrofit: shared projection block, ONE token head, ONE agreement-trained confidence head;
  C2-masked rationale head).
- Corpus: `mica/data/decoder_corpus.py` + `scripts/make_decoder_corpus.py` → `capture/decoder/`
  (12,209 samples, 125 sessions: 120 fresh seed-11 + 5 banked transfer; helper traces from
  plans with mistake/redundant-click skip sets; per-sample hygiene REFUSALS; arm2 slots cached
  over a pinned coverage subset).
- Training: `scripts/train_decoder.py` (Stage A NTP → Stage B MTP; λ_h .8, λ_c .5, n≤8; equal
  arm-mask ratios C5 + 15% counterfactual goal-slot sharpening) → `models/decoder_v1.pt/.json`
  (+ stage-A checkpoint kept for OQ1).
- Gate: `mica/gate/reversibility.py` (D5 §4 static table) + `mica/gate/commit.py` (K_commit
  staircase, kernel-propagated belief, δ̂ freeze, p*_max fixed point + A-PRIORI reachability
  hard-fail, +1/read and M-reads hysteresis, frozen masked-arm mappings) → `models/gate_v1.json`.
- Eval: `scripts/run_decoder_eval.py` → `capture/raw/decoder_report.json`, `decoder_proposals.
  jsonl` (955 chunks), `decoder_commit_trace.jsonl` (462 reads), 3 figures.
- Tests: 46 new (grammar round-trips, parser/decode rejections, hygiene violations, C1
  import-graph pin, C2 mask, staircase hand-checks, hysteresis, reachability, reversibility,
  arm mappings, split-leak regression).

**The numbers (all from decoder_report.json, one command):**
- **OQ1 retrofit check (gating milestone): PASS** — Stage B one-step holdout NLL 1.0479 vs
  Stage A 1.0574 (the retrofit HELPED — the FastSceneScript small-regime bet, measured);
  coherence kept (0.107 vs 0.097); horizon-4 τ-accuracy 0.849 vs token chance 0.0115.
- **C5 parity holds**: per-arm holdout NLL 1.047–1.049, proposal parse rate 1.0 in all four
  slot conditions — the decoder cannot tell arms apart beyond intent content.
- **Intervenability**: forcing the belief flips the argmax action on 48% of early ambiguous
  scenes (truth-forced alignment 38%) — the slot has causal power, but modest (see the honest
  caveat below).
- **Gate frozen (calibrated v1 heads per the D5 §9 pin)**: δ̂ 0.29 s, p*_max 0.9954, a-priori
  reachability PASS at every j≤8; c_min 0.95 (F1 sweep on 6,504 conf values), θ₁ 0.30,
  slope 0.04, θ_suggest 0.22, M 3. K_commit over 462 counterfactual reads: mean 0.68,
  distribution 0:268 / 1:105 / 2:66 / 3:18 / 4:4 / 5:1 — an OBSERVE/SUGGEST-leaning gate,
  exactly what D5's first-demo authority pin prescribes.
- **Honest caveats, recorded as measured**: (1) the calibrated v1 belief is NEAR-FLAT on
  scripted evidence (p* ≤ 0.215; real validation peaks at 0.335) — so the corpus's live arm3
  slots carried little goal signal and the counterfactual share did the conditioning work;
  the original θ grid (≥0.35) was structurally dead and the sweep's hard-fail caught it
  (review F8). (2) Absolute plan-coherence is weak (holdout 0.107 / banked transfer 0.216):
  naming EXACT remaining cells from a 131-float context is information-limited by design.
  Both share the standing lever: more matcher-agreed captures → stronger heads → regenerate
  corpus (one command) → retrain (one command).
- **Two bugs the harness caught before they could lie**: the banked transfer group leaked
  into training via a split-routing bug (caught by the eval's own count printout; fixed,
  RETRAINED, regression-tested — review F7); the token budget rejected 91% of proposals as
  malformed instead of trimming to the complete-action prefix (fixed — review F9).

Part 2 (next): D5 FSM + counterfactual `gate_trace.jsonl` wrapping this gate; Arm-0
legacy-mean neutral calibration; then the live advisory demo path.

# 📊 PHASE F EXECUTED — FOUR-ARM TABLE BANKED; THE PRE-REGISTERED EARLINESS BAR IS NOT MET ON NEVER-SEEN FREE BUILDS, BY ANY ARM (2026-07-05, night)

The user chose Phase F before D4 (per the pinned plan order + D4-C6's artifact ordering; D4
decisions recorded for Phase G: from-scratch 15–30M decoder, local-only corpus). Built same
evening: the 06-note metric suite (`mica/validation/intent_metrics.py`), Arm 1 implicit
classifier (`arm1.py` + trainer), Arm 2 LLM reader (`arm2.py`, local qwen2.5vl:7b, D4-S1
confidence mapping, model-bound disk cache), stitched pivot sessions (`pivot_builds.py`), and
the unified runner (`run_arms.py` — identical fused records to every arm, exposure-honest
groups). **`mica-review` on the harness BEFORE quoting numbers**: 1 MAJOR (headline earliness
was in step units; 06 defines build-progress units — both now reported, progress quoted),
2 MINOR fixed (pivot seam moved to first new-goal evidence; Arm 2 cache bound to its model),
3 NOTEs recorded → vault `2026-07-05 - Phase F Eval Harness Review`. 198 tests green.
Arm 2 provenance: 2,022 queries + 980 cache hits, 12 unusable replies.

**The table (held-out groups; sustained@build = build fraction when the call locks on):**
- `scripted_holdout` (assigned layout, 5): floor acc 1.00 @0.33 · LLM acc 1.00 @0.49
  (early-sep +0.024 PASS) · v0 tracker acc 1.00 @0.36 (+0.018 PASS) · v1 acc 0.40 · Arm 1 chance.
- `real_never_seen` (free builds, 5): **every arm FAILS the pre-registered criterion** (mean
  early separation > 0 before 40% of the build). Best final accuracy 0.20 (LLM, v0, v1 tied);
  the structure floor drops to 0.00 with early-sep −0.23 (confidently wrong templates); v1 has
  the least-bad earliness (0.881) and ECE 0.31.
- Pivots (5 stitched, recovery = steps from first new-goal evidence): **v1 recovers 4/5**
  (38–168 steps), v0 2/5, memoryless floor 3/5 (fast when the new template locks); Arm 1's
  "instant" recoveries are near-uniform wobble, not tracking (single-goal metrics at chance).
- Domain split worth remembering: v0 owns scripted symbolic (0.92 acc train-seen), v1 owns
  real pixel-carrying sessions (0.75 train-seen) — each head works where its likelihoods were
  shaped; the LLM reads clean scripted scenes well (0.76–1.00) and real free builds poorly (0.2).

**The honest verdict, exactly as pre-registration demands**: on the CURRENT never-seen set the
earliness claim fails for every mechanism — and the falsification clause says "across held-out
builders" while this corpus has ONE builder and 5 never-seen sessions with lopsided categories.
The result is a real, banked baseline table produced by a review-hardened harness, and it is
PROVISIONAL: the same data bottleneck named at Phase E's close (more matcher-agreed captures)
is what turns this table load-bearing. Artifacts: `capture/raw/arms_report.json` + 3 figures,
one command (`run_arms.py`), Arm 2 cache makes reruns free.

# 🚀 PHASE E EXECUTED — TRAINED HEADS v1 SHIPPED, BELIEF MOVES ON REAL PLAY FOR THE FIRST TIME (2026-07-05, evening)

All four gate checkboxes ran the same day the gate opened. Every artifact regenerates from one
command (D0 standard); 184 tests green (173 + 11 new); no stream core touched — golden
equivalence untouched by construction.

**1. Source B labeling pass** (`label_finished_builds.py`): scripted 30/30 kept, label accuracy
1.0/1.0; real: 5 kept-and-agreed sessions wrote pairs (crop field 2359, treehouse-002717 724,
tower 353, road 240, fountain 172 = **3,848 real pairs** + 3,098 scripted = 6,946), 4 contested
wrote none (contest rule), 3 discarded (225251 near-tie margin 0.002 — the fit×comp key reads
it fountain-vs-habitation; 113204 near-tie; 112050 below threshold). Honest note: the LOCAL VLM
cross-check contested 4/6 kept verdicts (it reads tiny builds as "habitation") — recorded in
the report; it never blocks pairs when builder+matcher agree, but it is currently too noisy to
arbitrate. → `capture/scripted/source_b_report.json`.

**2. B1-F3 stride probe** (`probe_sgoal_stride.py`, new — in-memory, banked artifacts untouched):
s_goal at stride 1/6/20 over the 5 template + 2 clean free sessions. **The channel is FLAT at
every stride** (aggregate margin +0.0046/+0.0043/+0.0043; early margins ≤0.0016; truth-first
0.346 vs 0.2 chance). Stride 1 stays the corpus default (longer spans are WORSE — the F2
multi-second hypothesis is dead); no regeneration needed. The CLIP4MC rung question is now
formally live; the trained heads (below) measure what the raw cosines are worth in the
meantime. Idle-precision frame sample banked in the report for eyeball review.
→ `capture/raw/sgoal_stride_probe.json`.

**3. Real-data reruns** (`run_tracker --real`, `probe_h3d_linear --real`, `probe_h3d_fusion
--real` — leave-one-session-out, truth = builder category, 210003 quarantined-excluded):
- **12-F3 verdict: the v0 hand-coded heads FAIL on real play** — fused final accuracy 0.100 vs
  the structure-only floor's 0.500; behavior channels HURT (−0.073 earliness, −0.100 accuracy).
  The scripted 0.416/0.933 numbers were the predicted pre-equip upper bound. This is the
  strongest possible motivation for trained heads, banked as a baseline.
- h3d linear probe on real embeddings: 0.289 overall vs 0.2 chance (early bins 0.375/0.333) —
  weak-positive, throttled by N=10 with defense/production single-session categories.
- **h3d fusion gate on real data: FAIL** (−0.100 accuracy, no earliness) — the readout weights
  stay UNSHIPPED (`train_h3d_readout.py` still refuses); root cause is the v0 belief never
  locking on at all (sustained-from ≈1.000 in every arm), not the embedding.
→ `capture/raw/belief_summary_real.json`, `h3d_linear_probe_real.json`, `h3d_fusion_probe_real.json`.

**4. Adapter v1 + heads TRAINED** (new: `mica/data/training_pairs.py` with the 09-F1 per-sample
assertion — join verified, window-strictly-pre-action, built-count leak detector, all TESTED
with constructed violations; `mica/intent/features.py` single-source featurizer;
`mica/intent/adapter.py` torch model per the pinned architecture; `mica/intent/heads_v1.py`
numpy inference behind the EXACT v0 `likelihood()` interface; `scripts/train_heads.py`).
Training: NTP only (MTP ablation deferred per arm-fairness), sessions weighted equally,
holdout = last scripted session per goal + fountain-134615; best val NLL 0.354; T_delib 1.4,
T_heur 1.0, then the pinned (ε, λg, λz) grid through the REAL filter on holdout sessions chose
ε 0.01, λg 0.02, λz 0.5 (NLPD 0.3395). Shipped `models/heads_v1.npz/.json` + training report.
**heads_v1 is OPT-IN** (`--heads v1` on run_tracker/calibration_report); v0 stays the baseline
arm and the live default.

**The result, honestly split** (`belief_summary_real_v1.json`, `calibration_report_real_v1*.json`,
per-session `*.belief_trace_v1.jsonl` + `belief_traces_v1.png`):
- Pooled 10 real sessions: fused 0.400 final accuracy (v0: 0.100), sustained-from 0.882 (v0:
  1.000); BOTH channel contributions turned positive (+0.100 accuracy each). p_top finally
  leaves the 0.20–0.26 dead band — reliability mass now sits at 0.4–0.7 confidence, and where
  v1 says ≥0.54 it is right 100% of the time (underconfident, the safe direction).
- Trained-on 4 sessions: 3/4 final-correct (road sustained from 55% progress) — the heads work
  where they have seen the category's data.
- **Never-seen 6 sessions: 1/6** — but that one is a real generalization win: the fancy
  fountain (113204, never trained, contested-by-matcher) read correctly from **38% progress**.
  Held-out ECE 0.1199 (v0's 0.0812 is cheaper-but-empty calibration: it never claims anything).
- **12-F8 partial**: the mode margin at choice points widened ~28× in magnitude but INVERTED
  (v0 +0.003 → v1 −0.084): Q_heur, trained only on shortcut sessions, over-favors placement, so
  P(z=1) inflates during building bursts. Recorded as a finding; needs more deliberate-mode
  variety, not a code fix.

**The Phase E conclusion: the machinery is DONE and the belief layer is now model-driven; the
remaining gap is DATA, exactly as D3 predicted.** 5 pairs-eligible real sessions (4 trainable)
cannot cover 5 categories × free-build variety. The lever is more template-exact captures +
free builds that the matcher can agree on — each new agreed session is ~200-2000 more pairs.
Next: more captures, then retrain (one command), then the Phase F four-arm comparison.

# 🎉 TEMPLATE GATE COMPLETE (5/5) — PHASE E DATA GATE OPEN (2026-07-05, afternoon)

Session `fabric-20260705-134615`: the exact 18-block fountain — **matcher-agrees at fit 1.00
comp 1.00**, replay a=8 b=0 c=0, 0 escapes, 31/31 events, 96% pixel-enriched, third fully-clean
live run; scan 21/21 covered at cosine 0.9465 (a fifth convergence point) before the agent was
kicked again mid-session (kicker unconfirmed; capture unaffected — the mind records regardless).

**The five template captures are banked**: production (crop field, comp 1.00) · habitation
(fit 1.00, style caveat) · infrastructure (road/flat span, category-agreed) · defense (square
tower, fit 0.89) · decorative (fountain, 1.00/1.00). Plus five labeled free builds as the
evaluation set. Phase E's standing sequence is now unblocked:
- [x] Source B labeling pass over the corpus (pairs from the template + agreed sessions)
- [x] B1-F3 early-separation probe (s_goal stride 1/6/20 on real captures)
- [x] 12-F3 fused-arm rerun + both h3d probe reruns on real data
- [x] Train A_φ + heads against the calibration harness (the λ grid + pinned architecture)
(All four executed same day — see the PHASE E EXECUTED block above.)

# ✅ THREE LOG-ONLY SESSIONS MATERIALIZED — TEMPLATE GATE AT 4/5 (2026-07-05, afternoon)

Sessions 131308/131826/132647 were built while the rig was down (the agent kick incident:
MICA_AI kicked with disconnect.genericReason from three worlds; no second bot process or code
change on this machine — source external, likely the parallel chat's game activity; our rig was
stopped to end the loop). They recorded LOG-ONLY — and the offline-first architecture proved
itself: `run_d2 --h3d` + `run_d1 --pixels` regenerated every structured field (h2d/s_goal/h3d/
structure/fused-ready evidence) deterministically from raw. All three: replay c=0, 0 escapes,
all events consumed, ~92% pixel-enriched. Only the agent scan is absent (embodied sensor — the
agent wasn't present; recorded honestly, never faked).
- [x] **131308 — infrastructure ✓ SLOT FILLED**: builder "highway road", matcher flat span
  fit 0.78 comp 0.85 (sibling infra styles; category doubly attested).
- [x] **131826 — defense ✓ SLOT FILLED, matcher-agrees**: square tower both sides, fit 0.89 —
  the second full agreement after the crop field.
- [x] **132647 — decorative contested**: builder "weird decorative building", matcher
  defense/perimeter wall (fit 0.82; 50 class-b = ~25 two-block items). No pairs; strong
  evaluation-set material.
- [ ] **TEMPLATE TALLY: production ✓, habitation ✓(caveat), infrastructure ✓, defense ✓ —
  DECORATIVE remains**: exactly the 18-block fountain (5×5 ring one high + 2-high center
  column). One small build opens Phase E's data gate.
- [x] Coordination rules re-stated after the kick incident: parallel chat's bot joins as
  MICA_AI_2 (replica support is built into the actor contract); its code work lives in its own
  branch/worktree; game time is one chat at a time.

# ✅ SESSION 113204 (FANCY FOUNTAIN) + THE TEMPLATE-EXACTNESS LESSON (2026-07-05)

- [x] Second fully-clean LIVE run (no quarantine, exit 0); replay a=106 b=0 c=0, 0 escapes;
  D1/D2 PASS (72/72 events, **97% pixel-enriched** — the pre-warm at full effect). Scan:
  86% coverage, cosine 0.8936 — the convergence curve now has four real points
  (29%→0.55, 83%→0.73, 86%→0.89, 100%→0.98), monotone throughout.
- [x] **Label: decorative/fountain (builder) — matcher contested CONFIDENTLY** (habitation/cabin,
  fit 0.83): a 44-cell fountain with raised basin walls is geometrically a roofless cabin. No
  pairs. **The lesson, now 3-of-4 attempts: template captures must be the LITERAL template
  shapes** — the matcher certifies geometry, not intent. Slots still open: infrastructure
  (EXACTLY 7×3 deck + two 7-long rails at +1), defense (3×3 ring ×6 + 3×3 cap), decorative
  (EXACTLY 5×5 ring one high + 2-high center column, 18 blocks, nothing more). Free builds
  remain valuable as the evaluation set — but they cannot fill template slots.

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
- [x] **Remaining**: ~~the in-game smoke session~~ superseded — real live sessions ran from
  07-04 on (none proof-grade yet; see the D6 CLOSE-OUT table at the top — the in-game
  proof-grade session now rides the D5 demo runbook); ~~`mica-review` pass on the stream
  cores~~ **done 2026-07-06** (vault `2026-07-06 - D6 Live Runtime Loop Review`).

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

## 2026-07-18 night -- first advisory session: rig verification findings

Session fabric-20260718-194844 (production/crop_farm per matcher, awaiting user label).
Live artifacts quarantined (correctly -- 46 events missed during a consumer gap);
offline evidence RECOVERED by manual `after_game.py --evidence-only --redo-evidence`
rerun with the GPU free: replay clean (45 snapshots, c=0), evidence_ready, awaiting_label.

What happened, in order (rig_log, UTC): watcher started 23:47:12, world opened 92 s
later -- pre-warm never finished, so run_live attached ~113 s late; it CRASHED 35 s
after attaching, within ~1 s of the session's FIRST block event (tick 1306); the
replacement attached 76 s later at tick 1951, so events 0-45 (ticks 1306-1915) were
never seen live (the mod's socket does not buffer without a consumer -- by design,
disk is the complete copy). The post-session chain then died in run_d2 AFTER its
voxel replay PASSED -- most plausibly GPU contention with the eagerly re-armed
run_live spawned 9 s before the chain's evidence step. Both tracebacks are
unrecoverable: child stdout/stderr goes to the watcher console only.

- [x] **FIXED -- acceptance signal v2** (namespace-blind block compare + causal
  voicing pairing; tests pin both). v1 could NEVER score a follow on real data
  (capture says "minecraft:x", proposals say "x"). True first-session lift is
  -0.0022 (1 control follow / 450), not 0.0. D9 section 3 amended.
- [x] **R-1 FIXED (2026-07-18 night) -- a crash is not a quarantine**: after_game's
  d2-failure branch now stamps STRUCTURE QUARANTINE only when THIS run's
  voxel_replay_report.json (mtime-checked against the run_d2 start) says
  quarantined; any other non-zero exit is a retryable chain failure that writes
  no report (same honesty rule as the run_d1 branch). Helper
  _fresh_replay_quarantine + tests/test_after_game_quarantine_verdict.py.
- [x] **R-2 FIXED (2026-07-18 night) -- GPU serialized, live play wins**: run_live
  yields the GPU only when a chain is running/queued, or when a chain is pending
  (manifest recheck) with Minecraft closed; if the game is up, the mind always
  spawns (a late chain is retryable, a missed session is not). A deferred spawn
  fires the moment the chain drains (rearmPending -> fireRearm, PREWARM-aware).
  Walked against tonight's timeline: the 00:03:33 rearm now defers (game closed,
  chain 9 s out) and the 23:51 mid-session crash still re-arms immediately.
- [x] **R-3 FIXED (2026-07-18 night) -- child output persisted**: tagPipe now also
  appends every line (incl. a torn traceback tail, flushed on child close) to
  capture/raw/child_logs/ -- runlive-<ts>.log, agent-<ts>.log,
  chain-<sid>-<step>.log; runlive_exit and step_fail rig_log rows carry the
  child_log path on failure. Verified live: a watcher smoke-start captured
  run_live's real stderr to the file. Delete old logs freely; nothing reads them.
- [x] **R-4 FIXED (2026-07-19) -- failed sessions are visible and retried**:
  (a) _surfaces_in_backlog (tested): unlabeled quarantined/gate-failed sessions
  now surface as the GUI's read-only cards -- first real snapshot revealed 14
  quarantined + 1 gate-failed historical sessions that were invisible; only a
  recorded verdict retires a session. (b) Watcher startup sweep: step_start
  without chain_done -> chain_retry (skips honestly-parked reports, gives up
  after 3 step_fails with chain_retry_exhausted); manifestFinalized now also
  reads the dated dir (a relocated session deferred as 'still open' in the first
  live test). Proven end to end: the sweep retried 194844's chain and COMPLETED
  the scan/report/readiness steps the 07-18 crash had aborted -> chain_done.
  (c) process_one refuses fast on a recorded quarantine unless --redo-evidence
  (closes R-8 too -- no more 5-min blind GPU regens on parked sessions).
- [x] **R-5 FIXED (2026-07-19) -- newest_capture is a whitelist**: fabric-*.jsonl,
  single-dot, manifest sidecar required (captures recognized by what they ARE);
  the derived-suffix blacklist is gone. Tested against every .jsonl class that
  bit a caller; live check returns a real capture, not rig_log.
- [x] **R-6 FIXED (2026-07-19) -- suggestions name what they propose**:
  proposal_summary (now a tested module-level helper in live_loop) builds a noun
  phrase from any placement proposal regardless of held_k -- "a stone_bricks
  block at (282, 70, 188)" -- adding the commit count when held_k >= 1;
  non-placement proposals keep the old line. Body side: re-voices only when the
  offer itself changed (lastVoicedSummary), not every 30 s for the same words;
  the voiced log's summary field is now self-describing.
- [x] **R-7 FIXED (2026-07-19) -- no mixed-generation pairs**: any run_d2 failure
  (quarantine or crash) shelves the plain evidence3d/fused/belief siblings to
  *.stale.jsonl (tested, incl. repeat-crash re-shelving) -- a fresh evidence2d
  can never sit next to a stale or torn partner; the live-quarantined bank keeps
  the live record and the retry's --overwrite regenerates from scratch.
- [x] **R-8 FIXED (2026-07-19, with R-4)**: process_one refuses a recorded
  structure quarantine up front (seconds, clear message); --redo-evidence is the
  deliberate retry.
- [ ] **OPERATIONAL (runbook): give the rig a head start** -- start the watcher,
  wait for "models pre-warming" to finish (~3.5 min cold) BEFORE opening the world;
  tonight's entire live loss traces to a 92 s head start.
- Verified working tonight: proposal_first on all 451 gate reads; body voiced log +
  30 s throttle; suggestion_acceptance join (pairing now causal); labeling-tool
  discovery/ingest of the dated layout (JsonlSource parses the capture end to end,
  11005 packets, 364/364 events); A7 no-op confirmed (agent placed nothing; all
  364 events HumanBuilder). Gate reads: mean 378 ms, p90 524 ms, 1/451 over the
  1 s budget (worst 1256 ms, late-session, not warmup) -- live headroom is ~2x,
  not the 5x the offline measurement implied.

## 2026-07-19 -- VLM cross-check role resolved: DIAGNOSTIC ONLY (code-vs-vault conflict)

Finding (user, verified by two independent sweeps): the finished-build VLM
cross-check decides NOTHING. The builder-vs-matcher contest is settled inside
_label_real() (agrees_with_builder / pairs_withheld) before the VLM runs in
main(); every consumer of the vlm field is display-only (report JSON,
inspect.md line, labeling-tool card, console print); cascade eligibility reads
kept + agrees_with_builder only. Three texts claimed otherwise: D3 line ~100
("the VLM arbitrates", the 193059 amendment), its Step-by-Step echo, and the
comment at label_finished_builds.py:132 -- the single "arbitrat" hit in all
Python. The rest of the record already said confirm/contest-only, and this
file's own 2026-07-06 entry called the VLM "too noisy to arbitrate".

- [x] **User decision: Option A -- vault follows code.** The VLM is pinned
  DIAGNOSTIC ONLY. Fixed the :132 comment; dated corrections in D3 (line ~100
  amendment + a display-only confirmation at the ~106 pin) and the
  Step-by-Step echo. No logic changed anywhere; no pairs move.
- **Scope guard added at D3's validation record**: the 94% (n=16) is BINARY
  clip-verdict agreement with the sole annotator -- one graded negative,
  missed -- and licenses nothing about the five-way finished-build call,
  which has no accuracy number. Never cite it as arbitration evidence.
- **Bar for ever revisiting (Option B)**: a five-way accuracy number against
  known truth (the scripted corpus is the only known-truth set and has ZERO
  rendered frames -- rendering comes first), plus a second-annotator story;
  until then the VLM verdict stays a filed opinion.
- Corrected stakes on the 34 real sessions: 10 matcher-agreed, 15 contested
  (VLM sided with the builder 7, the matcher 4, a THIRD category 3, no
  verdict 1), 9 discarded on score/margin alone. The VLM's 7 builder-side
  votes changed nothing -- all 15 contested sessions have pairs withheld.

## 2026-07-19 -- agent-scan diagnosis, status-file rotation, hotbar pre-registration

Scan finding (user): 07-18 session scanned 0/216 build cells (agent never got
within scan range -- scan cells x[57,215] vs build x[254,312]; it likely never
reached the human from spawn). 07-19 morning session proves the mechanism:
156/171 cells (91%), h3d cosine 0.916 vs exact. WHY the 07-18 agent got stuck
is unrecoverable: its position log was a fixed-name file the next launch
overwrote.

- [x] **FIXED -- agent status file rotates like the scan** (agent.js
  rotateAside for both sidecars; status rows now carry the sticky session tag)
  and organize_raw adopts rotated status files into the session folder with the
  same owner logic (row tag -> filename -> wallclock; stale-active sweep
  included; tests). The next stuck-agent night is diagnosable.
- [ ] **Hotbar + selection-moment features -- PRE-REGISTERED for the v2 heads
  retrain** (D7 section 3, dated 2026-07-19 note): hotbar_dense (pooled item
  embeddings + placeable count + presence flag) and held_changed (within-window
  held-item switch flag). Decision rule pinned BEFORE training: with/without
  comparison arm; adopt iff held-out pooled NLL not worse AND
  pre-first-placement separation (accuracy + margin before each session's
  first human PLACE) improves; negative result banked otherwise. Rides the
  pinned larger-batch retrain milestone -- no early retrain.
- Open question carried: why the 07-18 agent never reached the build
  (pathfinding? terrain?) -- answerable from the NEXT occurrence's rotated
  status log; not worth reverse-engineering blind.

## 2026-07-19 -- agent movement: dynamic band + stuck recovery (user direction)

- [x] **Dynamic follow band** (follow_math.js, pure + tested; agent.js wired):
  the fixed 4-8 band becomes moment-driven -- ENGAGED 3-6 for ~12 s after
  voicing a suggestion (step in, make it social), BUSY 6-10 while the human
  moves fast (>4 blocks/s), DEFAULT 4-8 otherwise; 2 s adoption hold against
  boundary churn. Yield widening, workspace rule, receiving posture unchanged.
- [x] **Stuck recovery** (the 07-18 root cause class): the far-follow goal was
  set ONCE and never re-issued when the path died. Now progress is watched
  (a block closer = progress); a 5 s stall re-issues the goal, the status
  file's follow mode reads "recovering", and a 30 s stall >=20 blocks away is
  said in chat (60 s gap, MICA_QUIET respected). Patrol may no longer claim
  the tick beyond 20 blocks -- reaching the human outranks scanning.
- Placement authority: NOT touched here -- the body's place machinery already
  exists; what stops it is the pinned gate (theta_place 0.438 above validation
  conf max 0.427 + the --place blockers). Decision pending with the user.

## 2026-07-19 -- CONSENT placement route (D5 s10 answered; user decision)

The user chose the consent route over waiting for Stage-1 calibration: an
explicit chat "yes" within 60 s of a voiced suggestion licenses EXACTLY ONE
placement of that proposal, whatever the confidence -- the license is the
human's word, so miscalibration cannot place a block.

- [x] Vault: D5 s10's open consent question answered with a dated decision
  note (mechanics, non-touches, arming, auditor rules).
- [x] Mind: LiveGateRunner(consent_place=...) -- consent rides the agent
  snapshot (materials.read_agent_snapshot 4th element); consent_veto (pure,
  tested) enforces single-use, 60 s freshness, cell match, entomb guard
  (1.5 blocks), stock; directive route "consent"; authority label "consent";
  vetoed yeses land in the trace reason. theta_place, the staircase, YIELD,
  A7 untouched.
- [x] Body: hears yes/no (materials ask outranks; negations excluded), relays
  consent {ts, cell} in its status line, executes any directive whose id is
  new whatever the FSM state (s9's rule is directive-presence, not state).
  One yes per voicing; stop still halts everything.
- [x] Arming: run_live --consent-place; rig MICA_CONSENT=1 (off by default --
  unattended rigs keep the pure demo posture).
- [x] Auditor: consent authority accepts commits only on route consent, one
  per read, reversible, never YIELD; the staircase check is unchanged (consent
  never arms place_low_risk); consent-route commits outside consent authority
  fail.
- [ ] Acceptance signal follow-up: a consented placement is an AGENT block
  (A7-filtered), so classify() reads the voiced suggestion as "ignored" today.
  Add an "accepted_explicitly" tier (voiced -> consent -> directive executed)
  to suggestion_acceptance -- the strongest acceptance datum there is.

## 2026-07-19 -- contested-pairs comparison RE-REGISTERED (rule v2, runs at next cascade)

The user asked whether training on the 15 contested sessions could improve
accuracy. The 07-12 experiment said no ("withholding rule STANDS") -- but its
inputs are all repaired since (idle leak was live in both arms, corpus predates
the symmetry/anchor regeneration, both arms sat at pooled ECE ~0.45). The
question re-asks on the fixed pipeline; v1 is history, not precedent.

- [x] Re-registration v2 appended to the 07-12 vault note BEFORE any v2 run:
  same arms (ALL 15 contested, builder's label as truth, NO selection -- and
  VLM-based selection explicitly prohibited under the diagnostic-only pin);
  eval set derives from the current corpus (4 reserved clean + all 9 discarded).
- [x] Decision rule v2 implemented in run_contested_comparison.py: adopt-worthy
  ONLY if all-eval final accuracy STRICTLY beats control (the off-template gain
  the selection-bias hypothesis predicts) AND clean accuracy >= control AND
  pooled ECE <= control + 0.02 AND mean sustained-from <= control + 0.02.
- [x] Wired as the final step of run_cascade_a.py -- the comparison rides every
  deliberate retrain; a win still changes nothing without a dated D3 amendment.
- Rationale kept honest: matcher agreement IS template similarity, so
  agreed-only training biases toward template-like builds while the headline
  failure (never-seen floor 0.050) lives off-template. The v1 counter-lesson
  (anti-associations from mislabeled evidence) stands as the risk the rule
  tests against.

## 2026-07-19 -- decoder real-session NTP pretraining: PRE-REGISTERED + BUILT

The label-free door for contested/discarded sessions (user decision: build now,
ride the next cascade alongside contested-pairs v2). Vault note:
"2026-07-19 - Decoder Real-Session NTP Pretraining (pre-registered)".

- [x] Architecture fact it rests on: the rationale goal is a separate head, not
  a token -- real rows train action tokens only (goal None; rationale CE and
  counterfactual sharpening skip them; leak impossible because no label exists).
- [x] Corpus: make_decoder_corpus --real builds real_pretrain + real_holdout
  groups -- human events as plan-shaped targets (A7 at the source, mistake
  pairing + ALL hygiene asserts unchanged, out-of-range dropped and counted,
  quiet records thinned 1-in-8). REAL_NTP_HOLDOUT pre-registered by id:
  194844, 052914 (contested), 004022 (agreed), 150013 (discarded).
- [x] Trainer: --real-pretrain mixes real rows into STAGE A only;
  --out-prefix ships comparison arms to their own files (pin logic skipped --
  pins govern production only). Smoke on 194844: 364 events -> 606 goal-free
  samples, every hygiene assert passed.
- [x] run_decoder_ntp_comparison.py: control = the production decoder the
  cascade just trained; pretrained arm -> models/decoder_cmp_real.*; verdict =
  real-holdout NLL STRICTLY improves AND scripted-holdout NLL within 2%.
  Banked at capture/raw/decoder_ntp_comparison.json. Adoption is two-step:
  user-approved recipe flip + next cascade's full OQ1 on the new recipe.
- [x] Cascade: make step gains --real; comparison appended after
  contested-pairs. Fixed in passing: train_decoder referenced _ROOT without
  defining it (the 07-17 stage-pin edit -- would have crashed the next retrain).

## 2026-07-19 night -- CASCADE A COMPLETE: both pre-registered experiments WIN

Run at the user's word (10 agreed; before-models banked in pre_cascade_a; ledger
stamped, 12,045-pair baseline). Two cascade-killing bugs found and fixed on the
way -- both the SAME ghost: 210003's quarantined truncated bank entering through
new doors (the real-NTP groups, then the contested eval set); both doors now
honor the quarantine rule. OQ1 PASSES again on the fresh corpus; gate thetas
STABLE at 0.338/0.438 (a good sign for the freeze's robustness).

- [x] **Contested-pairs v2: CONTESTED-INCLUSION WINS** (all four criteria; three
  improved outright): clean 0.000->0.333, all 0.182->0.273, pooled ECE
  0.2584->0.2324, sustained-from 0.990->0.897. The 07-12 anti-association
  failure did NOT reproduce on the repaired pipeline. Caveat: small eval
  (11 sessions). Banked in the 07-12 note (Results v2).
- [x] **Real-NTP pretraining WINS decisively**: real-holdout NLL 3.42->2.17
  (-37%) and scripted holdout ALSO improved (1.7683->1.7246). The domain gap
  was 1.25 nats -- real streams were the missing distribution, not noise.
  Banked in the 07-19 note.
- [x] **DECIDED (user, 2026-07-19): BOTH ADOPTED.** D3 amended -- contested
  pairs join production heads training (builder's label as truth; cascade runs
  --contested-pairs / --include-contested-pairs). Decoder recipe flipped to
  --real-pretrain in the cascade; the same chain's OQ1 adjudicates the new
  recipe before any live session exists. Both effective at the NEXT cascade;
  tonight's shipped models remain the old-recipe ones. The comparison harnesses
  stay in the chain as standing monitors of both adopted rules.
- Watcher restarted post-cascade with MICA_CONSENT in its environment -- the
  next session is the first with consent actually armed.

## 2026-07-19 late -- belief-slot probe (POST-HOC, not pre-registered): no gain

Question (report focus): does the inferred belief improve next-action
prediction on real sessions? scripts/probe_belief_slot_gain.py (one command,
banked at capture/raw/belief_slot_probe.json): forced intent slot belief vs
zeros vs arm1, 1,158 real_holdout rows, horizons 1/2/4/8, both decoders.

- Finding: NO belief gain at any horizon (|delta| <= 0.001) on either model.
  The belief is computed from the same evidence the decoder reads --
  conditionally redundant for action prediction. Its demonstrated value stays
  situation classification (0.042 vs 0.375) + gate licensing, not prediction
  fuel. Also: the pretrained decoder beats production at EVERY horizon on real
  sessions (h1 0.710 vs 0.632; h4 0.586 vs 0.513).
- Report consequence (user's focus question): primary thesis = anticipation
  with minimal context; intent recognition claimed as classification, not as
  prediction driver. Recorded in the Why draft (section 4b).
- Future-work framing: making the belief matter for prediction = Stage-1
  calibration + goal-conditioned proposal evaluation (the D9 arc).

## 2026-07-19 late -- PIN (user decision): no assistance-efficacy claim

The agent has placed ZERO blocks across all live sessions (demo/advisory
configs; confidence route unreachable; consent armed only after the last
session). Pinned in D9 s3b: the report describes assistance as designed,
instrumented, and safety-gated -- NEVER evaluated. Future bar for any
helpfulness claim: consent-route sessions with real placements + acceptance
signal at meaningful n + placement SURVIVAL as the outcome measure (a block
the agent placed remaining in the finished build -- measurable for free from
the existing capture's block events, A7 attribution already separates actors).
The sessions themselves stay fully valid for every non-assistance claim.
