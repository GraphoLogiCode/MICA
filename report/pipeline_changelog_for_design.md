# MICA pipeline — changelog for the diagram update (old → new, as of 2026-07-19)

Baseline: the pipeline as it stood mid-July (before the review-and-fix week).
Everything below is ADDED / CHANGED / REMOVED / RENAMED relative to that.

## Answers to the form

**Keep the current paper-figure style?** YES — same look as the paper figures:
serif/KaTeX look, exact palette:
- micablue  RGB 46,110,180  (primary / belief path)
- micagray  RGB 140,140,140 (baselines, secondary)
- micagreen RGB 60,145,90   (decoder / anticipation path, wins)
- micared   RGB 180,70,60   (annotations, vetoes)

**Updated sources**: `report/report.pdf` (7-page draft, all claims + data),
`report/evidence_figures.pdf` (6 figures), both with .tex sources in `report/`.
Code: `D:\2026projects\MICA` (single repo; FlowViz dashboard separate at
`D:\2026projects\mica-flowviz`).

## Stage-by-stage changes

### B0 capture (Fabric mod) — unchanged contract, one version note
- Unchanged: proof-grade capture (block events with actor attribution, POV
  frames, input state, hotbar, sparse full inventory).
- The live socket remains deliberately lossy (disk is the complete copy) — if
  the old diagram implied the socket buffers before a consumer attaches, it
  does not.

### D1 — 2D evidence (VPT h2d + MineCLIP s_goal + symbolic state)
- REMOVED: the idle-bit feature from shared_dense (it leaked the scoring
  target; SHARED_DIM 14 → 13). A leak tripwire test class now guards this.
- CHANGED: fusion hyperparameters re-swept under a renormalized objective —
  channel weights flipped from (0, 0.5, 1.0) to (1.0, 0.5, 0.0).

### D2 — 3D evidence (template matcher + frozen Uni3D h3d)
- CHANGED: symmetry computed by doubled-coordinate exact reflections (old
  formula was wrong); ADDED vertical anchoring by ground-level voting; the
  Pose contract gained a `dy` field.
- The whole evidence corpus was regenerated under these fixes (19/25 matcher
  verdicts changed).

### Belief tracker b(goal, mode)
- Retrained (heads v1) on the leak-free features + regenerated corpus.
- ADDED runtime asserts on likelihood range and belief positivity.
- NEW measured fact for the diagram's honesty labels: belief confidence is
  NOT calibrated above 0.4 (inversion); a Stage-1 recalibration box can be
  drawn as FUTURE (dashed).

### D4 — action decoder (NTP → MTP)
- ADDED: real-session NTP pretraining (ADOPTED recipe): stage A now trains on
  scripted samples PLUS goal-free samples built from real sessions' actual
  action streams — including contested and discarded sessions (the label
  cannot leak: the goal rationale is a separate head, masked for real rows).
  New corpus groups: `real_pretrain` (trains) and `real_holdout` (eval-only,
  4 sessions pinned by id).
- ADDED: an executable stage pin (`models/decoder_stage_pin.json`) — which
  stage ships is an adjudicated decision (OQ1), not a training side effect.
  Stage B (MTP) currently ships legitimately (OQ1 PASS).
- NEW measured fact: the belief slot adds NO prediction gain at any horizon
  (post-hoc probe) — if the old diagram drew a thick "belief → decoder"
  conditioning arrow, thin it: the belief's real downstream consumer is the
  GATE (licensing), not the decoder's accuracy.

### D5 — commit gate + FSM
- CHANGED: fully re-frozen against the retrained models: θ₁ 0.45,
  δ̂ 0.2295 s, θ_suggest 0.338, θ_place 0.438. Note for the diagram:
  θ_place sits ABOVE the max observed confidence — the confidence-licensed
  placement path is honestly unreachable until recalibration (draw it dashed
  / annotated "unreachable pending Stage 1").
- REMOVED (if still drawn): the declared-target placement route for live
  sessions (retired 07-13).
- ADDED — THE BIG NEW EDGE: the **CONSENT route**, a third placement license.
  Flow: gate voices a suggestion naming its offer (block + cell) → human says
  "yes" in chat → the BODY relays consent {ts, cell, block} in its status
  file → the MIND matches the offer (single-use, 60 s fresh, entomb guard,
  stock check, never in YIELD) → ONE placement directive, route="consent",
  authority="consent". Miscalibration cannot place a block: the license is
  the human's word. (Zero live placements so far — pinned; the route is
  built and armed, not yet exercised.)
- ADDED trace/status fields: `proposal_first` {block, cell} on EVERY gate
  read (the acceptance signal's control group); `target_block` in the status
  gate block; suggestions now NAME their offer ("a stone_bricks block at
  (x,y,z)") instead of a generic line.

### Embodied agent (body, mineflayer)
- CHANGED: follow band is now DYNAMIC (pure module `follow_math`): engaged
  3–6 blocks for ~12 s after voicing (step in), busy 6–10 while the human
  moves fast (give room), default 4–8. Old fixed 4–8 band is gone.
- ADDED: far-follow stuck recovery (progress watched; dead pathfinder goals
  re-issued every 5 s; 30 s stall announced in chat); patrol may not claim
  the agent beyond 20 blocks from the human.
- ADDED: consent capture (hears "yes", one per voicing, materials-ask
  outranks) and the voiced log `agent-<NAME>.voiced.jsonl` (NEW artifact —
  the acceptance signal's voiced side).
- CHANGED: any item dropped nearby is collected (hand-over), not only during
  a shortage; shortage requests now speak under consent authority too.
- CHANGED: the raycast scan re-records a cell when its BLOCK CHANGES
  (old: once forever — walls built on scanned ground were invisible);
  last-sighting-wins in every consumer. Camera viewDistance 4 → 8 chunks.
- ADDED: per-session rotation of the status file (like the scan), rows carry
  a session tag.

### D9 — acting layer (ENTIRELY NEW MODULE since the old diagram)
- The suggestion-acceptance signal: every voiced suggestion vs the unvoiced-
  proposal CONTROL group → outcome tiers (followed exact/near/type,
  contradicted, ignored) → the reportable number is the LIFT. v2 semantics
  (namespace-blind block matching, causal voicing pairing).
- Calibrated-autonomy staircase (design): Stage 1 recalibration → Stage 2
  EV decision rule → Stage 3 earned autonomy. All FUTURE (dashed) except the
  instrumentation, which is live.
- PIN for any diagram annotation: no assistance-efficacy claim — the agent
  has placed zero blocks in live sessions.

### Post-session + training loop (cascade A)
- ADDED to production training: CONTESTED sessions' pairs, builder's label as
  truth (D3 amendment after the pre-registered v2 win). The old
  "matcher-agreed only" filter on the training-data edge is gone; discarded
  sessions still produce no pairs (but DO feed decoder pretraining).
- ADDED to the cascade chain (standing monitors, run every retrain):
  the contested-pairs comparison and the decoder real-NTP comparison — both
  pre-registered, both verdict-banking, neither ships anything by itself.
- CHANGED chain robustness: a run_d2 crash is no longer stamped "structure
  quarantine" (only the replay's own fresh verdict is); failed evidence
  leaves no mixed-generation artifacts (stale siblings shelved); quarantined
  sessions are excluded consistently at every consumer.
- Labeling tool: quarantined/gate-failed sessions now VISIBLE as read-only
  cards; contested card text: "your word wins."

### Rig automation (watcher)
- ADDED: per-child log files (`capture/raw/child_logs/`) — crash tracebacks
  survive; GPU serialization (run_live pre-warm defers while the post-session
  chain runs; live game always outranks the chain); startup chain-retry sweep
  for crashed chains; `MICA_CONSENT=1` arms the consent route.

### FlowViz dashboard (separate repo)
- Scan panel tells the truth off-frame ("N seen — none in the build frame"),
  keeps the newest 8192 cells (was: lowest-sorted 1024 — blocks vanished),
  updates re-recorded cells; camera badge dies with the agent (was: "live"
  forever); cross-session file guards.

## New/renamed artifacts (for a data-flow diagram)
- NEW: `agent-<NAME>.voiced.jsonl`, `consent` field in agent status rows,
  `proposal_first` + `target_block` fields, `<sid>.suggestion_acceptance.json`,
  `models/decoder_stage_pin.json`, decoder corpus `real_pretrain`/`real_holdout`
  groups, `capture/raw/decoder_ntp_comparison.json`,
  `capture/raw/belief_slot_probe.json`, `capture/raw/child_logs/`.
- RENAMED/semantic: gate trace `authority` now three-valued: demo | consent |
  place (was two).

## Current headline numbers (if the diagram carries them)
- Necessity: 0.042 (reactive) vs 0.375 (belief v1), 24 never-seen sessions.
- Anticipation on real holdout: 0.71 (next action) / 0.59 (8 ahead), chance 0.012.
- Scan convergence: 0.92 cosine at 91% coverage.
- Corpus: 37 labeled real sessions (10 agreed / 18 contested / 9 discarded).
