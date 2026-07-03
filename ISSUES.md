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
- [~] **D-8 — pause/resume straddle:** mostly handled by the `justPaused` flush; a sub-tick residual remains.

## C. Working-as-intended (documented, no action)
- [~] **L-2 — a tick-0 build action fails the B0 gate** (`_actions_have_prior_moment`, no tick −1). This is
  *correct* — the very first action has no "before" to score against, so the gate rightly rejects such a
  capture. (Made non-silent by U-1: `run_d1` now flags any unconsumed event.)

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
  tracker beats the structure-only floor for the first time.** Handoff verified per record; invariants
  every step; tests 82 → **86** (channel-arrival, held-item-reaches-evidence, mismatch-refusal,
  load-bearing-wire tests).
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
