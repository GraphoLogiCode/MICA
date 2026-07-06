"""The D4 proof-log run: proposals, confidences, OQ1, contracts, and the gate.

    python scripts/run_decoder_eval.py [--trace-every N]

One command regenerates everything (D0 standard):
  capture/raw/decoder_report.json      every number quoted anywhere
  capture/raw/decoder_proposals.jsonl  proposal dumps (D0's D4 proof-log row)
  capture/raw/decoder_commit_trace.jsonl  per-read K_commit staircase rows
  capture/raw/decoder_conf_hist.png / decoder_coherence.png / decoder_commit.png
  models/gate_v1.json                  frozen delta_hat + thresholds + reachability

What it measures, and under which rule:
  OQ1 retrofit check (GATING milestone, MultiTokenPrediction Open Question 1): the
      Stage-B MTP decoder against the Stage-A single-head checkpoint on held-out
      sessions. No decoder-quality claim enters the paper unless this is recorded.
      Verdict rule, pre-registered here: the retrofit must not hurt — Stage B
      one-step holdout NLL within 2% of Stage A's, coherence within 0.02 — and its
      deep horizons must carry signal (horizon-4 tau-accuracy above chance).
  Plan coherence: a proposed placement is coherent when its cell is genuinely still
      missing from the true plan (precision@K, by build-progress bin).
  C3 faithfulness: rationale_goal vs the belief argmax — mismatch logs and rejects
      the chunk; DIAGNOSTIC only, never a gate condition.
  Intervenability (the proof-log test D4 pins): forcing the belief to each goal on
      early ambiguous holdout scenes must flip the proposed action.
  C5 parity: held-out one-step NLL and proposal acceptance per slot condition —
      the decoder must not quietly prefer one arm's slot beyond intent content.
  Gate calibration, POOLED and then frozen (never per-arm, never on the headline
      metric): delta_hat from validation sessions; c_min by F1 of confidence
      predicting proposal-vs-target agreement; theta by the most demanding line
      that stays A-PRIORI REACHABLE (12-F6, checked before the sweep) and keeps
      commit availability on validation. Belief side runs the calibrated v1 heads
      (the D5 §9 pin: production thresholds freeze only against calibrated heads).
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.contracts.b3 import fuse_dicts                              # noqa: E402
from mica.contracts.b4 import SLOT_KINDS                              # noqa: E402
from mica.contracts.goals import GOALS                                # noqa: E402
from mica.data import decoder_corpus                                  # noqa: E402
from mica.decoder import context as context_builder                   # noqa: E402
from mica.decoder import model as decoder_model                       # noqa: E402
from mica.decoder import tokenizer                                    # noqa: E402
from mica.decoder.grammar import Place, action_from_json, action_to_json  # noqa: E402
from mica.gate import commit, reversibility                           # noqa: E402
from mica.intent import heads_v1                                      # noqa: E402
from run_tracker import real_labeled_sessions                         # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_SCRIPTED = os.path.join(_ROOT, "capture", "scripted")
_GATE_META = os.path.join(_ROOT, "models", "gate_v1.json")

_TRACE_EVERY_DEFAULT = 10        # sample every Nth correction for gate traces
_MAX_ACTIONS = 8                 # chunk length cap: K positions the gate can read
_PROGRESS_BINS = [round(0.1 * b, 1) for b in range(1, 11)]

_C_MIN_GRID = [round(0.5 + 0.05 * i, 2) for i in range(10)]
# theta_1 grid floor sits above the |G| = 5 chance mass (0.2). The 0.25/0.30 rungs
# were added 2026-07-05 after the first sweep hard-failed: the CALIBRATED v1 belief
# tops out at p* = 0.335 on the real validation session (measured) — honest
# underconfidence, Phase E's own finding. Extending the grid before any freeze is
# calibration, not headline tuning; the deviation is recorded in the review note.
_THETA_GRID = [(theta_1, slope) for theta_1 in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
               for slope in (0.01, 0.02, 0.04)]
_IRREV_PENALTY = 0.2
_MIN_COMMIT_AVAILABILITY = 0.05  # a frozen theta must leave the gate reachable in practice


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _context_from_row(row: dict, slot_kind: str):
    """A corpus row + a slot condition -> the B4 record the decoder reads."""
    slot_key = {"arm3": "arm3", "arm1_dense": "arm1", "arm2_llm": "arm2"}.get(slot_kind)
    if slot_kind == "arm0_zero":
        slot = context_builder.arm0_slot()
    else:
        stored = row["slots"][slot_key]
        if stored is None:
            return None
        slot = {"goal_marginal": tuple(stored["goal_marginal"]),
                "top_goal": stored["top_goal"], "p_top": stored["p_top"],
                "entropy_nats": stored["entropy_nats"], "p_z1": stored["p_z1"]}
    from mica.contracts.b4 import ControlContext

    return ControlContext(
        tick=row["tick"], k=row["k"], belief_snapshot_id=row["tick"],
        slot_kind=slot_kind, goal_marginal=tuple(slot["goal_marginal"]),
        top_goal=slot["top_goal"], p_top=slot["p_top"],
        entropy_nats=slot["entropy_nats"], p_z1=slot["p_z1"],
        evidence=tuple(row["evidence"]))


def _teacher_forced_nll(model, rows: list[dict], slot_kind: str, device) -> float | None:
    """Held-out one-step token NLL under ONE fixed slot condition (the C5 read)."""
    import torch
    import torch.nn.functional as functional

    values, counts = 0.0, 0
    with torch.no_grad():
        for row in rows:
            ctx = _context_from_row(row, slot_kind)
            if ctx is None:
                continue
            ids = tokenizer.encode_chunk([action_from_json(b) for b in row["target"]])
            evidence, slot, kind = decoder_model.context_tensors(ctx)
            inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            hidden = model.trunk_hidden(evidence.unsqueeze(0).to(device),
                                        slot.unsqueeze(0).to(device),
                                        kind.unsqueeze(0).to(device), inputs)
            token_hidden = hidden[:, model.config.prefix_len:, :]
            logits = model.token_logits(token_hidden, 0)
            labels = torch.tensor([ids[1:]], dtype=torch.long, device=device)
            values += float(functional.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), reduction="sum"))
            counts += labels.numel()
    return round(values / counts, 4) if counts else None


def _horizon_accuracy(model, rows: list[dict], device, horizon: int) -> float | None:
    """Tau-agreement accuracy of the horizon head against gold, teacher-forced."""
    import torch

    hits, total = 0, 0
    with torch.no_grad():
        for row in rows:
            ctx = _context_from_row(row, "arm3")
            ids = tokenizer.encode_chunk([action_from_json(b) for b in row["target"]])
            if len(ids) <= horizon:
                continue
            evidence, slot, kind = decoder_model.context_tensors(ctx)
            inputs = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            hidden = model.trunk_hidden(evidence.unsqueeze(0).to(device),
                                        slot.unsqueeze(0).to(device),
                                        kind.unsqueeze(0).to(device), inputs)
            token_hidden = hidden[:, model.config.prefix_len:, :]
            usable = inputs.shape[1] - (horizon - 1)
            picks = model.token_logits(token_hidden[:, :usable, :],
                                       horizon - 1).argmax(dim=-1)[0]
            for at, pick in enumerate(picks.tolist()):
                label = ids[at + horizon]
                if label == tokenizer.PAD_ID:
                    continue
                hits += tokenizer.agreement(pick, label)
                total += 1
    return round(hits / total, 4) if total else None


def _actions_agree(action_a, action_b) -> bool:
    ids_a, ids_b = tokenizer.encode_action(action_a), tokenizer.encode_action(action_b)
    return (len(ids_a) == len(ids_b)
            and all(tokenizer.agreement(a, b) for a, b in zip(ids_a, ids_b)))


def _remaining_cells(placements, mistakes, action_tick, origin) -> set:
    """Every cell the true plan still owes, origin-relative (coherence's yardstick)."""
    cells = set()
    for action in decoder_corpus.helper_actions(placements, mistakes, action_tick,
                                                origin, limit=10 ** 6):
        if isinstance(action, Place):
            cells.add((action.dx, action.dy, action.dz))
    return cells


def _bin_of(progress: float) -> str:
    for edge in _PROGRESS_BINS:
        if progress <= edge:
            return str(edge)
    return str(_PROGRESS_BINS[-1])


def main() -> int:
    import torch

    trace_every = _flag_value("--trace-every", _TRACE_EVERY_DEFAULT)
    if not decoder_model.available():
        print("no trained decoder on disk — run scripts/train_decoder.py first")
        return 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, meta = decoder_model.load()
    model.to(device)
    stage_a = decoder_model.build_model(config)
    stage_a.load_state_dict(torch.load(decoder_model.STAGE_A_PATH,
                                       map_location="cpu", weights_only=True))
    stage_a.eval().to(device)

    train_rows, eval_rows, labels = decoder_corpus.load_rows()
    holdout = [r for r in eval_rows if labels[r["session"]]["group"] == "holdout"]
    banked = [r for r in eval_rows if labels[r["session"]]["group"] == "banked_holdout"]
    print(f"decoder eval on {device.type}: {len(holdout)} holdout + "
          f"{len(banked)} banked-transfer samples")

    placements_by_session, skips_by_session = {}, {}
    for session_id, label in labels.items():
        if label["group"] in ("holdout", "banked_holdout"):
            placements = decoder_corpus.regenerate_placements(label)
            placements_by_session[session_id] = placements
            skips_by_session[session_id] = decoder_corpus.skipped_place_ids(placements)

    # ---- proposals + confidence histogram + coherence + C3 + c_min labels ------
    proposals_path = os.path.join(_RAW, "decoder_proposals.jsonl")
    conf_values: list[float] = []
    conf_labels: list[tuple[float, bool]] = []       # (confidence, agreed-with-target)
    coherence_bins: dict[str, list[float]] = {}
    group_precision: dict[str, list[float]] = {"holdout": [], "banked_holdout": []}
    parse_failures, nothing_to_do, mismatches, chunks = 0, 0, 0, 0
    with open(proposals_path, "w", encoding="utf-8") as dump:
        for row in holdout + banked:
            ctx = _context_from_row(row, "arm3")
            proposal, reason = decoder_model.propose(model, ctx, _MAX_ACTIONS)
            group = labels[row["session"]]["group"]
            if proposal is None:
                # "build looks finished" is an answer, not a failure (review F6)
                if reason == decoder_model.NOTHING_TO_DO:
                    nothing_to_do += 1
                else:
                    parse_failures += 1
                dump.write(json.dumps({"session": row["session"], "k": row["k"],
                                       "rejected": reason}) + "\n")
                continue
            chunks += 1
            conf_values.extend(proposal.token_conf[1:])
            targets = [action_from_json(b) for b in row["target"]]
            for index, action in enumerate(proposal.actions[1:], start=1):
                agreed = index < len(targets) and _actions_agree(action, targets[index])
                conf_labels.append((proposal.token_conf[index], agreed))
            remaining = _remaining_cells(placements_by_session[row["session"]],
                                         skips_by_session[row["session"]],
                                         row["tick"], tuple(row["origin"]))
            placed = [a for a in proposal.actions if isinstance(a, Place)]
            if placed and remaining:
                precision = statistics.mean(
                    (a.dx, a.dy, a.dz) in remaining for a in placed)
                coherence_bins.setdefault(_bin_of(row["progress"]), []).append(precision)
                group_precision[group].append(precision)
            faithful = proposal.rationale_goal == ctx.top_goal
            mismatches += not faithful
            dump.write(json.dumps({
                "session": row["session"], "k": row["k"], "progress": row["progress"],
                "belief_snapshot_id": ctx.belief_snapshot_id,
                "top_goal": ctx.top_goal, "p_top": round(ctx.p_top, 4),
                "actions": [action_to_json(a) for a in proposal.actions],
                "token_conf": [round(c, 4) for c in proposal.token_conf],
                "rationale_goal": proposal.rationale_goal,
                "faithful_c3": faithful,
                "chunk_rejected_by_c3": not faithful}) + "\n")

    coherence = {bin_: round(statistics.mean(vals), 4)
                 for bin_, vals in sorted(coherence_bins.items())}
    faithfulness = {"chunks": chunks, "rationale_mismatches": mismatches,
                    "mismatch_rate": round(mismatches / chunks, 4) if chunks else None,
                    "rule": "mismatch => chunk logged and rejected; diagnostic only (C3)"}

    # ---- OQ1: the retrofit check (gating milestone) ----------------------------
    oq1 = {"stage_a": {}, "stage_b": {}}
    for name, checkpoint in (("stage_a", stage_a), ("stage_b", model)):
        nll = _teacher_forced_nll(checkpoint, holdout, "arm3", device)
        precisions = []
        for row in holdout:
            proposal, _ = decoder_model.propose(checkpoint, _context_from_row(row, "arm3"),
                                                _MAX_ACTIONS)
            if proposal is None:
                continue
            remaining = _remaining_cells(placements_by_session[row["session"]],
                                         skips_by_session[row["session"]],
                                         row["tick"], tuple(row["origin"]))
            placed = [a for a in proposal.actions if isinstance(a, Place)]
            if placed and remaining:
                precisions.append(statistics.mean(
                    (a.dx, a.dy, a.dz) in remaining for a in placed))
        oq1[name] = {"holdout_token_nll": nll,
                     "coherence_precision": round(statistics.mean(precisions), 4)
                     if precisions else None}
    horizon_4 = _horizon_accuracy(model, holdout, device, horizon=4)
    chance = round(1 / tokenizer.vocab_size(), 4)
    nll_ok = (oq1["stage_b"]["holdout_token_nll"] is not None
              and oq1["stage_b"]["holdout_token_nll"]
              <= oq1["stage_a"]["holdout_token_nll"] * 1.02)
    coh_ok = (oq1["stage_b"]["coherence_precision"] is not None
              and oq1["stage_b"]["coherence_precision"]
              >= (oq1["stage_a"]["coherence_precision"] or 0.0) - 0.02)
    deep_ok = horizon_4 is not None and horizon_4 > chance * 5
    oq1["horizon_4_tau_accuracy"] = horizon_4
    oq1["token_chance_level"] = chance
    oq1["verdict"] = ("PASS" if (nll_ok and coh_ok and deep_ok) else "FAIL") + (
        f" — retrofit NTP cost ok={nll_ok}, coherence kept={coh_ok}, "
        f"deep horizon carries signal={deep_ok}")

    # ---- intervenability (belief-level do(g), early ambiguous scenes) ----------
    early = [r for r in holdout if r["progress"] < 0.3 and r["target"]]
    flips, aligned, tested = 0, 0, 0
    for row in early[:60]:
        first_actions = {}
        for goal in GOALS:
            # the exact slot shape training's counterfactual augmentation produced:
            # point-mass goal, mode read kept, entropy = the mode's own entropy
            p_z1 = row["slots"]["arm3"]["p_z1"]
            marginal = tuple(1.0 if g == goal else 0.0 for g in GOALS)
            slot = {"goal_marginal": marginal, "top_goal": goal, "p_top": 1.0,
                    "entropy_nats": _mode_entropy(p_z1), "p_z1": p_z1}
            forced = context_builder.control_context(
                _FakeFused(row), row["k"], "arm3", slot, tuple(row["origin"]),
                evidence=tuple(row["evidence"]))
            proposal, _ = decoder_model.propose(model, forced, max_actions=2)
            if proposal is not None:
                first_actions[goal] = action_to_json(proposal.actions[0])
        if len(first_actions) >= 2:
            tested += 1
            flips += len({json.dumps(a, sort_keys=True)
                          for a in first_actions.values()}) >= 2
            truth = labels[row["session"]]["goal"]
            truth_action = first_actions.get(truth)
            if truth_action and truth_action.get("pos"):
                remaining = _remaining_cells(placements_by_session[row["session"]],
                                             skips_by_session[row["session"]],
                                             row["tick"], tuple(row["origin"]))
                aligned += tuple(truth_action["pos"]) in remaining
    intervenability = {"scenes_tested": tested,
                       "flip_rate": round(flips / tested, 4) if tested else None,
                       "forced_truth_alignment": round(aligned / tested, 4) if tested else None,
                       "rule": "forcing g must flip the argmax action (D4 proof-log test)"}

    # ---- C5 parity: per-arm NLL + acceptance ----------------------------------
    parity = {}
    for slot_kind in SLOT_KINDS:
        rows_with = [r for r in holdout if _context_from_row(r, slot_kind) is not None]
        accepted, proposals_made = 0, 0
        for row in rows_with[::5]:
            proposal, _ = decoder_model.propose(
                model, _context_from_row(row, slot_kind), _MAX_ACTIONS)
            proposals_made += 1
            accepted += proposal is not None
        parity[slot_kind] = {
            "holdout_token_nll": _teacher_forced_nll(model, rows_with, slot_kind, device),
            "samples": len(rows_with),
            "proposal_parse_rate": round(accepted / proposals_made, 4)
            if proposals_made else None}

    # ---- gate: freeze delta_hat, check reachability, sweep, trace --------------
    params = heads_v1.tracker_params()
    holdout_sessions = sorted({r["session"] for r in holdout})
    tick_sequences = [[r["tick"] for r in holdout if r["session"] == s]
                      for s in holdout_sessions]
    real_sessions = real_labeled_sessions()
    validation_real = "fabric-20260705-134615"
    validation_p_tops: list[float] = []
    if validation_real in real_sessions:
        from mica.intent.tracker import category_marginal

        fused = _load_banked_fused(_RAW, validation_real)
        tick_sequences.append([f.tick for f in fused])
        validation_p_tops = [max(category_marginal(b).values())
                             for b in decoder_corpus.replay_beliefs(fused)]
    gap = commit.expected_gap(tick_sequences)
    delta_hat = gap["delta_hat_seconds"]

    c_min = _sweep_c_min(conf_labels)
    # availability is measured on the REAL validation session's replayed belief —
    # the gate's production domain. The scripted corpus's v1 beliefs are near-flat
    # (the Phase-F domain split, measured again here: p_top <= 0.215), so sweeping
    # availability there would freeze against a population the live gate never sees.
    thresholds, reachability, theta_sweep = _sweep_theta(
        validation_p_tops or [r["slots"]["arm3"]["p_top"] for r in holdout],
        params, delta_hat, c_min)
    with open(_GATE_META, "w", encoding="utf-8") as handle:
        json.dump({"delta_hat": gap, "thresholds": commit.thresholds_to_json(thresholds),
                   "reachability": reachability, "c_min_sweep_rule":
                       "F1 of confidence predicting proposal-vs-target agreement",
                   "theta_sweep": theta_sweep,
                   "availability_population":
                       f"replayed v1 belief on {validation_real} "
                       f"({len(validation_p_tops)} steps); scripted-corpus beliefs "
                       "are near-flat under v1 heads (measured) and are not the "
                       "gate's production domain",
                   "heads": "calibrated v1 (D5 §9 pin)"}, handle, indent=2)

    trace_path = os.path.join(_RAW, "decoder_commit_trace.jsonl")
    trace_stats = _commit_traces(model, holdout, labels, real_sessions, params,
                                 delta_hat, thresholds, trace_every, trace_path)

    # ---- report + figures ------------------------------------------------------
    report = {
        "model": {"parameters": meta["parameters"], "config": meta["config"],
                  "trained": meta["trained"]},
        "data": {"holdout_samples": len(holdout), "banked_transfer_samples": len(banked)},
        "proposals": {"chunks": chunks, "parse_failures": parse_failures,
                      "nothing_to_do": nothing_to_do,
                      "confidence_values": len(conf_values)},
        "coherence_precision_by_progress": coherence,
        "coherence_by_group": {g: round(statistics.mean(v), 4) if v else None
                               for g, v in group_precision.items()},
        "oq1_retrofit_check": oq1,
        "faithfulness_c3": faithfulness,
        "intervenability": intervenability,
        "arm_parity_c5": parity,
        "gate": {"delta_hat": gap, "c_min": c_min,
                 "thresholds": commit.thresholds_to_json(thresholds),
                 "reachability": reachability, "trace": trace_stats},
    }
    out = os.path.join(_RAW, "decoder_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    _figures(conf_values, coherence, trace_stats)

    print(f"\nOQ1 retrofit check: {oq1['verdict']}")
    print(f"  stage A NLL {oq1['stage_a']['holdout_token_nll']} vs "
          f"stage B {oq1['stage_b']['holdout_token_nll']}; "
          f"coherence {oq1['stage_a']['coherence_precision']} vs "
          f"{oq1['stage_b']['coherence_precision']}; "
          f"horizon-4 tau-acc {horizon_4}")
    print(f"coherence precision by group: {report['coherence_by_group']}")
    print(f"C3 rationale mismatch rate: {faithfulness['mismatch_rate']}")
    print(f"intervenability: flip rate {intervenability['flip_rate']}, "
          f"truth alignment {intervenability['forced_truth_alignment']}")
    print(f"gate: delta_hat {delta_hat}s, p*max {reachability['p_star_max']}, "
          f"reachable={reachability['all_reachable']}, c_min {c_min}, "
          f"theta_1 {thresholds.theta_1} slope {thresholds.slope}")
    print(f"-> {os.path.relpath(out, _ROOT)} + proposals + commit trace + 3 figures")
    return 0


class _FakeFused:
    """The minimum fused-record surface control_context() touches when the evidence
    vector is already computed: just the tick. Corpus rows store everything else."""

    def __init__(self, row: dict):
        self.tick = row["tick"]


def _load_banked_fused(directory: str, session_id: str):
    b1 = [json.loads(line) for line in
          open(os.path.join(directory, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")
          if line.strip()]
    b2 = [json.loads(line) for line in
          open(os.path.join(directory, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")
          if line.strip()]
    scored = [record for record in b1 if record.get("scored")]
    return [fuse_dicts(a, b) for a, b in zip(scored, b2)]


def _sweep_c_min(conf_labels: list[tuple[float, bool]]) -> float:
    """The pooled c_min: best F1 of (conf >= c_min) predicting target agreement."""
    best, best_f1 = _C_MIN_GRID[0], -1.0
    for candidate in _C_MIN_GRID:
        true_pos = sum(1 for conf, ok in conf_labels if conf >= candidate and ok)
        false_pos = sum(1 for conf, ok in conf_labels if conf >= candidate and not ok)
        false_neg = sum(1 for conf, ok in conf_labels if conf < candidate and ok)
        denominator = 2 * true_pos + false_pos + false_neg
        f1 = 2 * true_pos / denominator if denominator else 0.0
        if f1 > best_f1:
            best, best_f1 = candidate, f1
    return best


def _mode_entropy(p_z1: float) -> float:
    import math

    if p_z1 <= 0.0 or p_z1 >= 1.0:
        return 0.0
    return -(p_z1 * math.log(p_z1) + (1 - p_z1) * math.log(1 - p_z1))


def _sweep_theta(p_stars, params, delta_hat, c_min):
    """theta_1/slope: the most demanding line that is a-priori reachable AND leaves
    the gate available on validation beliefs. Availability is a validation statistic
    (fraction of steps where the belief clears theta(1)), not the headline metric.
    (theta_suggest is NOT swept here: the OBSERVE/SUGGEST split is the FSM's,
    canonically under discounted-conf semantics — run_gate.py freezes it.)"""
    sweep_rows = []
    chosen = None
    for theta_1, slope in sorted(_THETA_GRID, reverse=True):
        candidate = commit.GateThresholds(
            c_min=c_min, theta_1=theta_1, slope=slope,
            irrev_penalty=_IRREV_PENALTY, m_consecutive=3)
        reach = commit.reachability_report(candidate, params, delta_hat)
        availability = statistics.mean(p >= candidate.theta(1, False) for p in p_stars)
        sweep_rows.append({"theta_1": theta_1, "slope": slope,
                           "a_priori_reachable": reach["all_reachable"],
                           "validation_availability": round(availability, 4)})
        if (chosen is None and reach["all_reachable"]
                and availability >= _MIN_COMMIT_AVAILABILITY):
            chosen = (candidate, reach)
    if chosen is None:
        raise RuntimeError(
            "no theta line in the grid is both a-priori reachable and available on "
            "validation — the staircase would be structurally dead (12-F6); this is "
            "a hard failure, not something to calibrate around")
    return chosen[0], chosen[1], sweep_rows


def _commit_traces(model, holdout, labels, real_sessions, params, delta_hat,
                   thresholds, trace_every, trace_path):
    """K_commit staircase rows over held-out scripted and real sessions — the input
    D5's FSM consumes in part 2. Every Nth correction (cost decision, recorded)."""
    by_session: dict[str, list[dict]] = {}
    for row in holdout:
        by_session.setdefault(row["session"], []).append(row)

    k_values, read_count = [], 0
    with open(trace_path, "w", encoding="utf-8") as handle:
        for session_id, rows in sorted(by_session.items()):
            hysteresis = commit.CommitHysteresis()
            for row in rows[::trace_every]:
                ctx = _context_from_row(row, "arm3")
                proposal, reason = decoder_model.propose(model, ctx, _MAX_ACTIONS)
                if proposal is None:
                    continue
                belief = _belief_from_slot(row["slots"]["arm3"])
                flags = reversibility.prefix_flags(proposal.actions, frozenset())
                raw_k, positions = commit.k_commit(
                    proposal, flags, "arm3", belief, ctx.p_top, params, delta_hat,
                    thresholds)
                held_k = hysteresis.read(raw_k)
                k_values.append(held_k)
                read_count += 1
                handle.write(json.dumps({
                    "session": session_id, "tick": row["tick"], "k": row["k"],
                    "belief_snapshot_id": row["tick"], "raw_k_commit": raw_k,
                    "k_commit": held_k, "per_position": positions,
                    "source": "decoder_holdout"}) + "\n")
        for session_id, meta in sorted(real_sessions.items()):
            fused = _load_banked_fused(_RAW, session_id)
            beliefs = decoder_corpus.replay_beliefs(fused)
            origin = context_builder.build_origin(fused)
            hysteresis = commit.CommitHysteresis()
            for index in range(0, len(fused), trace_every):
                record, belief = fused[index], beliefs[index]
                slot = context_builder.arm3_slot(belief)
                ctx = context_builder.control_context(record, index, "arm3", slot, origin)
                proposal, reason = decoder_model.propose(model, ctx, _MAX_ACTIONS)
                if proposal is None:
                    continue
                flags = reversibility.prefix_flags(proposal.actions, frozenset())
                raw_k, positions = commit.k_commit(
                    proposal, flags, "arm3", belief, slot["p_top"], params, delta_hat,
                    thresholds)
                held_k = hysteresis.read(raw_k)
                k_values.append(held_k)
                read_count += 1
                handle.write(json.dumps({
                    "session": session_id, "tick": record.tick, "k": index,
                    "belief_snapshot_id": record.tick, "raw_k_commit": raw_k,
                    "k_commit": held_k, "per_position": positions,
                    "source": "real"}) + "\n")
    distribution = {str(v): k_values.count(v) for v in sorted(set(k_values))}
    return {"reads": read_count, "trace_every": trace_every,
            "k_commit_distribution": distribution,
            "mean_k_commit": round(statistics.mean(k_values), 3) if k_values else None,
            "note": "real-session breaks classified against an EMPTY human-cell set "
                    "here (offline frame has no origin-relative human map yet); the "
                    "D5 counterfactual gate supplies it in part 2"}


def _belief_from_slot(slot: dict):
    """Rebuild a (goal, mode) belief consistent with a stored arm3 slot: the goal
    marginal split by the slot's mode read. Exact for the propagation the staircase
    needs (the kernels act on the two marginals independently)."""
    from mica.intent.tracker import MODES

    belief = {}
    for goal, mass in zip(GOALS, slot["goal_marginal"]):
        for mode in MODES:
            belief[(goal, mode)] = mass * (slot["p_z1"] if mode == 1 else 1 - slot["p_z1"])
    return belief


def _figures(conf_values, coherence, trace_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    if conf_values:
        axis.hist(conf_values, bins=30, range=(0.0, 1.0))
    axis.set_title("per-action confidence (positions >= 2; agreement-trained)",
                   fontsize=9)
    figure.tight_layout()
    figure.savefig(os.path.join(_RAW, "decoder_conf_hist.png"), dpi=110)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    if coherence:
        bins = sorted(coherence, key=float)
        axis.plot([float(b) for b in bins], [coherence[b] for b in bins], marker="o")
    axis.set_ylim(0.0, 1.05)
    axis.set_title("plan-coherence precision of proposed placements, by progress",
                   fontsize=9)
    axis.set_xlabel("build progress", fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(_RAW, "decoder_coherence.png"), dpi=110)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4))
    distribution = trace_stats["k_commit_distribution"]
    if distribution:
        axis.bar(list(distribution), list(distribution.values()))
    axis.set_title("K_commit across gate reads (hysteresis applied)", fontsize=9)
    axis.set_xlabel("committed prefix length", fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(_RAW, "decoder_commit.png"), dpi=110)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(main())
