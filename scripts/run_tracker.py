"""Run the belief tracker (hand-coded heads v0) over the scripted corpus — twice.

    python scripts/run_tracker.py [--real]

Reads capture/scripted (labels + evidence jsonl; run make_scripted_corpus.py first).
Every correction's evidence is built through the VERIFIED B3 handoff (contracts/b3.py:
the B1 and B2 records must be the same step or fusion refuses), then the tracker runs
two arms over identical sessions:

  fused     — the full B3 bundle: D2 structure + D1 behavior (held item, action streak,
              crosshair dwell, s_goal when pixels exist)
  stripped  — the same records with the 2D behavior channels blanked (strip_behavior),
              i.e. structure + action label only

The gap between the arms is the D1 stream's measured contribution — the with/without
probe the D1 design prescribes, at the belief level. Also reports the structure-only
floor and checks the structural invariants at every step. Writes belief_summary.json.

--real (the 12-F3 rerun) runs the identical three-arm comparison over the labeled REAL
captures instead: evidence from capture/raw's banked files, truth = the builder's own
category in labels.json (the synthetic corpus pre-equips the exactly-right block from
20 ticks out, so its held-item cue is noiseless by construction — the scripted numbers
are an upper bound, and this run is the measurement that decides what the behavior
channels are worth on real play). The quarantined session is excluded; the scripted
artifact is untouched. Writes capture/raw/belief_summary_real.json.

--heads v1 swaps in the TRAINED heads (models/heads_v1.*, from scripts/train_heads.py)
with their jointly-fitted temperatures and filter knobs. The output gets a _v1 suffix
so the v0 baselines — including the byte-identical scripted belief_summary.json the
golden gate regenerates — are never overwritten.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                       # noqa: E402
from mica.contracts.b3 import fuse_dicts, fuse_streams                     # noqa: E402
from mica.intent.heads_v0 import likelihood, strip_behavior, strip_structure  # noqa: E402
from mica.intent.tracker import (                            # noqa: E402
    TrackerParams, category_marginal, correct, mode_marginal, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORPUS = os.path.join(_ROOT, "capture", "scripted")
_RAW = os.path.join(_ROOT, "capture", "raw")

# Belt-and-suspenders quarantine pin: the PRIMARY exclusion mechanism is each
# session's own report (structure_quarantined, read below, added 2026-07-12) —
# this set only guards sessions whose report was lost, so a deleted file can
# never silently re-admit known-bad evidence. (210003: capture pauses dropped
# ~130 placements; its replayed world fails the proof check permanently.)
_QUARANTINED = {"fabric-20260704-210003"}


def _report_quarantined(session_id: str) -> bool:
    """The session's own report says its structure evidence is quarantined."""
    path = os.path.join(session_store.session_dir(session_id), "session_report.json")
    try:
        with open(path, encoding="utf-8") as handle:
            return bool(json.load(handle).get("structure_quarantined"))
    except (OSError, ValueError):
        return False


def real_labeled_sessions() -> dict[str, dict]:
    """The labeled real captures a belief run may score: every labels.json entry whose
    banked evidence files exist, minus quarantined ones. Truth is the BUILDER's own
    category — matcher-contested sessions stay in (the builder's word outranks the
    matcher; the contest rule governs training pairs, not evaluation truth).

    Quarantine is read from each session's OWN report — a newly quarantined session
    leaves every eval the moment its report says so, no constant to edit."""
    with open(os.path.join(_RAW, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    sessions = {}
    for session_id, label in labels.items():
        if session_id in _QUARANTINED or _report_quarantined(session_id):
            print(f"  excluded (structure quarantined): {session_id}")
            continue
        b1 = session_store.session_file(session_id, ".evidence2d.jsonl")
        b2 = session_store.session_file(session_id, ".evidence3d.jsonl")
        if os.path.exists(b1) and os.path.exists(b2):
            sessions[session_id] = {"goal": label["goal"], "mode": label.get("mode", ""),
                                    "b1": b1, "b2": b2}
    return sessions


def _sustained_from(calls: list[str], truth: str) -> float:
    """The earliest progress fraction from which every call is the true category."""
    start = len(calls)
    for index in range(len(calls) - 1, -1, -1):
        if calls[index] != truth:
            break
        start = index
    return start / len(calls) if calls else 1.0


def _track(fused_records, truth: str, params: TrackerParams, trace: list | None = None) -> dict:
    from mica.intent.tracker import entropy

    belief = uniform_belief()
    previous_tick = 0
    calls, z1 = [], []
    for fused in fused_records:
        dt = max(fused.tick - previous_tick, 1) / 20.0
        previous_tick = fused.tick
        belief = predict(belief, dt, params)
        belief, normalizer = correct(belief, likelihood(fused, fused.a_hat), params)
        assert abs(sum(belief.values()) - 1.0) < 1e-9, "belief mass drifted"
        assert normalizer >= params.epsilon / params.action_count - 1e-15, "floor violated"
        marginal = category_marginal(belief)
        calls.append(max(marginal, key=marginal.get))
        z1.append(mode_marginal(belief)[1])
        if trace is not None:
            trace.append({"tick": fused.tick, "marginal": {g: round(p, 4) for g, p in marginal.items()},
                          "p_top": round(max(marginal.values()), 4),
                          "p_truth": round(marginal[truth], 4),
                          "entropy": round(entropy(belief), 4),
                          "p_heuristic": round(mode_marginal(belief)[1], 4)})
    return {"final_correct": calls[-1] == truth,
            "sustained_from": round(_sustained_from(calls, truth), 3),
            "mean_p_heuristic": round(statistics.mean(z1), 3)}


def _plot_traces(traces: dict, directory: str, suffix: str) -> str:
    """One panel per session: does the belief find and hold the truth? p_truth is the
    category the builder actually pursued; p_top is whatever leads; entropy (scaled to
    [0,1]) falling means the belief is committing to something."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_entropy = math.log(10)                   # ten (goal, mode) states
    columns = 2
    rows = (len(traces) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(11, 2.6 * rows), squeeze=False)
    for axis, (session_id, (truth, rows_)) in zip(
            (a for row in axes for a in row), sorted(traces.items())):
        steps = range(len(rows_))
        axis.plot(steps, [r["p_truth"] for r in rows_], color="tab:green", label="p(truth)")
        axis.plot(steps, [r["p_top"] for r in rows_], color="tab:gray",
                  linestyle="--", linewidth=0.8, label="p(top)")
        axis.plot(steps, [r["entropy"] / max_entropy for r in rows_], color="tab:red",
                  linestyle=":", linewidth=0.8, label="entropy/max")
        axis.axhline(0.2, color="black", linewidth=0.5, alpha=0.4)   # uniform-category line
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"{session_id}  [{truth}]", fontsize=8)
        axis.tick_params(labelsize=7)
    for axis in list(a for row in axes for a in row)[len(traces):]:
        axis.axis("off")
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels_, loc="lower right", fontsize=7)
    figure.suptitle(f"belief traces ({suffix} heads) — corrections vs marginals", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(directory, f"belief_traces_{suffix}.png")
    figure.savefig(out, dpi=110)
    plt.close(figure)
    return out


def main() -> int:
    global likelihood
    real = "--real" in sys.argv
    v1 = "--heads" in sys.argv and sys.argv[sys.argv.index("--heads") + 1] == "v1"
    if v1:
        from mica.intent import heads_v1
        likelihood = heads_v1.likelihood
    if real:
        labels = real_labeled_sessions()
        if not labels:
            print("no labeled real captures with banked evidence found in capture/raw")
            return 1
    else:
        labels_path = os.path.join(_CORPUS, "labels.json")
        if not os.path.exists(labels_path):
            print("no corpus found - run scripts/make_scripted_corpus.py first")
            return 1
        with open(labels_path, encoding="utf-8") as handle:
            labels = json.load(handle)
    if v1:
        from mica.intent import heads_v1
        params = heads_v1.tracker_params()   # the jointly-fitted knobs, or none of it
    else:
        params = TrackerParams()

    results = []
    traces: dict[str, tuple[str, list]] = {}
    for session_id, label in labels.items():
        directory = session_store.session_dir(session_id) if real else _CORPUS
        b1 = [json.loads(line) for line in
              open(os.path.join(directory, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")]
        b2 = [json.loads(line) for line in
              open(os.path.join(directory, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")]
        fused = fuse_streams(b1, b2, session_id)   # the handoff: counts asserted, verified per record

        trace: list | None = [] if "--trace" in sys.argv else None
        floor_calls = [max(f.per_goal, key=lambda g: (f.per_goal[g].fit, f.per_goal[g].comp))
                       for f in fused]
        entry = {"session": session_id, "goal": label["goal"], "mode": label["mode"],
                 "fused": _track(fused, label["goal"], params, trace),
                 "no_behavior": _track([strip_behavior(f) for f in fused], label["goal"], params),
                 "no_structure": _track([strip_structure(f) for f in fused], label["goal"], params),
                 "floor_sustained_from": round(_sustained_from(floor_calls, label["goal"]), 3),
                 "floor_final_correct": floor_calls[-1] == label["goal"]}
        results.append(entry)
        if trace is not None:
            suffix = "v1" if v1 else "v0"
            trace_path = os.path.join(directory, f"{session_id}.belief_trace_{suffix}.jsonl")
            with open(trace_path, "w", encoding="utf-8") as handle:
                for row in trace:
                    handle.write(json.dumps(row) + "\n")
            traces[session_id] = (label["goal"], trace)

    def arm(name):
        return (statistics.mean(r[name]["final_correct"] for r in results),
                statistics.mean(r[name]["sustained_from"] for r in results))

    fused_acc, fused_early = arm("fused")
    no_b_acc, no_b_early = arm("no_behavior")
    no_s_acc, no_s_early = arm("no_structure")
    floor_acc = statistics.mean(r["floor_final_correct"] for r in results)
    floor_early = statistics.mean(r["floor_sustained_from"] for r in results)

    summary = {
        "params": {"lambda_g": params.lambda_g, "lambda_z": params.lambda_z,
                   "epsilon": params.epsilon,
                   "heads": "trained v1" if v1 else "hand-coded v0"},
        "data": ("labeled real captures (truth = builder category; quarantined excluded)"
                 if real else "scripted corpus (generator ground truth)"),
        "sessions": len(results),
        "fused": {"final_accuracy": round(fused_acc, 3), "mean_sustained_from": round(fused_early, 3)},
        "no_behavior_3d_only": {"final_accuracy": round(no_b_acc, 3),
                                "mean_sustained_from": round(no_b_early, 3)},
        "no_structure_2d_only": {"final_accuracy": round(no_s_acc, 3),
                                 "mean_sustained_from": round(no_s_early, 3)},
        "structure_only_floor": {"final_accuracy": round(floor_acc, 3),
                                 "mean_sustained_from": round(floor_early, 3)},
        "per_session": results,
    }
    name = f"belief_summary{'_real' if real else ''}{'_v1' if v1 else ''}.json"
    out = os.path.join(_RAW if real else _CORPUS, name)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"belief tracker ({'trained heads v1' if v1 else 'hand-coded heads v0'})"
          f" over {len(results)} {'REAL' if real else 'scripted'} sessions - three arms")
    print(f"  final accuracy:   fused {fused_acc:.3f}   3D-only {no_b_acc:.3f}"
          f"   2D-only {no_s_acc:.3f}   floor {floor_acc:.3f}")
    print(f"  sustained from:   fused {fused_early:.3f}   3D-only {no_b_early:.3f}"
          f"   2D-only {no_s_early:.3f}   floor {floor_early:.3f}   (lower = earlier)")
    print(f"  3D structure contribution (vs 2D-only): {no_s_early - fused_early:+.3f} earliness,"
          f" {fused_acc - no_s_acc:+.3f} accuracy")
    print(f"  2D behavior contribution (vs 3D-only):  {no_b_early - fused_early:+.3f} earliness,"
          f" {fused_acc - no_b_acc:+.3f} accuracy")
    print(f"  handoff verified per record; invariants held every step  ->  {os.path.basename(out)}")
    if traces:
        plot = _plot_traces(traces, _RAW if real else _CORPUS, "v1" if v1 else "v0")
        print(f"  belief traces written per session (+ {os.path.basename(plot)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
