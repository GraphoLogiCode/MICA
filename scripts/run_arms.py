"""The four-arm offline comparison — Phase F, the thesis-deciding run.

    python scripts/run_arms.py [--skip-arm2] [--arm2-idle-every N] [--pivots N]

Every arm reads the IDENTICAL fused evidence records, replayed from the banked
artifacts (D0's arm-swap rule) — so any difference in the intent metrics is
attributable to the intent mechanism alone:

  arm0_reactive   structure-only nearest-template argmax per step (the pinned floor;
                  one-hot output, excluded from calibration by honesty)
  arm1_implicit   feedforward per-step goal classifier, no recursion (models/arm1.*)
  arm2_llm        local LLM names goal + stated confidence from the symbolic scene;
                  queried on event-carrying steps + every Nth idle step (default 10,
                  pinned BEFORE any result was seen), held between; disk-cached
  arm3_mica_v1    the belief tracker with the trained Phase-E heads
  arm3_mica_v0    the tracker with hand-coded heads (a named baseline in vault 06)

Session groups keep the honesty visible: scripted (assigned-layout; split holdout vs
train-seen), real (free/template; split never-seen vs tuning-validation vs
train-seen), and stitched PIVOT sessions (in-memory, seeded — the recovery metric's
scenario; excluded from every other aggregate). The pre-registered falsification
criterion (mean separation > 0 before 40% of the build, held-out) is evaluated per
arm and stated either way.

Output: capture/raw/arms_report.json + arms_early_separation.png,
arms_calibration.png, arms_entropy.png. Rerunning is free where it matters: Arm 2
replies come from the cache.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                              # noqa: E402
from mica.capture.pivot_builds import pivot_build                    # noqa: E402
from mica.capture.scripted_goals import plan_variants                # noqa: E402
from mica.capture.synthetic import generate_session                  # noqa: E402
from mica.contracts.b1 import GOALS                                  # noqa: E402
from mica.contracts.b3 import fuse, fuse_dicts, fuse_streams                       # noqa: E402
from mica.data import vlm                                            # noqa: E402
from mica.intent import arm1, heads_v0, heads_v1                     # noqa: E402
from mica.intent.arm2 import Arm2                                    # noqa: E402
from mica.intent.tracker import (                                    # noqa: E402
    TrackerParams, category_marginal, correct, predict, uniform_belief,
)
from mica.perception.evidence2d import evidence_stream               # noqa: E402
from mica.perception.evidence3d import build_evidence3d              # noqa: E402
from mica.perception.voxel_replay import ReplayWorld, region_around_events  # noqa: E402
from mica.validation import intent_metrics                           # noqa: E402
from run_tracker import real_labeled_sessions                        # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTED = os.path.join(_ROOT, "capture", "scripted")
_RAW = os.path.join(_ROOT, "capture", "raw")

_ARM2_IDLE_EVERY = 10        # pinned cadence: query on events + every Nth idle step
_PIVOT_SEED = 21             # distinct from the corpus seed so pivots are fresh builds
_PIVOT_PAIRS = (("habitation", "defense"), ("production", "decorative"),
                ("infrastructure", "habitation"), ("defense", "production"),
                ("decorative", "infrastructure"))

def _exposure_groups() -> tuple[set[str], set[str]]:
    """The real sessions each learned arm has SEEN, read from the model files'
    own exposure provenance (review 13 F2) — never from a hand-maintained set,
    which is how six training-exposed sessions once landed in the 'never seen'
    cell. Both learned arms must agree on the split or the grouping is undefined."""
    exposures = {}
    for name in ("heads_v1", "arm1"):
        path = os.path.join(_ROOT, "models", f"{name}.json")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)["data"]
        if "train_sessions_real" not in data:
            raise SystemExit(
                f"models/{name}.json predates exposure provenance (review 13 F2) - "
                "retrain it (scripts/train_arm1.py / train_heads.py) before running "
                "the four-arm comparison; grouping cannot be trusted without it")
        exposures[name] = (set(data["train_sessions_real"]),
                           {s for s in data.get("validation_sessions",
                                                data["held_out_sessions"])
                            if s.startswith("fabric-")})
    assert exposures["heads_v1"] == exposures["arm1"], (
        "Arm 1 and the v1 heads report different real-session exposure - they no "
        f"longer share the pinned split: {exposures}")
    return exposures["heads_v1"]


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _model_provenance(name: str) -> dict:
    with open(os.path.join(_ROOT, "models", f"{name}.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    return {"file": f"models/{name}.json", "trained": meta["trained"],
            "train_sessions_real": meta["data"].get("train_sessions_real"),
            "knobs": {k: meta[k] for k in ("epsilon", "lambda_g", "lambda_z") if k in meta}}


def _floor_dist(fused) -> dict[str, float]:
    """Arm 0: the structure-only argmax as a one-hot (fit-first key, same as the
    banked floor rows; vault 06 amended 2026-07-16 to pin this exact score — the
    original match-minus-waste form needs template cell counts that do not exist
    at the B3 boundary, and fit-first was the form validated on real sessions)."""
    top = max(fused.per_goal, key=lambda g: (fused.per_goal[g].fit, fused.per_goal[g].comp))
    return {goal: 1.0 if goal == top else 0.0 for goal in GOALS}


class _Tracker:
    """One tracker arm: the exact filter loop every belief run uses."""

    def __init__(self, likelihood, params: TrackerParams):
        self.likelihood = likelihood
        self.params = params
        self.belief = uniform_belief()
        self.previous_tick = 0

    def read(self, fused) -> dict[str, float]:
        dt = max(fused.tick - self.previous_tick, 1) / 20.0
        self.previous_tick = fused.tick
        self.belief = predict(self.belief, dt, self.params)
        self.belief, _ = correct(self.belief, self.likelihood(fused, fused.a_hat), self.params)
        return category_marginal(self.belief)


def _session_steps(fused_records, truths, arms, arm2, idle_every):
    """Replay one session's records through every arm. `truths` gives the true goal
    per record (constant except in pivot sessions). Returns {arm: [step, ...]}."""
    trackers = {name: _Tracker(*spec) for name, spec in arms["trackers"].items()}
    total_events = sum(len(f.event_ids) for f in fused_records) or 1
    consumed = 0
    steps: dict[str, list] = {name: [] for name in arms["all"]}
    held_arm2: dict[str, float] | None = None
    since_query = 0
    for index, fused in enumerate(fused_records):
        progress = consumed / total_events
        consumed += len(fused.event_ids)
        truth = truths[index]
        dists = {"arm0_reactive": _floor_dist(fused)}
        if arms["arm1"]:
            dists["arm1_implicit"] = arm1.distribution(fused)
        if arm2 is not None:
            since_query += 1
            if fused.event_ids or held_arm2 is None or since_query >= idle_every:
                held_arm2 = arm2.read(fused)
                since_query = 0
            dists["arm2_llm"] = held_arm2
        for name, tracker in trackers.items():
            dists[name] = tracker.read(fused)
        for name, dist in dists.items():
            steps[name].append(intent_metrics.step(dist, truth, progress))
    return steps


def _load_banked(directory: str, session_id: str):
    b1 = [json.loads(line) for line in
          open(os.path.join(directory, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")
          if line.strip()]
    b2 = [json.loads(line) for line in
          open(os.path.join(directory, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")
          if line.strip()]
    return fuse_streams(b1, b2, session_id)


def _pivot_records(count: int):
    """Fresh stitched sessions, evidence computed in memory (symbolic channels only,
    like the scripted corpus). Returns [(session_id, fused_records, truths, seam)]."""
    out = []
    for pair_index in range(count):
        goal_a, goal_b = _PIVOT_PAIRS[pair_index % len(_PIVOT_PAIRS)]
        plan_a = plan_variants(goal_a, 1, _PIVOT_SEED + pair_index)[0]
        plan_b = plan_variants(goal_b, 1, _PIVOT_SEED + 100 + pair_index)[0]
        build, truth = pivot_build(plan_a, plan_b)
        session = generate_session(build)
        corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
        events = {e.event_id: e for p in session.packets for e in p.server.block_events}
        world = ReplayWorld(region_around_events(session), {})
        b2 = build_evidence3d(world, corrections, events)
        fused = [fuse(a, b) for a, b in zip(corrections, b2)]
        truths = [truth["goal_a"] if f.tick < truth["seam_tick"] else truth["goal_b"]
                  for f in fused]
        out.append((build.session_id, fused, truths, truth))
    return out


def _aggregate(session_rows: list[dict], pooled_steps: list[dict]) -> dict:
    """Mean the per-session curves and rates for one (group, arm) cell."""
    def curve_mean(key):
        merged = {}
        for b in intent_metrics.PROGRESS_BINS:
            values = [r[key][str(b)] for r in session_rows if r[key][str(b)] is not None]
            merged[str(b)] = round(statistics.mean(values), 4) if values else None
        return merged

    early = [r["early_separation"]["mean_separation"] for r in session_rows
             if r["early_separation"]["mean_separation"] is not None]
    mean_early = round(statistics.mean(early), 4) if early else None
    # ECE is pooled-then-binned (review 13 F8): per-session ECEs on a handful of
    # steps carry a positive noise floor that differs BY ARM, so a mean of them
    # ranks confidence-spread geometry, not calibration. The per-session mean is
    # kept only as a dispersion diagnostic, under a name that cannot be mistaken
    # for ECE.
    pooled_ece = (intent_metrics.reliability(pooled_steps)["ece"]
                  if pooled_steps else None)
    session_eces = [r["reliability"]["ece"] for r in session_rows if r["reliability"]]
    return {
        "sessions": len(session_rows),
        "final_accuracy": round(statistics.mean(r["final_correct"] for r in session_rows), 3),
        "mean_sustained_from": round(statistics.mean(r["sustained_from"] for r in session_rows), 3),
        # the quotable earliness number, in 06's units (fraction of the BUILD placed)
        "mean_sustained_from_progress": round(
            statistics.mean(r["sustained_from_progress"] for r in session_rows), 3),
        "mean_top2": round(statistics.mean(r["top2_accuracy"] for r in session_rows), 3),
        "separation_by_progress": curve_mean("separation_by_progress"),
        "accuracy_by_progress": curve_mean("accuracy_by_progress"),
        "entropy_by_progress": curve_mean("entropy_by_progress"),
        "pooled_ece": pooled_ece,
        "per_session_ece_mean_DISPERSION_ONLY": (
            round(statistics.mean(session_eces), 4) if session_eces else None),
        # The pre-registered criterion, in vault 06's own words (reconciled
        # 2026-07-16, review 13 F7 + user decision): a session counts as an early
        # success iff the top-1 call locks onto the truth AND STAYS there by 40% of
        # the build; the arm passes iff the success count beats 1/|G| chance under
        # an exact one-sided binomial test. Mean margin stays as a diagnostic only.
        "falsification": intent_metrics.falsification_verdict(session_rows, mean_early),
    }


def _figures(report: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [float(b) for b in intent_metrics.PROGRESS_BINS]
    groups = [g for g in ("real_never_seen", "scripted_holdout") if g in report["groups"]]
    arm_names = report["arms"]
    made = []
    if not groups:
        return made

    def per_group_lines(metric, title, filename, hline=None):
        figure, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
        for axis, group in zip(axes[0], groups):
            for arm in arm_names:
                curve = report["groups"][group].get(arm, {}).get(metric)
                if not curve:
                    continue
                ys = [curve[str(b)] for b in intent_metrics.PROGRESS_BINS]
                axis.plot(bins, ys, marker="o", markersize=3, label=arm)
            if hline is not None:
                axis.axhline(hline, color="black", linewidth=0.6, alpha=0.5)
            axis.axvline(intent_metrics.FALSIFICATION_PROGRESS, color="red",
                         linewidth=0.6, alpha=0.5, linestyle=":")
            axis.set_title(f"{title} — {group}", fontsize=9)
            axis.set_xlabel("build progress", fontsize=8)
            axis.tick_params(labelsize=7)
            axis.legend(fontsize=6)
        figure.tight_layout()
        out = os.path.join(_RAW, filename)
        figure.savefig(out, dpi=110)
        plt.close(figure)
        made.append(out)

    per_group_lines("separation_by_progress",
                    "early separation: P(truth) − nearest competitor",
                    "arms_early_separation.png", hline=0.0)
    per_group_lines("entropy_by_progress", "goal-distribution entropy (nats)",
                    "arms_entropy.png")

    figure, axis = plt.subplots(figsize=(6, 4))
    group = groups[0] if groups else None
    if group:
        for arm in arm_names:
            rel = report["groups"][group].get(arm, {}).get("pooled_ece")
            if rel is not None:
                axis.bar(arm, rel)
        axis.set_title(f"pooled ECE per arm — {group} (lower = better calibrated)", fontsize=9)
        axis.tick_params(axis="x", labelsize=6, rotation=20)
    figure.tight_layout()
    out = os.path.join(_RAW, "arms_calibration.png")
    figure.savefig(out, dpi=110)
    plt.close(figure)
    made.append(out)
    return made


def main() -> int:
    skip_arm2 = "--skip-arm2" in sys.argv
    idle_every = _flag_value("--arm2-idle-every", _ARM2_IDLE_EVERY)
    pivot_count = _flag_value("--pivots", 5)

    if not skip_arm2 and vlm.backend() == "none":
        print("Arm 2 needs the local Ollama server (or --skip-arm2); neither is available.")
        return 1
    if not arm1.available():
        print("Arm 1 not trained — run scripts/train_arm1.py first")
        return 1
    arm2 = None if skip_arm2 else Arm2()

    trackers = {"arm3_mica_v0": (heads_v0.likelihood, TrackerParams())}
    if heads_v1.available():
        trackers["arm3_mica_v1"] = (heads_v1.likelihood, heads_v1.tracker_params())
    arms = {"trackers": trackers, "arm1": True,
            "all": (["arm0_reactive", "arm1_implicit"]
                    + (["arm2_llm"] if arm2 else []) + list(trackers))}

    with open(os.path.join(_ROOT, "models", "arm1.json"), encoding="utf-8") as handle:
        arm1_meta = json.load(handle)
    scripted_holdout = {s for s in arm1_meta["data"]["held_out_sessions"]
                        if not s.startswith("fabric-")}

    with open(os.path.join(_SCRIPTED, "labels.json"), encoding="utf-8") as handle:
        scripted_labels = json.load(handle)

    groups: dict[str, dict[str, list]] = {}
    pooled: dict[str, dict[str, list]] = {}    # raw steps per (group, arm) — pooled ECE
    per_session: dict[str, dict] = {}
    skipped: list[str] = []

    def run_one(group: str, session_id: str, fused, truths):
        if not fused:
            # zero scored corrections: record and skip — one degenerate capture must
            # never abort the whole comparison run (review 13 F17/F21)
            skipped.append(session_id)
            per_session[session_id] = {"group": group, "skipped": "no scored corrections"}
            print(f"  {session_id} SKIPPED (no scored corrections)")
            return None
        steps = _session_steps(fused, truths, arms, arm2, idle_every)
        per_session[session_id] = {"group": group}
        for arm_name, arm_steps in steps.items():
            metrics = intent_metrics.session_metrics(
                arm_steps, calibrated=arm_name != "arm0_reactive")
            per_session[session_id][arm_name] = {
                "final_correct": metrics["final_correct"],
                "sustained_from": metrics["sustained_from"]}
            groups.setdefault(group, {}).setdefault(arm_name, []).append(metrics)
            if arm_name != "arm0_reactive":    # one-hot floor stays out of calibration
                pooled.setdefault(group, {}).setdefault(arm_name, []).extend(arm_steps)
        return steps

    print("scripted corpus (assigned layout) ...")
    for session_id, label in sorted(scripted_labels.items()):
        fused = _load_banked(_SCRIPTED, session_id)
        group = "scripted_holdout" if session_id in scripted_holdout else "scripted_train_seen"
        run_one(group, session_id, fused, [label["goal"]] * len(fused))
        print(f"  {session_id} [{label['goal']}] ({group})")

    print("real captures (free choice / template) ...")
    train_seen, validation_set = _exposure_groups()
    for session_id, meta in real_labeled_sessions().items():
        fused = _load_banked(session_store.session_dir(session_id), session_id)
        group = ("real_train_seen" if session_id in train_seen
                 else "real_validation" if session_id in validation_set
                 else "real_never_seen")
        run_one(group, session_id, fused, [meta["goal"]] * len(fused))
        print(f"  {session_id} [{meta['goal']}] ({group})")

    print(f"pivot scenarios ({pivot_count} stitched sessions, seed {_PIVOT_SEED}) ...")
    pivot_rows = []
    for session_id, fused, truths, truth in _pivot_records(pivot_count):
        steps = run_one("pivot", session_id, fused, truths)
        pivot_index = next((i for i, t in enumerate(truths) if t == truth["goal_b"]), None)
        if steps is None or pivot_index is None or pivot_index == 0:
            # a seam with no records on one side: skip-and-record, never fake a
            # recovery row (review 13 F23)
            pivot_rows.append({"session": session_id, "skipped": "degenerate seam",
                               "goal_a": truth["goal_a"], "goal_b": truth["goal_b"]})
            print(f"  {session_id} SKIPPED (degenerate seam)")
            continue
        row = {"session": session_id, "goal_a": truth["goal_a"], "goal_b": truth["goal_b"]}
        for arm_name, arm_steps in steps.items():
            row[arm_name] = intent_metrics.recovery_after_pivot(arm_steps, pivot_index)
        pivot_rows.append(row)
        print(f"  {session_id}")

    report = {
        "arms": arms["all"],
        "provenance": {
            "arm1": {k: arm1_meta[k] for k in ("temperature", "trained", "best_val_nll")},
            "arm2": (None if arm2 is None else
                     {"model": vlm.describe(), "cadence":
                      f"events + every {idle_every}th idle step, held between",
                      "queries": arm2.queries, "cache_hits": arm2.cache_hits,
                      "unusable_replies": arm2.unusable}),
            # provenance is READ from the model file, never hardcoded — a stale
            # training-date string once masked the exposure staleness (review 13 F34)
            "arm3_v1": (_model_provenance("heads_v1")
                        if heads_v1.available() else None),
            "arm1_training_history": [
                "config 1 (256/64 hidden, decay 1e-4, lr 1e-3): best held-out NLL at "
                "epoch 0, diverged to 9.6 — memorized train sessions immediately",
                "config 2 SHIPPED (64/32, dropout 0.3, decay 1e-2, lr 3e-4): best "
                "1.6133 vs uniform 1.6094; T=5.6 flattens to 1.6092 — the per-step "
                "no-recursion mechanism generalizes at chance on this corpus; "
                "reported as the mechanism's result, not a strawman (review F5)",
            ],
            "caveats": [
                "10 real sessions, lopsided categories (defense/production single-session)",
                "real_train_seen sessions were in Arm 1 + v1-heads training; "
                "real_validation tuned their temperatures and filter knobs",
                "arm0 is one-hot by construction and excluded from calibration",
            ],
        },
        "groups": {group: {arm_name: _aggregate(
                               rows, pooled.get(group, {}).get(arm_name, []))
                           for arm_name, rows in arm_rows.items()}
                   for group, arm_rows in groups.items() if group != "pivot"},
        "pivot_recovery": pivot_rows,
        "per_session": per_session,
        "skipped_sessions": skipped,
    }
    out = os.path.join(_RAW, "arms_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    figures = _figures(report)

    print("\nfour-arm comparison — held-out groups (lower sustained-from = earlier):")
    for group in ("scripted_holdout", "real_never_seen", "real_validation"):
        if group not in report["groups"]:
            continue
        print(f"  {group}:")
        for arm_name in arms["all"]:
            cell = report["groups"][group].get(arm_name)
            if not cell:
                continue
            falsification = cell["falsification"]
            print(f"    {arm_name:<16} acc {cell['final_accuracy']:.3f}"
                  f"  sustained@build {cell['mean_sustained_from_progress']:.3f}"
                  f"  locked-by-40% {falsification['locked_by_040']}/{falsification['sessions']}"
                  f" (p={falsification['p_value']})"
                  f"  [{'PASS' if falsification['passes'] else 'FAIL'}]"
                  + (f"  pooled-ECE {cell['pooled_ece']}"
                     if cell["pooled_ece"] is not None else ""))
    print(f"  -> {os.path.relpath(out, _ROOT)} + {len(figures)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
