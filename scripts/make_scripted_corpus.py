"""Generate the Source-A scripted corpus and measure whether it is diverse enough.

    python scripts/make_scripted_corpus.py [--per-goal N] [--seed S]

Writes to capture/scripted/: per session an evidence2d.jsonl (symbolic channels) and
evidence3d.jsonl, plus labels.json (the ground truth the generator knew) and
corpus_diversity_report.json. Everything is seeded — same seed, same corpus.

The diversity report answers the memorization question with numbers:
  - every session's action sequence must be unique (no two identical scripts)
  - poses, orders, pacing, modes, and materials must actually spread
  - the early-identifiability curve: structure-only goal accuracy at each fraction of
    build progress. If it starts near chance and climbs, early prefixes are genuinely
    ambiguous and earliness is something a model must LEARN; if it starts near 1.0,
    the corpus is trivially separable and would prove nothing.

Scripted data has no pixels (h2d/s_goal stay None) and one "builder" (the generator) —
it bootstraps the heads and calibrates the machinery. The generalization claim rests on
real free builds (Source B) and the five real in-game template captures for the pixel
probe; this corpus is the controlled floor under them.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.scripted_goals import build_from_plan, plan_variants   # noqa: E402
from mica.capture.synthetic import generate_session                      # noqa: E402
from mica.contracts.b1 import GOALS                                      # noqa: E402
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict  # noqa: E402
from mica.perception.evidence2d import evidence_stream                   # noqa: E402
from mica.perception.evidence3d import build_evidence3d                  # noqa: E402
from mica.perception.voxel_replay import ReplayWorld, region_around_events  # noqa: E402
from mica.validation.evidence2d_check import check_stream as check_b1    # noqa: E402
from mica.validation.evidence3d_check import check_join, check_stream as check_b2  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "scripted")


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _structure_guess(record) -> str:
    """The structure-only baseline's call: best player-anchored fit, completion breaking
    ties — no behavior stream, no belief, no memory. What a model must beat, and the
    yardstick for how identifiable each moment of the corpus already is."""
    return max(record.per_goal, key=lambda g: (record.per_goal[g].fit, record.per_goal[g].comp))


def _generate(per_goal: int, seed: int):
    sessions = []
    for goal in GOALS:
        for plan in plan_variants(goal, per_goal, seed):
            build, label = build_from_plan(plan)
            session = generate_session(build)
            b1 = tuple(evidence_stream(session.packets))
            corrections = tuple(r for r in b1 if r.scored)
            events = {e.event_id: e for p in session.packets for e in p.server.block_events}
            world = ReplayWorld(region_around_events(session), {})
            b2 = build_evidence3d(world, corrections, events)
            problems = check_b1(list(b1)) + check_b2(b2) + check_join(list(corrections), b2)
            if problems:
                raise RuntimeError(f"{build.session_id}: contract violation: {problems[0]}")
            sessions.append((build, label, b1, b2))
    return sessions


def _write_corpus(sessions) -> None:
    os.makedirs(_OUT, exist_ok=True)
    labels = {}
    for build, label, b1, b2 in sessions:
        labels[build.session_id] = label
        with open(os.path.join(_OUT, f"{build.session_id}.evidence2d.jsonl"), "w",
                  encoding="utf-8") as handle:
            for record in b1:
                handle.write(json.dumps(evidence2d_to_dict(record)) + "\n")
        with open(os.path.join(_OUT, f"{build.session_id}.evidence3d.jsonl"), "w",
                  encoding="utf-8") as handle:
            for record in b2:
                handle.write(json.dumps(evidence3d_to_dict(record)) + "\n")
    with open(os.path.join(_OUT, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)


def _diversity_report(sessions) -> dict:
    order_hashes = set()
    per_goal: dict[str, dict] = {}
    gaps: list[int] = []
    for build, label, _, _ in sessions:
        sequence = ",".join(f"{p.pos.x},{p.pos.y},{p.pos.z},{p.op.value}" for p in build.placements)
        order_hashes.add(hashlib.sha256(sequence.encode()).hexdigest()[:12])
        stats = per_goal.setdefault(label["goal"], {
            "sessions": 0, "subtypes": set(), "rotations": set(), "origins": set(),
            "orders": set(), "modes": Counter(), "materials": set(), "break_events": 0,
            "completions": [],
        })
        stats["sessions"] += 1
        stats["subtypes"].add(label["subtype"])
        stats["rotations"].add(label["rotation"])
        stats["origins"].add(tuple(label["origin"]))
        stats["orders"].add(label["order"])
        stats["modes"][label["mode"]] += 1
        stats["materials"].add(label["block"])
        stats["break_events"] += sum(1 for p in build.placements if p.op.value == "break")
        stats["completions"].append(label["completion"])
        ticks = [p.tick for p in build.placements]
        gaps.extend(b - a for a, b in zip(ticks, ticks[1:]) if b > a)

    # the memorization question, measured: structure-only accuracy along build progress
    bins = [round(0.1 * b, 1) for b in range(1, 11)]
    curve = {}
    for target in bins:
        correct = 0
        for _, label, _, b2 in sessions:
            event_records = [r for r in b2 if r.event_ids]
            index = min(len(event_records) - 1, max(0, round(target * len(event_records)) - 1))
            correct += _structure_guess(event_records[index]) == label["goal"]
        curve[str(target)] = round(correct / len(sessions), 3)

    # the fine layer: does the winning instance name the labeled style at build's end?
    style_correct = 0
    for _, label, _, b2 in sessions:
        final = [r for r in b2 if r.event_ids][-1]
        style_correct += final.per_goal[label["goal"]].subtype == label["subtype"]

    return {
        "sessions": len(sessions),
        "unique_action_sequences": len(order_hashes),
        "gap_ticks_mean": round(statistics.mean(gaps), 1),
        "gap_ticks_sd": round(statistics.stdev(gaps), 1),
        "style_accuracy_at_end": round(style_correct / len(sessions), 3),
        "per_goal": {
            goal: {
                "sessions": s["sessions"],
                "subtypes": sorted(s["subtypes"]),
                "rotations": sorted(s["rotations"]),
                "distinct_origins": len(s["origins"]),
                "orders": sorted(s["orders"]),
                "modes": dict(s["modes"]),
                "materials": sorted(s["materials"]),
                "break_events": s["break_events"],
                "completion_range": [min(s["completions"]), max(s["completions"])],
            }
            for goal, s in per_goal.items()
        },
        "structure_only_accuracy_by_progress": curve,
        "chance_level": round(1 / len(GOALS), 3),
    }


def main() -> int:
    # Defaults must reproduce the banked corpus exactly (30 sessions = 6 per goal,
    # seed 7): "regenerable by one command" means THIS command with no flags.
    per_goal = _flag_value("--per-goal", 6)
    seed = _flag_value("--seed", 7)
    sessions = _generate(per_goal, seed)
    _write_corpus(sessions)
    report = _diversity_report(sessions)
    report_path = os.path.join(_OUT, "corpus_diversity_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"scripted corpus: {report['sessions']} sessions -> {os.path.relpath(_OUT, _ROOT)}")
    print(f"  unique action sequences: {report['unique_action_sequences']}/{report['sessions']}")
    print(f"  placement gap ticks: mean {report['gap_ticks_mean']} sd {report['gap_ticks_sd']}")
    for goal, s in report["per_goal"].items():
        print(f"  {goal}: {s['sessions']} sessions, styles {s['subtypes']},"
              f" rotations {s['rotations']}, {s['distinct_origins']} origins,"
              f" orders {s['orders']}, modes {s['modes']},"
              f" {s['break_events']} breaks, completion {s['completion_range']}")
    print(f"  style (subtype) accuracy at build end: {report['style_accuracy_at_end']}")
    print(f"  structure-only accuracy by progress (chance {report['chance_level']}):")
    print("    " + "  ".join(f"{k}:{v}" for k, v in report["structure_only_accuracy_by_progress"].items()))
    print(f"  report: {os.path.basename(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
