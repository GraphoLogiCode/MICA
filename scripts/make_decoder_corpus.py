"""Generate the D4 decoder corpus: (control context -> helper trace) samples.

    python scripts/make_decoder_corpus.py [--per-goal N] [--seed S] [--skip-arm2]

Writes capture/decoder/: per session a <id>.decoder.jsonl of samples, plus
decoder_labels.json (session ground truth + split group) and
decoder_corpus_report.json (provenance). NEVER touches capture/scripted — the
perception corpus and the decoder corpus are separate artifacts with separate seeds.

Three session groups land in the corpus:
  train           fresh scripted sessions (defaults: 24 per goal, seed 11)
  holdout         the last fresh session per goal (sorted ids) — never trained on
  banked_holdout  the perception corpus's own holdout sessions, targets rebuilt from
                  their labels (eval-only: proves the decoder transfers to sessions
                  generated under a different seed)

Every sample's belief slot comes from a real tracker replay (trained v1 heads); the
arm1 slot from the Phase-F classifier; the arm2 slot from the cached LLM reader over
a PINNED coverage subset (querying every session would cost hours of LLM time for
augmentation variety it does not need — coverage sessions are recorded in the
report, and reruns hit the cache). Defaults reproduce the banked corpus exactly:
"regenerable by one command" means THIS command with no flags.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.scripted_goals import build_from_plan, plan_variants   # noqa: E402
from mica.capture.synthetic import generate_session                      # noqa: E402
from mica.contracts.b1 import GOALS                                      # noqa: E402
from mica.contracts.b3 import fuse, fuse_dicts                           # noqa: E402
from mica.data import decoder_corpus, vlm                                # noqa: E402
from mica.intent import arm1, heads_v1                                   # noqa: E402
from mica.perception.evidence2d import evidence_stream                   # noqa: E402
from mica.perception.evidence3d import build_evidence3d                  # noqa: E402
from mica.perception.voxel_replay import ReplayWorld, region_around_events  # noqa: E402
from mica.validation.evidence2d_check import check_stream as check_b1    # noqa: E402
from mica.validation.evidence3d_check import check_join, check_stream as check_b2  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = decoder_corpus.CORPUS_DIR
_SCRIPTED = os.path.join(_ROOT, "capture", "scripted")

_PER_GOAL = 24     # pinned defaults — the banked decoder corpus is exactly this
_SEED = 11         # distinct from the perception corpus's seed 7 on purpose


def _flag_value(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _fused_for_build(build):
    """One generated session's verified fused records (same path as the perception
    corpus generator, contract checks included)."""
    session = generate_session(build)
    b1 = tuple(evidence_stream(session.packets))
    corrections = tuple(r for r in b1 if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    world = ReplayWorld(region_around_events(session), {})
    b2 = build_evidence3d(world, corrections, events)
    problems = check_b1(list(b1)) + check_b2(b2) + check_join(list(corrections), b2)
    if problems:
        raise RuntimeError(f"{build.session_id}: contract violation: {problems[0]}")
    return [fuse(a, b) for a, b in zip(corrections, b2)]


def _load_banked_fused(session_id: str):
    b1 = [json.loads(line) for line in
          open(os.path.join(_SCRIPTED, f"{session_id}.evidence2d.jsonl"), encoding="utf-8")
          if line.strip()]
    b2 = [json.loads(line) for line in
          open(os.path.join(_SCRIPTED, f"{session_id}.evidence3d.jsonl"), encoding="utf-8")
          if line.strip()]
    scored = [record for record in b1 if record.get("scored")]
    return [fuse_dicts(a, b) for a, b in zip(scored, b2)]


def _banked_holdout_ids() -> list[str]:
    """The perception corpus's holdout sessions — the same set every Phase-F group
    used, read from the same place run_arms reads it."""
    with open(os.path.join(_ROOT, "models", "arm1.json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    return sorted(s for s in meta["data"]["held_out_sessions"] if not s.startswith("fabric-"))


def main() -> int:
    per_goal = _flag_value("--per-goal", _PER_GOAL)
    seed = _flag_value("--seed", _SEED)
    skip_arm2 = "--skip-arm2" in sys.argv

    if not heads_v1.available():
        print("decoder corpus needs the trained v1 heads — run scripts/train_heads.py first")
        return 1
    if not arm1.available():
        print("decoder corpus needs the trained Arm 1 — run scripts/train_arm1.py first")
        return 1

    arm2_reader = None
    arm2_note = "skipped by flag" if skip_arm2 else None
    if not skip_arm2:
        if vlm.backend() == "none":
            arm2_note = ("Ollama unavailable — arm2 slots absent; training falls back "
                         "to three-way slot ratios (deviation recorded)")
            print(f"NOTE: {arm2_note}")
        else:
            from mica.intent.arm2 import Arm2
            arm2_reader = Arm2()

    # ---- the fresh sessions -------------------------------------------------
    generated: dict[str, dict] = {}
    builds = {}
    for goal in GOALS:
        for plan in plan_variants(goal, per_goal, seed):
            build, label = build_from_plan(plan)
            builds[build.session_id] = (build, label)
    holdout_ids = {sorted(sid for sid, (_, lbl) in builds.items() if lbl["goal"] == goal)[-1]
                   for goal in GOALS}
    # Arm 2 coverage: every holdout session + the first two train sessions per goal.
    coverage = set(holdout_ids)
    for goal in GOALS:
        train_ids = sorted(sid for sid, (_, lbl) in builds.items()
                           if lbl["goal"] == goal and sid not in holdout_ids)
        coverage.update(train_ids[:2])
    banked_ids = _banked_holdout_ids()
    coverage_banked = set(banked_ids)

    os.makedirs(_OUT, exist_ok=True)
    labels: dict[str, dict] = {}
    sample_counts, target_lengths = [], []
    empty_targets, skipped_placements = 0, 0

    def write_session(session_id: str, label: dict, fused, placements, group: str,
                      covered: bool):
        nonlocal skipped_placements
        skipped_placements += len(decoder_corpus.skipped_place_ids(placements))
        rows = decoder_corpus.session_rows(
            session_id, label, fused, placements,
            arm2_reader=arm2_reader if covered else None)
        with open(os.path.join(_OUT, f"{session_id}.decoder.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        labels[session_id] = dict(label, group=group)
        sample_counts.append(len(rows))
        for row in rows:
            target_lengths.append(len(row["target"]))
        nonlocal empty_targets
        empty_targets += sum(1 for row in rows if not row["target"])
        print(f"  {session_id} [{label['goal']}] ({group}) "
              f"{len(rows)} samples{' +arm2' if covered and arm2_reader else ''}")

    print(f"decoder corpus: generating {per_goal * len(GOALS)} sessions (seed {seed}) ...")
    for session_id in sorted(builds):
        build, label = builds[session_id]
        fused = _fused_for_build(build)
        group = "holdout" if session_id in holdout_ids else "train"
        write_session(session_id, label, fused, build.placements, group,
                      session_id in coverage)

    print("banked perception-corpus holdout (eval-only transfer group) ...")
    with open(os.path.join(_SCRIPTED, "labels.json"), encoding="utf-8") as handle:
        scripted_labels = json.load(handle)
    for session_id in banked_ids:
        label = scripted_labels[session_id]
        placements = decoder_corpus.regenerate_placements(label)
        fused = _load_banked_fused(session_id)
        write_session(session_id, label, fused, placements, "banked_holdout",
                      session_id in coverage_banked)

    with open(os.path.join(_OUT, "decoder_labels.json"), "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=2)

    report = {
        "per_goal": per_goal, "seed": seed,
        "sessions": {"train": sum(1 for l in labels.values() if l["group"] == "train"),
                     "holdout": sum(1 for l in labels.values() if l["group"] == "holdout"),
                     "banked_holdout": len(banked_ids)},
        "samples": sum(sample_counts),
        "mean_target_actions": round(statistics.mean(target_lengths), 2),
        "empty_target_samples": empty_targets,
        "skipped_placements": skipped_placements,   # mistakes + redundant clicks
        "belief_slot": "tracker replay, trained v1 heads + jointly fitted knobs",
        "arm2": (arm2_note or {
            "model": vlm.describe(),
            "cadence": f"events + every {decoder_corpus.ARM2_IDLE_EVERY}th idle, held between",
            "coverage_sessions": sorted(coverage | coverage_banked),
            "queries": arm2_reader.queries, "cache_hits": arm2_reader.cache_hits,
            "unusable_replies": arm2_reader.unusable} if arm2_reader else arm2_note),
        "hygiene": "asserted per sample (join verified; place targets never standing; "
                   "break targets always standing); a violation refuses the run",
    }
    report_path = os.path.join(_OUT, "decoder_corpus_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"decoder corpus: {report['samples']} samples across "
          f"{len(labels)} sessions -> {os.path.relpath(_OUT, _ROOT)}")
    print(f"  mean target actions {report['mean_target_actions']}, "
          f"{empty_targets} build-finished (empty-target) samples")
    print(f"  report: {os.path.basename(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
