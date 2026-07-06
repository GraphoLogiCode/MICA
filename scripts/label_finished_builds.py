"""Source B labeling — the D3 data-stage proof artifact, one command.

    python scripts/label_finished_builds.py [--no-vlm] [--skip-real]

Runs the finished-build recognizer over everything labelable:

  1. The SCRIPTED corpus (regenerated in memory from the banked seed — labels are
     known by construction), which doubles as the labeler's accuracy measurement:
     the report's kept-label accuracy is the go/no-go number for the recognizer.
  2. Every REAL capture named in capture/raw/labels.json (the builder's own label is
     the comparison there, not training truth — the matcher's label is what would
     enter training).

For each KEPT session it writes <session>.source_b.jsonl — the replay pairs (partial
pre-action evidence -> next action, all tagged with the finished-build label) — and
for kept real captures it runs the VLM cross-check over the session's final POV
frames (Anthropic API when ANTHROPIC_API_KEY is set; otherwise the report carries a
sample sheet for manual review). The VLM only agrees or contests; it never labels.

The labeling report lands at capture/scripted/source_b_report.json with the counts,
the confident-match fraction, the label distribution, sample early pairs, and the
pinned thresholds — regenerable by rerunning this command.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.data import vlm                                             # noqa: E402
from mica.capture import session_store                              # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource                     # noqa: E402
from mica.capture.scripted_goals import build_from_plan, plan_variants  # noqa: E402
from mica.capture.synthetic import generate_session                   # noqa: E402
from mica.contracts.b1 import GOALS                                   # noqa: E402
from mica.contracts.serialize import evidence2d_to_dict, evidence3d_to_dict  # noqa: E402
from mica.data.source_b import (                                      # noqa: E402
    MIN_MARGIN, MIN_SCORE, build_pairs, finished_world, label_finished_build,
)
from mica.perception.evidence2d import evidence_stream                # noqa: E402
from mica.perception.evidence3d import build_evidence3d               # noqa: E402
from mica.perception.voxel_replay import (                            # noqa: E402
    ReplayWorld, load_snapshot, region_around_events, region_from_manifest,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_SCRIPTED = os.path.join(_ROOT, "capture", "scripted")
_VLM_FRAMES = 4            # final POV frames shown to the cross-checker
_SAMPLE_PAIRS = 6          # early pairs quoted in the report for eyeball review
_EARLY_CUTOFF = 0.35       # "early" = less than this fraction of the build placed


def _capture_roots() -> list[str]:
    env = os.environ.get("MICA_RAW_DIRS")
    if env:
        return [d.strip() for d in env.split(";") if d.strip()]
    return [_RAW]


# ------------------------------------------------------------- scripted sessions

def _label_scripted() -> list[dict]:
    """Regenerate the banked corpus (per-goal 6, seed 7 — the no-flag command's own
    defaults) and label each finished build; truth is known, so this measures the
    recognizer. Pairs are written from the corpus's banked evidence files."""
    results = []
    for goal in GOALS:
        for plan in plan_variants(goal, 6, 7):
            build, truth = build_from_plan(plan)
            session = generate_session(build)
            world = finished_world(session.packets, region_around_events(session), {})
            label = label_finished_build(world)
            entry = {
                "session": build.session_id, "kind": "scripted",
                "truth_goal": truth["goal"], "truth_subtype": truth["subtype"],
                "label": label, "goal_correct": label.goal == truth["goal"],
                "subtype_correct": label.subtype == truth["subtype"],
            }
            if label.kept:
                entry["pairs"] = _write_pairs(_SCRIPTED, build.session_id, label)
            results.append(entry)
    return results


# ----------------------------------------------------------------- real sessions

def _label_real() -> list[dict]:
    """Every real capture named in capture/raw/labels.json. Needs the session's region
    and base snapshot to reconstruct the finished build; sessions without them are
    skipped with the reason on the record — never guessed at."""
    labels_path = os.path.join(_RAW, "labels.json")
    if not os.path.exists(labels_path):
        return []
    with open(labels_path, encoding="utf-8") as handle:
        human_labels = json.load(handle)
    results = []
    for session_id, human in human_labels.items():
        entry = {"session": session_id, "kind": "real",
                 "builder_label": {"goal": human.get("goal"), "subtype": human.get("subtype")}}
        # session_store resolves each root to the session's dated dir (or flat, for
        # un-migrated captures / external MICA_RAW_DIRS roots); snapshots + written
        # source_b then follow os.path.dirname(jsonl), which is the session dir.
        jsonl = next((path for root in _capture_roots()
                      for path in [session_store.session_file(session_id, ".jsonl", root)]
                      if os.path.exists(path)), None)
        if jsonl is None:
            entry["skipped"] = "recording not found in any capture root"
            results.append(entry)
            continue
        session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
        region = region_from_manifest(session)
        snapshots = sorted(
            glob.glob(os.path.join(os.path.dirname(jsonl), session_id, "snapshots", "*.json")),
            key=lambda p: int(os.path.basename(p)[:-5]))
        if region is None or not snapshots:
            entry["skipped"] = "no snapshot region/files — finished world cannot be reconstructed"
            results.append(entry)
            continue
        base = load_snapshot(snapshots[0])[2]
        world = finished_world(session.packets, region, base)
        label = label_finished_build(world)
        entry["label"] = label
        entry["agrees_with_builder"] = label.goal == human.get("goal")
        if label.kept and not entry["agrees_with_builder"]:
            # The 193059 lesson: a KEPT matcher verdict can still be wrong on a real
            # free build (it confidently read the builder's HOUSE as a fountain).
            # A contested label must never become training pairs — the builder's
            # word outranks the matcher, and the VLM cross-check arbitrates below.
            entry["pairs_withheld"] = "matcher contests the builder's label"
        elif label.kept:
            entry["pairs"] = _write_pairs(os.path.dirname(jsonl), session_id, label,
                                          session=session, region=region, base=base)
        if label.kept:
            entry["frames"] = sorted(
                glob.glob(os.path.join(os.path.dirname(jsonl), session_id, "frames", "*.png")),
                key=lambda p: int(os.path.basename(p)[:-4]))[-_VLM_FRAMES:]
        results.append(entry)
    return results


def _write_pairs(evidence_dir: str, session_id: str, label,
                 session=None, region=None, base=None) -> int:
    """Pairs from the banked evidence files when they exist; recomputed through the
    same streams otherwise (symbolic channels only — no h3d without its flag)."""
    b1_path = os.path.join(evidence_dir, f"{session_id}.evidence2d.jsonl")
    b2_path = os.path.join(evidence_dir, f"{session_id}.evidence3d.jsonl")
    if os.path.exists(b1_path) and os.path.exists(b2_path):
        b1 = [json.loads(line) for line in open(b1_path, encoding="utf-8") if line.strip()]
        b2 = [json.loads(line) for line in open(b2_path, encoding="utf-8") if line.strip()]
    else:
        records = tuple(evidence_stream(session.packets))
        corrections = tuple(r for r in records if r.scored)
        events = {e.event_id: e for p in session.packets for e in p.server.block_events}
        world = ReplayWorld(region, dict(base))
        b2_records = build_evidence3d(world, corrections, events)
        b1 = [evidence2d_to_dict(r) for r in records]
        b2 = [evidence3d_to_dict(r) for r in b2_records]
    pairs = build_pairs(b1, b2, label, session_id)
    out = os.path.join(evidence_dir, f"{session_id}.source_b.jsonl")
    with open(out, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    return len(pairs)


# -------------------------------------------------------------- VLM cross-check

def _vlm_verdict(frame_paths: list[str]) -> dict | None:
    """Ask a vision model which category the final frames show; agree/contest only.
    Local-first through mica/data/vlm.py (Ollama on this GPU, else the Anthropic
    API); None when no backend exists — the caller then leaves the frames on the
    sample sheet for manual review instead."""
    if not frame_paths:
        return None
    text = vlm.ask(frame_paths,
                   "These are the final frames of a Minecraft building session. Which ONE "
                   "category best describes what the player built: habitation, infrastructure, "
                   "production, defense, or decorative? Answer with only the category word.")
    if text is None:
        return None
    lower = text.lower()
    call = next((goal for goal in GOALS if goal in lower), None)
    return {"model": vlm.describe(), "raw": lower.strip()[:120], "category": call}


# ----------------------------------------------------------------------- report

def _sample_pairs(results: list[dict]) -> list[dict]:
    """A handful of EARLY pairs for eyeball review — the ones the early-separation
    claim rests on: weak evidence, the finished-build label, the next action."""
    samples = []
    for entry in results:
        if not entry.get("pairs"):
            continue
        directory = _SCRIPTED if entry["kind"] == "scripted" else None
        path = os.path.join(directory or _RAW, f"{entry['session']}.source_b.jsonl")
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            pair = json.loads(line)
            if 0.0 < pair["progress"] <= _EARLY_CUTOFF and pair["b1"]["event_ids"]:
                goal = pair["label"]["goal"]
                samples.append({
                    "session": pair["session"], "progress": pair["progress"],
                    "label": pair["label"], "next_action": pair["next_action"],
                    "held_item": pair["b1"]["state_feats"]["held_item"],
                    "labeled_goal_read_now": {
                        "comp": pair["b2"]["per_goal"][goal]["comp"],
                        "fit": pair["b2"]["per_goal"][goal]["fit"]},
                })
                break                     # one early pair per session is enough to read
        if len(samples) >= _SAMPLE_PAIRS:
            break
    return samples


def main() -> int:
    results = _label_scripted()
    real = [] if "--skip-real" in sys.argv else _label_real()

    for entry in real:
        if entry.get("frames") and "--no-vlm" not in sys.argv:
            entry["vlm"] = _vlm_verdict(entry.pop("frames"))
        elif entry.get("frames"):
            entry["manual_review_frames"] = entry.pop("frames")

    scripted_kept = [r for r in results if r["label"].kept]
    kept_accuracy = (sum(r["goal_correct"] for r in scripted_kept) / len(scripted_kept)
                     if scripted_kept else 0.0)
    labeled_real = [r for r in real if "label" in r]
    kept_real = [r for r in labeled_real if r["label"].kept]

    def _label_dict(entry):
        out = dict(entry)
        if "label" in out:
            out["label"] = vars(out["label"])
        return out

    distribution: dict[str, dict] = {}
    for entry in scripted_kept + kept_real:
        goal_stats = distribution.setdefault(entry["label"].goal, {"kept": 0, "subtypes": {}})
        goal_stats["kept"] += 1
        subtype = entry["label"].subtype
        goal_stats["subtypes"][subtype] = goal_stats["subtypes"].get(subtype, 0) + 1

    report = {
        "thresholds": {"min_score_fit_x_comp": MIN_SCORE, "min_margin": MIN_MARGIN},
        "scripted": {
            "sessions": len(results),
            "confident_match_fraction": round(len(scripted_kept) / len(results), 3),
            "kept_label_accuracy": round(kept_accuracy, 3),
            "kept_subtype_accuracy": round(
                sum(r["subtype_correct"] for r in scripted_kept) / len(scripted_kept), 3)
            if scripted_kept else 0.0,
            "discarded": [{"session": r["session"], "reason": r["label"].reason}
                          for r in results if not r["label"].kept],
        },
        "real": {
            "sessions": len(real),
            "labeled": [_label_dict(r) for r in real],
        },
        "label_distribution_kept": distribution,
        "pairs_written": {r["session"]: r["pairs"] for r in results + real if r.get("pairs")},
        "sample_early_pairs": _sample_pairs(results + real),
    }
    out = os.path.join(_SCRIPTED, "source_b_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    s = report["scripted"]
    print(f"Source B labeling report -> {os.path.relpath(out, _ROOT)}")
    print(f"  thresholds: score >= {MIN_SCORE} (fit x comp), margin >= {MIN_MARGIN}")
    print(f"  scripted corpus: {s['sessions']} sessions, kept {s['confident_match_fraction']:.0%},"
          f" kept-label accuracy {s['kept_label_accuracy']}"
          f" (subtype {s['kept_subtype_accuracy']})")
    for discarded in s["discarded"]:
        print(f"    discarded {discarded['session']}: {discarded['reason']}")
    for entry in real:
        if entry.get("skipped"):
            print(f"  real {entry['session']}: SKIPPED — {entry['skipped']}")
            continue
        label = entry["label"]
        agrees = "agrees with" if entry["agrees_with_builder"] else "CONTESTS"
        verdict = f"{label.goal}/{label.subtype} score {label.score}"
        verdict += f" — {agrees} the builder's label" if label.kept else f" — DISCARDED ({label.reason})"
        print(f"  real {entry['session']}: {verdict}")
        vlm = entry.get("vlm")
        if vlm:
            print(f"    VLM cross-check: {vlm.get('category') or vlm.get('error')}"
                  + (" (agrees)" if vlm.get("category") == label.goal else " (CONTESTS)"
                     if vlm.get("category") else ""))
        elif entry.get("manual_review_frames"):
            print(f"    VLM skipped (no key): {len(entry['manual_review_frames'])} frames on the sheet")
    total_pairs = sum(report["pairs_written"].values())
    print(f"  replay pairs written: {total_pairs} across {len(report['pairs_written'])} sessions"
          f" (<session>.source_b.jsonl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
