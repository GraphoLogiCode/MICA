"""One table answering: what data do we have, is it filed right, is it trainable?

    python scripts/audit_sessions.py             # the table + problems
    python scripts/audit_sessions.py --json      # full audit -> capture/raw/session_audit.json

For every capture session this checks, in plain terms:

  filing      the folder matches the session id, the date folder matches the id's
              date, and the manifest inside really names this session
  complete    which stage artifacts exist: capture, manifest finalized, evidence
              (2d/3d), belief/fused logs, gate trace (and whether it is one run
              per file), report, label, matcher verdict, training pairs
  coherent    the manifest's wallclock start agrees with the id's timestamp, and
              the session_report's verdict state matches what labels.json says
  trainable   evidence ready + matcher kept + agrees with the builder — the same
              rule the cascade uses; the last column says READY or why not

It also lists what sits at the RAW ROOT: sessions never relocated, agent scans
never adopted (organize_raw.py moves both), and the cross-session artifacts that
belong there (left alone by design — their writers pin those paths).

Verdict per session: OK (all present and consistent), PART (incomplete but honest
— e.g. awaiting label), or BAD (something contradicts: wrong folder, torn
metadata, mismatched report). Exit 1 only when a BAD exists.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = session_store.RAW_ROOT
_SOURCE_B_REPORT = os.path.join(_ROOT, "capture", "scripted", "source_b_report.json")
_ID_STAMP = re.compile(r"^fabric-(\d{8})-(\d{6})$")
# The id encodes the mod's local start time; the manifest's session_start_ms is
# wallclock. Agreement within a day tolerates timezone offsets without letting a
# mislabeled file (wrong session's manifest) slip through.
_START_TOLERANCE_S = 26 * 3600.0


def _read_json(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _labels() -> dict:
    return _read_json(os.path.join(_RAW, "labels.json")) or {}


def _verdicts() -> dict:
    report = _read_json(_SOURCE_B_REPORT) or {}
    return {entry["session"]: entry
            for entry in report.get("real", {}).get("labeled", [])}


def _trace_runs(path: str) -> int:
    """How many runs share this trace file (k restarting at 1 marks a new run)."""
    runs = 0
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    k = json.loads(line).get("k")
                except ValueError:
                    continue
                if k == 1 or runs == 0:
                    runs += 1
    except OSError:
        return 0
    return runs


def audit_session(session_id: str) -> dict:
    directory = session_store.session_dir(session_id)
    base = os.path.join(directory, session_id)
    problems: list[str] = []
    notes: list[str] = []

    # -- filing -----------------------------------------------------------------
    nested = os.path.basename(directory) == session_id
    if nested:
        date_dir = os.path.basename(os.path.dirname(directory))
        try:
            if session_store.session_date(session_id) != date_dir:
                problems.append(f"sits under {date_dir}/ but its id says "
                                f"{session_store.session_date(session_id)}")
        except ValueError:
            problems.append("non-dated id in a dated folder")
    else:
        notes.append("still flat at the raw root (organize/after_game will file it)")

    manifest = _read_json(base + ".manifest.json")
    finalized = None
    if manifest is None:
        problems.append("manifest missing or unreadable")
    else:
        if manifest.get("session_id") != session_id:
            problems.append(f"manifest names {manifest.get('session_id')!r} — wrong file")
        finalized = int(manifest.get("declared_event_count", -1)) >= 0
        if not finalized:
            notes.append("manifest still provisional (session open, or crashed mid-write)")
        stamp = _ID_STAMP.match(session_id)
        if stamp and manifest.get("session_start_ms"):
            id_day = stamp.group(1)
            import datetime
            start = datetime.datetime.fromtimestamp(
                manifest["session_start_ms"] / 1000.0)
            drift = abs((start - datetime.datetime.strptime(
                stamp.group(1) + stamp.group(2), "%Y%m%d%H%M%S")).total_seconds())
            if drift > _START_TOLERANCE_S:
                problems.append(f"manifest start is {drift / 3600.0:.0f} h away from "
                                "the id's timestamp — mismatched capture?")
            del id_day

    # -- completeness -----------------------------------------------------------
    have = {name: os.path.exists(base + suffix) for name, suffix in (
        ("capture", ".jsonl"), ("evidence2d", ".evidence2d.jsonl"),
        ("evidence3d", ".evidence3d.jsonl"), ("belief", ".belief.jsonl"),
        ("fused", ".fused.jsonl"), ("gate_trace", ".gate_trace.jsonl"),
        ("pairs", ".source_b.jsonl"), ("live_run", ".live_run.json"))}
    have["report"] = os.path.exists(os.path.join(directory, "session_report.json"))
    have["scan"] = bool(glob.glob(os.path.join(directory, "agent-*.scan*.jsonl")))
    if not have["capture"]:
        problems.append("no capture jsonl where the layout says it should be")
    if have["gate_trace"]:
        runs = _trace_runs(base + ".gate_trace.jsonl")
        if runs > 1:
            problems.append(f"gate trace holds {runs} runs in one file "
                            "(run organize_raw.py to split)")

    # -- label / verdict coherence ----------------------------------------------
    # The verdict lives in the GLOBAL report; the per-session report is a later
    # convention — its absence must not hide a session the cascade already counts.
    label = _labels().get(session_id)
    verdict = _verdicts().get(session_id)
    report = _read_json(os.path.join(directory, "session_report.json"))
    trainable = "no"
    why = ""
    if report is not None and report.get("gate_file_checks") == "FAIL":
        why = "B0 gate failed"
    elif report is not None and report.get("structure_quarantined"):
        why = "structure quarantined"
    elif label is None:
        why = ("awaiting builder label" if have["evidence2d"]
               else "unprocessed (no evidence yet)")
    elif verdict is None or verdict.get("label") is None:
        why = "labeled, no matcher verdict yet"
        if report is not None and report.get("verdict") is not None:
            problems.append("session_report has a verdict but the global report "
                            "does not — one of them is stale")
    else:
        kept = verdict["label"].get("kept")
        agrees = verdict.get("agrees_with_builder")
        report_verdict = (report or {}).get("verdict")
        if report_verdict is not None and \
                report_verdict.get("agrees_with_builder") != agrees:
            problems.append("session_report and the global report disagree on the "
                            "verdict — relabel to resync")
        if kept and agrees:
            trainable = "READY"
            why = f"{verdict.get('pairs', 0)} pairs"
            if not have["pairs"]:
                problems.append("counts as agreed but its pairs file is missing")
        else:
            why = "matcher contests" if kept else "matcher discarded"

    status = "BAD" if problems else ("OK" if trainable == "READY" or why in
                                     ("matcher contests", "matcher discarded")
                                     else "PART")
    return {"session": session_id, "status": status, "have": have,
            "label": (f'{label["goal"]}/{label.get("subtype", "")}' if label else "-"),
            "trainable": trainable, "why": why,
            "problems": problems, "notes": notes}


def _all_sessions() -> list[str]:
    ids = set()
    for path in glob.glob(os.path.join(_RAW, "**", "fabric-*.jsonl"), recursive=True):
        name = os.path.basename(path)
        if name.count(".") == 1 and "previous_runs" not in path:
            ids.add(name[: -len(".jsonl")])
    return sorted(ids)


def _root_strays() -> tuple[list[str], list[str]]:
    """(actionable strays, expected cross-session artifacts) at the raw root."""
    actionable, expected = [], []
    for name in sorted(os.listdir(_RAW)):
        path = os.path.join(_RAW, name)
        if os.path.isdir(path):
            continue
        if re.match(r"^fabric-\d{8}-\d{6}\.", name):
            actionable.append(f"{name} — unrelocated session file")
        elif re.match(r"^agent-.*\.scan\..+\.jsonl$", name):
            actionable.append(f"{name} — agent scan not yet adopted "
                              "(organize_raw.py moves it)")
        else:
            expected.append(name)
    return actionable, expected


def main() -> int:
    rows = [audit_session(sid) for sid in _all_sessions()]
    width = max((len(r["session"]) for r in rows), default=20)
    stages = ("capture", "evidence2d", "evidence3d", "belief", "gate_trace",
              "report", "scan", "pairs")
    header = ("STATUS  " + "SESSION".ljust(width) + "  "
              + " ".join(s[:4] for s in stages) + "  " + "LABEL".ljust(24)
              + "TRAINABLE")
    print(header)
    print("-" * len(header))
    for row in rows:
        marks = " ".join(("y" if row["have"][s] else ".").ljust(4) for s in stages)
        print(f"{row['status']:<6}  {row['session']:<{width}}  {marks}  "
              f"{row['label']:<23.23} {row['trainable']} ({row['why']})")
        for problem in row["problems"]:
            print(f"        !! {problem}")
        for note in row["notes"]:
            print(f"        -- {note}")

    actionable, expected = _root_strays()
    if actionable:
        print("\nat the raw root, needing a home:")
        for line in actionable:
            print(f"  !! {line}")
    print(f"\ncross-session artifacts at the root (left in place by design): "
          f"{len(expected)}")
    ready = sum(1 for r in rows if r["trainable"] == "READY")
    bad = sum(1 for r in rows if r["status"] == "BAD")
    print(f"{len(rows)} sessions: {ready} training-ready, "
          f"{sum(1 for r in rows if r['status'] == 'PART')} in progress, {bad} bad")

    if "--json" in sys.argv:
        out = os.path.join(_RAW, "session_audit.json")
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"sessions": rows, "root_strays": actionable,
                       "root_expected": expected}, handle, indent=2)
        os.replace(tmp, out)
        print(f"full audit -> {os.path.relpath(out, _ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
