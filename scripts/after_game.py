"""Process a capture after a game: gate -> evidence -> label -> inspect report.

    # one game you just played (builder's own label is required)
    python scripts/after_game.py --goal habitation --subtype treehouse
    python scripts/after_game.py --goal defense --subtype "square tower" --session fabric-20260706-...

    # catch up a stack of already-captured games (labels come from labels.json)
    python scripts/after_game.py --all

    # is the accumulated batch worth a cascade A yet?
    python scripts/after_game.py --status

For each session it: relocates the raw capture into its date/session folder (so its
logs are organized and future derived outputs land beside it), runs the B0 gate,
regenerates evidence offline (`run_d1 --pixels`, `run_d2 --h3d`) — which also recovers
a live-socket quarantine, since the disk copy is authoritative — labels the finished
build against the builder's stated goal, and writes a per-session `inspect.md` +
`session_report.json`. It STOPS before any retrain: matcher-agreed captures stack up
and `--status` tells you when the batch is worth firing `run_cascade_a.py`.

A structure quarantine (run_d2 exits non-zero: an unexplained voxel divergence, not a
recoverable socket drop) flags the session and skips labeling — it is not proof-grade.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from migrate_raw_layout import _session_moves                       # noqa: E402
from mica.capture import session_store                              # noqa: E402
from mica.capture.discovery import newest_capture                   # noqa: E402

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_RAW = session_store.RAW_ROOT
_LABELS = os.path.join(_RAW, "labels.json")
_LEDGER = os.path.join(_RAW, "cascade_status.json")
_SOURCE_B_REPORT = os.path.join(os.path.dirname(_RAW), "scripted", "source_b_report.json")


# ----------------------------------------------------------------- small helpers

def _flag(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


def _run(script: str, *args: str) -> int:
    """A pipeline step as a subprocess (streams its own output). Returns exit code."""
    print(f"\n$ {script} {' '.join(args)}")
    return subprocess.run([sys.executable, os.path.join(_SCRIPTS, script), *args]).returncode


def _load_labels() -> dict:
    with open(_LABELS, encoding="utf-8") as handle:
        return json.load(handle)


def _wait_for_manifest(session_id: str, timeout_s: float = 30.0) -> bool:
    """Wait until the mod has finalized the manifest (non-provisional). Returns
    immediately for already-finished captures; times out for a session still open."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        manifest = session_store.session_file(session_id, ".manifest.json")
        if os.path.exists(manifest):
            with open(manifest, encoding="utf-8") as handle:
                meta = json.load(handle)
            if int(meta.get("declared_event_count", -1)) >= 0:
                return True
        time.sleep(2.0)
    return False


def _relocate(session_id: str) -> None:
    """Move a flat capture into its date/session dir (no-op if already nested)."""
    moves = _session_moves(session_id)
    if not moves:
        return
    dest = os.path.join(_RAW, session_store.session_date(session_id), session_id)
    os.makedirs(dest, exist_ok=True)
    for source, target in moves:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.move(source, target)
    print(f"  relocated {session_id} -> {os.path.relpath(dest, _RAW)}/")


def _bank_quarantined_live(session_id: str) -> bool:
    """If the live run flagged a socket quarantine, copy the live outputs to
    *.live-quarantined.* before the offline --overwrite regen replaces them. Returns
    True if a live quarantine was seen."""
    run = session_store.session_file(session_id, ".live_run.json")
    if not os.path.exists(run):
        return False
    with open(run, encoding="utf-8") as handle:
        quarantined = bool(json.load(handle).get("quarantined"))
    if not quarantined:
        return False
    for suffix in (".evidence2d.jsonl", ".evidence3d.jsonl", ".fused.jsonl",
                   ".belief.jsonl", ".live_run.json"):
        live = session_store.session_file(session_id, suffix)
        if os.path.exists(live):
            banked = live.replace(suffix, suffix.replace(".", ".live-quarantined.", 1))
            if not os.path.exists(banked):
                shutil.copy2(live, banked)
    print("  live loop was quarantined (socket drops) — banked *.live-quarantined.*; "
          "regenerating from the authoritative disk copy")
    return True


def _upsert_label(session_id: str, goal: str, subtype: str, note: str) -> None:
    labels = _load_labels()
    entry = labels.get(session_id, {})
    entry.update({"goal": goal, "subtype": subtype,
                  "mode": entry.get("mode", "deliberate"),
                  "source": entry.get("source", "free build (real, pixels + h3d); "
                            "processed by after_game.py"),
                  "note": note})
    labels[session_id] = entry
    with open(_LABELS, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=1)


def _verdict_for(session_id: str) -> dict | None:
    """This session's matcher verdict, read from the freshly written labeling report."""
    if not os.path.exists(_SOURCE_B_REPORT):
        return None
    with open(_SOURCE_B_REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    for entry in report.get("real", {}).get("labeled", []):
        if entry["session"] == session_id:
            return entry
    return None


# ------------------------------------------------------------ readiness ledger

def _agreed_and_pairs() -> tuple[list[str], int]:
    """Current matcher-agreed real sessions and the total pair pool, from the report."""
    if not os.path.exists(_SOURCE_B_REPORT):
        return [], 0
    with open(_SOURCE_B_REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    agreed = sorted(e["session"] for e in report.get("real", {}).get("labeled", [])
                    if e.get("label", {}).get("kept") and e.get("agrees_with_builder"))
    pairs = sum(report.get("pairs_written", {}).values())
    return agreed, pairs


def _ledger() -> dict:
    if os.path.exists(_LEDGER):
        with open(_LEDGER, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def readiness() -> dict:
    agreed, pairs = _agreed_and_pairs()
    base = _ledger().get("last_cascade", {})
    base_agreed = set(base.get("agreed_sessions", []))
    new_agreed = [s for s in agreed if s not in base_agreed]
    return {
        "agreed_captures_total": len(agreed),
        "new_agreed_since_cascade": new_agreed,
        "pairs_total": pairs,
        "new_pairs_since_cascade": pairs - base.get("pair_pool", 0),
        "last_cascade": base.get("date", "never"),
    }


def print_readiness() -> None:
    r = readiness()
    print("\ncascade-A readiness:")
    print(f"  matcher-agreed real captures: {r['agreed_captures_total']} total")
    if r["last_cascade"] == "never":
        print(f"  no cascade recorded yet; {r['pairs_total']} pairs banked")
    else:
        print(f"  since the last cascade ({r['last_cascade']}): "
              f"{len(r['new_agreed_since_cascade'])} new agreed captures, "
              f"+{r['new_pairs_since_cascade']} pairs")
        if r["new_agreed_since_cascade"]:
            print(f"    new: {', '.join(r['new_agreed_since_cascade'])}")
    ready = r["last_cascade"] == "never" or r["new_agreed_since_cascade"]
    print(f"  {'READY — fire scripts/run_cascade_a.py when you want the retrain' if ready else 'nothing new since the last cascade'}")


# --------------------------------------------------------------- per session

def _write_inspect(session_id: str, gate_ok: bool, quarantined: bool,
                   verdict: dict | None) -> None:
    directory = session_store.session_dir(session_id)
    summary = {"session": session_id, "gate_file_checks": "PASS" if gate_ok else "FAIL",
               "structure_quarantined": quarantined, "verdict": verdict}
    with open(os.path.join(directory, "session_report.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    lines = [f"# {session_id}", ""]
    lines.append(f"- B0 gate (file checks): **{'PASS' if gate_ok else 'FAIL'}**")
    if quarantined:
        lines.append("- **STRUCTURE QUARANTINE** — unexplained voxel divergence; not proof-grade, not labeled.")
    if verdict:
        label = verdict.get("label", {})
        agree = verdict.get("agrees_with_builder")
        vlm = (verdict.get("vlm") or {}).get("category")
        lines.append(f"- matcher: **{label.get('goal')}/{label.get('subtype')}** "
                     f"score {label.get('score')} — "
                     f"{'AGREES' if agree else 'CONTESTS'} the builder"
                     + (f"; kept" if label.get("kept") else f"; DISCARDED ({label.get('reason')})"))
        lines.append(f"- VLM cross-check: {vlm or 'n/a'}")
        lines.append(f"- training pairs written: {verdict.get('pairs', 0)}")
        if label.get("kept") and agree:
            lines.append("- **matcher-agreed capture** — pairs eligible; counts toward the cascade-A batch.")
    with open(os.path.join(directory, "inspect.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  wrote {os.path.relpath(directory, _RAW)}/inspect.md")


def process_one(session_id: str, goal: str, subtype: str, no_vlm: bool) -> dict:
    print(f"\n=== processing {session_id} ({goal}/{subtype}) ===")
    _wait_for_manifest(session_id)
    _relocate(session_id)
    _bank_quarantined_live(session_id)
    jsonl = session_store.session_jsonl(session_id)

    if _run("b0_gate.py", jsonl) != 0:
        print("  B0 gate FAILED — not proof-grade; skipping evidence + label")
        _write_inspect(session_id, gate_ok=False, quarantined=False, verdict=None)
        return {"session": session_id, "ok": False, "reason": "gate fail"}

    _run("run_d1.py", jsonl, "--pixels", "--overwrite")
    d2 = _run("run_d2.py", jsonl, "--h3d", "--overwrite")
    if d2 != 0:
        print("  run_d2 exited non-zero — STRUCTURE QUARANTINE (unrecoverable); skipping label")
        _write_inspect(session_id, gate_ok=True, quarantined=True, verdict=None)
        return {"session": session_id, "ok": False, "reason": "structure quarantine"}

    _upsert_label(session_id, goal, subtype, "processed by after_game.py; matcher verdict below")
    label_args = ["--no-vlm"] if no_vlm else []
    _run("label_finished_builds.py", *label_args)
    verdict = _verdict_for(session_id)
    _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=verdict)
    return {"session": session_id, "ok": True, "verdict": verdict}


def _backlog_sessions() -> tuple[list[str], list[str]]:
    """(labeled-and-processable, needs-a-builder-label). A session is in the backlog
    if it has a raw jsonl but no evidence yet — "processed" means perception ran
    (evidence2d exists), regardless of whether the matcher then kept or contested it,
    so contested/discarded sessions aren't reprocessed forever."""
    labels = _load_labels()
    processable, needs_label = [], []
    for path in glob.glob(os.path.join(_RAW, "**", "fabric-*.jsonl"), recursive=True):
        name = os.path.basename(path)
        if name.count(".") != 1:
            continue                                   # derived sibling, not a capture
        session_id = name[:-len(".jsonl")]
        if os.path.exists(session_store.session_file(session_id, ".evidence2d.jsonl")):
            continue                                   # perception already ran
        (processable if session_id in labels else needs_label).append(session_id)
    return sorted(processable), sorted(needs_label)


def main() -> int:
    if "--status" in sys.argv:
        print_readiness()
        return 0

    no_vlm = "--no-vlm" in sys.argv
    if "--all" in sys.argv:
        labels = _load_labels()
        processable, needs_label = _backlog_sessions()
        if needs_label:
            print("sessions with a capture but no builder label (run with "
                  "--session <id> --goal <cat> --subtype <style> to process):")
            for sid in needs_label:
                print(f"  {sid}")
        for session_id in processable:
            label = labels[session_id]
            process_one(session_id, label["goal"], label.get("subtype", ""), no_vlm)
        if not processable:
            print("no un-processed labeled sessions in the backlog")
        print_readiness()
        return 0

    target = _flag("--session")
    if target and target.endswith(".jsonl"):
        target = os.path.basename(target)[:-len(".jsonl")]
    if target is None:
        target = newest_capture(_RAW)
        if target is None:
            print("no capture found — play a game first, or pass --session")
            return 1
        target = os.path.basename(target)[:-len(".jsonl")] if target.endswith(".jsonl") else target

    goal, subtype = _flag("--goal"), _flag("--subtype")
    if goal is None or subtype is None:
        existing = _load_labels().get(target, {})
        goal, subtype = goal or existing.get("goal"), subtype or existing.get("subtype")
    if not goal or not subtype:
        print(f"{target}: needs the builder's own label — pass --goal <category> "
              "--subtype <style> (the matcher compares against it)")
        return 1

    result = process_one(target, goal, subtype, no_vlm)
    print_readiness()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
