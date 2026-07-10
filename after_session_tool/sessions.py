"""The data + logic layer for the after-session labeling dashboard.

Everything the pipeline already knows lives in scripts/after_game.py; this module
imports those functions rather than re-deriving anything. It adds only what the
dashboard itself owns: a tiny skip store, an in-memory registry for the background
label jobs, and the pure helpers that turn a session's files into a card's state.

The pure helpers (derive_state, inbox_from, subtype_presets) take plain data and
return plain data, so they test without touching the pipeline or the filesystem.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # the MICA project root
_SCRIPTS = os.path.join(_ROOT, "scripts")
for path in (_ROOT, _SCRIPTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import after_game                                    # noqa: E402  (the one source of truth)
from mica.capture import session_store               # noqa: E402
from mica.contracts.goals import GOALS, TAXONOMY     # noqa: E402

_RAW = session_store.RAW_ROOT
_REPORTS = os.path.join(os.path.dirname(_RAW), "reports")
_RIG_LOG = os.path.join(_RAW, "rig_log.jsonl")
_SKIPPED = os.path.join(_HERE, "skipped.json")


# ------------------------------------------------------------------ pure logic

def derive_state(report: dict | None, has_evidence: bool, job: str | None) -> str:
    """One word for where a session is, from its report + evidence + any live job.

    Only "ready" enables the label form; "labeled" belongs in History, not the
    inbox; everything else is a status the human can read but not act on yet."""
    if job == "running":
        return "labeling"
    if report is None:
        # No report at all: "processing" ONLY if evidence exists (a chain is
        # genuinely mid-run); otherwise the capture was never processed — say so
        # honestly rather than implying work is happening.
        return "processing" if has_evidence else "unprocessed"
    if report.get("gate_file_checks") == "FAIL":
        return "gate failed"
    if report.get("structure_quarantined"):
        return "quarantined"
    if report.get("verdict") is not None:
        return "labeled"
    if report.get("evidence_ready") and report.get("awaiting_label") and has_evidence:
        return "ready"
    return "processing"


def subtype_presets(goal: str) -> tuple[str, ...]:
    """The taxonomy's style suggestions for a goal (free text is still allowed —
    a builder's own subtype can sit outside the taxonomy, e.g. 'sheep cage')."""
    return TAXONOMY.get(goal, ())


def inbox_from(needs_label: list[str], skipped: list[str]) -> list[str]:
    """The sessions the human still owes a label for: those needing one, minus the
    ones they set aside. (needs_label already excludes anything in labels.json.)"""
    skip = set(skipped)
    return [sid for sid in needs_label if sid not in skip]


# ------------------------------------------------------------------ skip store

def load_skipped() -> list[str]:
    try:
        with open(_SKIPPED, encoding="utf-8") as handle:
            return list(json.load(handle))
    except (OSError, ValueError):
        return []


def _save_skipped(ids: list[str]) -> None:
    # Write-then-rename: a crash mid-write must never leave a half-written file
    # (a corrupt skip store would silently resurface every skipped session).
    tmp = _SKIPPED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(sorted(set(ids)), handle, indent=1)
    os.replace(tmp, _SKIPPED)


def skip(sid: str) -> None:
    ids = load_skipped()
    if sid not in ids:
        ids.append(sid)
        _save_skipped(ids)


def unskip(sid: str) -> None:
    _save_skipped([s for s in load_skipped() if s != sid])


# --------------------------------------------------------------- label jobs

# A label runs after_game.py in the background so the browser never blocks on the
# matcher. State per session id: running / done / failed, plus the output tail.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def job(sid: str) -> dict | None:
    with _jobs_lock:
        entry = _jobs.get(sid)
        return dict(entry) if entry else None


def _run_job(sid: str, cmd: list[str]) -> None:
    try:
        proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        state = "done" if proc.returncode == 0 else "failed"
    except Exception as error:                       # a spawn failure is a job failure
        out, state = str(error), "failed"
    with _jobs_lock:
        _jobs[sid].update(state=state, tail=out.splitlines()[-25:])


def _start(sid: str, cmd: list[str], meta: dict) -> bool:
    """Run an after_game.py command for sid in the background; False if ANY job is
    already in flight. One at a time, whatever the session: two concurrent
    after_game runs race on the shared files (labels.json is read-modify-written,
    and the matcher rewrites the global report and every kept session's pairs), so
    a second click must wait for the first job, not corrupt it."""
    with _jobs_lock:
        if any(job.get("state") == "running" for job in _jobs.values()):
            return False
        _jobs[sid] = {"state": "running", "started": time.time(), "tail": [], **meta}
    threading.Thread(target=_run_job, args=(sid, cmd), daemon=True).start()
    return True


def start_label_job(sid: str, goal: str, subtype: str, vlm: bool) -> bool:
    """The label run: after_game.py --session --goal --subtype (matcher + verdict)."""
    cmd = [sys.executable, os.path.join(_SCRIPTS, "after_game.py"),
           "--session", sid, "--goal", goal, "--subtype", subtype]
    if not vlm:
        cmd.append("--no-vlm")
    return _start(sid, cmd, {"kind": "label", "goal": goal, "subtype": subtype})


def start_process_job(sid: str) -> bool:
    """The evidence-only run for an unprocessed capture: after_game.py --session
    --evidence-only (gate + run_d1 + run_d2, no label) — turns 'not processed'
    into 'ready to label' without waiting for a full replay of the game."""
    cmd = [sys.executable, os.path.join(_SCRIPTS, "after_game.py"),
           "--session", sid, "--evidence-only"]
    return _start(sid, cmd, {"kind": "process"})


# ------------------------------------------------------ per-session file reads

def _has_evidence(sid: str) -> bool:
    return os.path.exists(session_store.session_file(sid, ".evidence2d.jsonl"))


def reports_dir(sid: str) -> str:
    """Where this session's report graphs live (may not exist yet)."""
    return os.path.join(_REPORTS, sid)


def frames_dir(sid: str) -> str:
    """Where the mod's in-game POV screenshots live for this session. The frames
    sit in a <sid>/frames/ subdir next to the capture, whatever layout (dated or
    legacy-flat) session_store resolves the jsonl to."""
    base = session_store.session_jsonl(sid)[: -len(".jsonl")]
    return os.path.join(base, "frames")


def pov_frame(sid: str) -> str | None:
    """The newest in-game POV screenshot: the HIGHEST-TICK <tick>.png in the
    frames dir, or None when the session recorded none. The tick is compared as
    an integer, not lexically — otherwise '999.png' would beat '10266.png' and
    the picture would be from mid-session, not the finished build. Present even
    for unprocessed captures, so it is the one 'what was built' view that always
    works."""
    directory = frames_dir(sid)
    best_tick, best_path = -1, None
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if not name.endswith(".png"):
            continue
        try:
            tick = int(name[:-4])
        except ValueError:
            continue
        if tick > best_tick:
            best_tick, best_path = tick, os.path.join(directory, name)
    return best_path


def report_pngs(sid: str) -> list[str]:
    """The report graph filenames for this session (empty if none rendered yet)."""
    order = ["clouds.png", "scan.png", "structure.png", "belief.png",
             "channels.png", "actions.png"]
    have = {os.path.basename(p) for p in
            glob.glob(os.path.join(_REPORTS, sid, "*.png"))}
    return [name for name in order if name in have] + sorted(have - set(order))


def shot_path(sid: str, name: str) -> str | None:
    """Resolve a report PNG path, refusing anything that escapes the reports dir."""
    if not name.endswith(".png") or "/" in name or "\\" in name or ".." in name:
        return None
    path = os.path.join(_REPORTS, sid, name)
    return path if os.path.exists(path) else None


def _summary(sid: str) -> dict:
    path = os.path.join(_REPORTS, sid, "summary.json")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _inspect_md(sid: str) -> str:
    path = os.path.join(session_store.session_dir(sid), "inspect.md")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _latest_rig_event(sid: str) -> dict | None:
    """The session's most recent line in the rig automation log — the live-ish
    hint for what the background chain last did with it."""
    try:
        with open(_RIG_LOG, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("session") == sid:
            return row
    return None


# ----------------------------------------------------------------- view models

def _card(sid: str) -> dict:
    report = after_game._prior_report(sid)
    state = derive_state(report, _has_evidence(sid), (job(sid) or {}).get("state"))
    verdict = (report or {}).get("verdict")
    label = after_game._load_labels().get(sid, {})
    return {
        "sid": sid,
        "state": state,
        "report": report,
        "verdict": verdict,
        "label": label,
        "pngs": report_pngs(sid),
        "job": job(sid),
    }


# Ready-to-label sessions come first — they are the ones the human can act on;
# the rest (still processing, quarantined, gate-failed) sink to the bottom.
_STATE_ORDER = {"ready": 0, "labeling": 1, "processing": 2, "quarantined": 3,
                "gate failed": 4, "unprocessed": 5, "labeled": 6}


def inbox() -> list[dict]:
    _processable, needs_label = after_game._backlog_sessions()
    sids = inbox_from(needs_label, load_skipped())
    cards = [_card(sid) for sid in sids]
    cards.sort(key=lambda c: (_STATE_ORDER.get(c["state"], 9), c["sid"]))
    return cards


def session_view(sid: str) -> dict:
    card = _card(sid)
    card["summary"] = _summary(sid)
    card["inspect"] = _inspect_md(sid)
    card["rig_event"] = _latest_rig_event(sid)
    return card


def history() -> dict:
    labels = after_game._load_labels()
    labeled = []
    for sid in sorted(labels):
        if not sid.startswith("fabric-"):
            continue                                 # scripted/other labels aren't sessions
        report = after_game._prior_report(sid)
        labeled.append({"sid": sid, "label": labels[sid],
                        "verdict": (report or {}).get("verdict")})
    skipped = [{"sid": sid, "label": labels.get(sid, {})}
               for sid in sorted(load_skipped())]
    return {"labeled": labeled, "skipped": skipped}


def status() -> dict:
    return after_game.readiness()
