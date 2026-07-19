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
from mica.perception.templates import template_backed  # noqa: E402

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
    """The goal's MATCHER-CONFIRMABLE v3 subtypes — the ones backed by a template
    instance, so the matcher can actually read the style back. (Free text is still
    allowed — a builder's own subtype can sit outside the taxonomy.)"""
    return template_backed(goal) if goal in TAXONOMY else ()


def roadmap_presets(goal: str) -> tuple[str, ...]:
    """The goal's definitions-only v3 subtypes: fully valid labels (definitions in
    the Taxonomy v3 vault note), but no template exists yet, so the matcher cannot
    confirm the style — agreement stays category-level either way."""
    backed = set(template_backed(goal)) if goal in TAXONOMY else set()
    return tuple(s for s in TAXONOMY.get(goal, ()) if s not in backed)


def batch_problem(entries: list[dict]) -> str | None:
    """Why a staged batch can't run, or None when it is fine. The GUI checks this
    when staging AND before launch, so a bad queue never reaches after_game."""
    if not entries:
        return "the queue is empty"
    seen = set()
    for entry in entries:
        sid = entry.get("session", "")
        if not sid:
            return "an entry has no session id"
        if sid in seen:
            return f"{sid} is staged twice"
        seen.add(sid)
        if entry.get("goal") not in GOALS:
            return f"{sid}: pick a goal from the five categories"
        if not str(entry.get("subtype", "")).strip():
            return f"{sid}: subtype is empty"
    return None


def checklist_from(labels: dict, matcher_rows: dict, states: dict[str, str],
                   skipped: list[str]) -> dict:
    """Split every session into 'feeds the cascade' vs 'excluded, with the reason'.

    Only a session that is labeled, KEPT by the matcher, and category-AGREED
    produces training pairs; this lists everything else with why it is out, so no
    unlabeled or contested session can slip into a retrain unnoticed.

    labels: labels.json; matcher_rows: the labeling report's real rows by session;
    states: chain state per not-yet-labeled inbox session; skipped: the skip store."""
    included, excluded = [], []
    for sid in sorted(labels):
        if not sid.startswith("fabric-"):
            continue                                     # scripted labels aren't sessions
        row = matcher_rows.get(sid)
        label = (row or {}).get("label") or {}
        if row is None:
            excluded.append({"sid": sid, "reason": "label saved, matcher not run on it yet"})
        elif row.get("skipped"):
            excluded.append({"sid": sid, "reason": f"matcher skipped it ({row['skipped']})"})
        elif label.get("kept") and row.get("agrees_with_builder"):
            included.append({"sid": sid,
                             "label": f"{label.get('goal')}/{label.get('subtype')}",
                             "pairs": row.get("pairs", 0)})
        elif label.get("kept"):
            excluded.append({"sid": sid, "reason": "matcher CONTESTS your label — your "
                             "word wins: pairs train with the builder's label "
                             "(D3 amendment 2026-07-19)"})
        else:
            excluded.append({"sid": sid,
                             "reason": f"discarded: {label.get('reason', 'below threshold')}"})
    for sid in sorted(states):
        if sid not in labels:
            excluded.append({"sid": sid, "reason": f"no label yet ({states[sid]})"})
    for sid in sorted(skipped):
        if sid not in labels and sid not in states:
            excluded.append({"sid": sid, "reason": "skipped by you — no label"})
    return {"included": included, "excluded": excluded}


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

# A label runs after_game.py in the background so the window never blocks on the
# matcher. State per session id: running / done / failed, plus the output tail.
# A batch job sits under one reserved key and carries its staged session ids.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_BATCH_KEY = "__batch__"
_STAGED = os.path.join(_HERE, "staged_labels.json")


def job(sid: str) -> dict | None:
    with _jobs_lock:
        entry = _jobs.get(sid)
        if entry is None:
            # A session staged in the batch job shares the batch's state, so its
            # card reads "labeling" while the one matcher run covers it.
            batch = _jobs.get(_BATCH_KEY)
            if batch and sid in batch.get("sessions", ()):
                entry = batch
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


def start_batch_label_job(entries: list[dict], vlm: bool) -> bool:
    """One after_game.py --batch run for a staged queue: the matcher (and its
    model load) runs ONCE for every queued session instead of once per label.
    Entries must already pass batch_problem(); returns False when another job is
    in flight, like the other starters."""
    problem = batch_problem(entries)
    if problem:
        raise ValueError(problem)
    tmp = _STAGED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=1)
    os.replace(tmp, _STAGED)
    cmd = [sys.executable, os.path.join(_SCRIPTS, "after_game.py"), "--batch", _STAGED]
    if not vlm:
        cmd.append("--no-vlm")
    return _start(_BATCH_KEY, cmd,
                  {"kind": "batch", "sessions": [e["session"] for e in entries]})


def declared_target(sid: str) -> dict | None:
    """What the builder declared they were building (or None) — read through
    after_game's store, never labels.json (a declaration is not a label)."""
    return after_game.load_declared().get(sid)


def set_declared_target(sid: str, goal: str, subtype: str) -> str | None:
    """Record a declaration through the canonical after_game path. Fast (no models),
    so it runs synchronously — returns an error message, or None on success."""
    if goal not in GOALS:
        return "pick a goal from the five categories"
    if not subtype.strip():
        return "type the subtype you are building"
    proc = subprocess.run(
        [sys.executable, os.path.join(_SCRIPTS, "after_game.py"),
         "--session", sid, "--declare", f"{goal}/{subtype.strip()}"],
        cwd=_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return ((proc.stdout or "") + (proc.stderr or "")).strip()[-200:] or "declare failed"
    return None


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
    hint for what the background chain last did with it. Only the log's last
    ~256 KB is read: it grows for months and this runs on every session
    selection. (A line cut in half by the seek fails json parsing and is skipped;
    recent events are whole lines near the end.)"""
    try:
        with open(_RIG_LOG, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 256 * 1024))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
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

def _card(sid: str, labels: dict) -> dict:
    """One session's view data. `labels` is the caller's single read of
    labels.json — a refresh builds many cards, and re-reading the file per card
    was the old refresh's main cost."""
    report = after_game._prior_report(sid)
    state = derive_state(report, _has_evidence(sid), (job(sid) or {}).get("state"))
    verdict = (report or {}).get("verdict")
    return {
        "sid": sid,
        "state": state,
        "report": report,
        "verdict": verdict,
        "label": labels.get(sid, {}),
        "pngs": report_pngs(sid),
        "job": job(sid),
    }


# Ready-to-label sessions come first — they are the ones the human can act on;
# the rest (still processing, quarantined, gate-failed) sink to the bottom.
_STATE_ORDER = {"ready": 0, "labeling": 1, "processing": 2, "quarantined": 3,
                "gate failed": 4, "unprocessed": 5, "labeled": 6}


def inbox() -> list[dict]:
    labels = after_game._load_labels()
    _processable, needs_label = after_game._backlog_sessions()
    sids = inbox_from(needs_label, load_skipped())
    cards = [_card(sid, labels) for sid in sids]
    cards.sort(key=lambda c: (_STATE_ORDER.get(c["state"], 9), c["sid"]))
    return cards


def session_view(sid: str) -> dict:
    card = _card(sid, after_game._load_labels())
    card["summary"] = _summary(sid)
    card["inspect"] = _inspect_md(sid)
    card["rig_event"] = _latest_rig_event(sid)
    card["declared"] = declared_target(sid)
    return card


def history(labels: dict | None = None, skipped: list[str] | None = None) -> dict:
    if labels is None:
        labels = after_game._load_labels()
    if skipped is None:
        skipped = load_skipped()
    labeled = []
    for sid in sorted(labels):
        if not sid.startswith("fabric-"):
            continue                                 # scripted/other labels aren't sessions
        report = after_game._prior_report(sid)
        labeled.append({"sid": sid, "label": labels[sid],
                        "verdict": (report or {}).get("verdict")})
    return {"labeled": labeled,
            "skipped": [{"sid": sid, "label": labels.get(sid, {})}
                        for sid in sorted(skipped)]}


def status() -> dict:
    return after_game.readiness()


def snapshot() -> dict:
    """Everything all three tabs need, gathered in ONE pass: labels.json, the
    backlog scan, and the skip store are each read once, and every view is built
    from those in-memory copies. Plain data, no Tk — the GUI calls this from a
    worker thread so the window stays responsive while the files are read."""
    labels = after_game._load_labels()
    _processable, needs_label = after_game._backlog_sessions()
    skipped = load_skipped()
    cards = [_card(sid, labels) for sid in inbox_from(needs_label, skipped)]
    cards.sort(key=lambda c: (_STATE_ORDER.get(c["state"], 9), c["sid"]))
    return {
        "inbox": cards,
        "history": history(labels, skipped),
        "status": after_game.readiness(),
        "models": model_status(),
        "checklist": checklist_from(labels, _matcher_rows(),
                                    {c["sid"]: c["state"] for c in cards}, skipped),
    }


# ------------------------------------------------------- models + cascade audit

_MODELS_DIR = os.path.join(_ROOT, "models")

# (display name, files that must all exist — the .json with the facts is last)
_MODEL_ROWS = [
    ("belief heads (heads_v1)", ("heads_v1.npz", "heads_v1.json")),
    ("implicit arm (arm1)", ("arm1.npz", "arm1.json")),
    ("action decoder (decoder_v1)", ("decoder_v1.pt", "decoder_v1.json")),
    ("commit gate (gate_v1)", ("gate_v1.json",)),
]


def _model_facts(name: str, meta: dict) -> str:
    """One line of the numbers a human checks at a glance, per model."""
    if name.startswith("belief heads"):
        return (f"T {meta.get('temperature_delib', '?')}/{meta.get('temperature_heur', '?')}"
                f" · ε {meta.get('epsilon', '?')} · λg {meta.get('lambda_g', '?')}")
    if name.startswith("implicit arm"):
        return f"val NLL {meta.get('best_val_nll', '?')}"
    if name.startswith("action decoder"):
        parameters = meta.get("parameters")
        size = f"{parameters / 1e6:.1f}M params · " if parameters else ""
        return f"{size}stage B NLL {(meta.get('stage_b') or {}).get('best_holdout_token_nll', '?')}"
    if name.startswith("commit gate"):
        thresholds, fsm = meta.get("thresholds") or {}, meta.get("fsm") or {}
        return (f"θ1 {thresholds.get('theta_1', '?')} · suggest {fsm.get('theta_suggest', '?')}"
                f" · place {fsm.get('theta_place', '?')}")
    return ""


def _file_date(path: str) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def model_status(models_dir: str | None = None) -> list[dict]:
    """Which trained models are live on disk: present?, when trained, and one
    line of key facts from each model's json — plus whether the pre_cascade_a
    'before' bank exists (run_cascade_a.py fills it before every retrain)."""
    directory = models_dir or _MODELS_DIR
    rows = []
    for name, files in _MODEL_ROWS:
        paths = [os.path.join(directory, f) for f in files]
        present = all(os.path.exists(path) for path in paths)
        meta, when = {}, None
        if os.path.exists(paths[-1]):
            try:
                with open(paths[-1], encoding="utf-8") as handle:
                    meta = json.load(handle)
            except (OSError, ValueError):
                meta = {}
            when = meta.get("trained") or _file_date(paths[-1])
        rows.append({"name": name, "present": present, "trained": when,
                     "facts": _model_facts(name, meta) if present else "missing"})
    bank = os.path.join(directory, "pre_cascade_a")
    banked = glob.glob(os.path.join(bank, "*"))
    rows.append({"name": "'before' bank (pre_cascade_a)", "present": bool(banked),
                 "trained": _file_date(bank) if banked else None,
                 "facts": (f"{len(banked)} files — the pre-retrain models for before/after"
                           if banked else "missing (run_cascade_a.py creates it)")})
    return rows


def _matcher_rows() -> dict:
    """The labeling report's real-session rows, keyed by session id."""
    try:
        with open(after_game._SOURCE_B_REPORT, encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {row["session"]: row for row in report.get("real", {}).get("labeled", [])}


def cascade_checklist() -> dict:
    """The standalone gather for checklist_from (snapshot() feeds it directly)."""
    labels = after_game._load_labels()
    _processable, needs_label = after_game._backlog_sessions()
    skipped = load_skipped()
    cards = [_card(sid, labels) for sid in inbox_from(needs_label, skipped)]
    return checklist_from(labels, _matcher_rows(),
                          {c["sid"]: c["state"] for c in cards}, skipped)
