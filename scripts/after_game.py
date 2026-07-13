"""Process a capture after a game: gate -> evidence -> label -> inspect report.

    # one game you just played (builder's own label is required)
    python scripts/after_game.py --goal habitation --subtype treehouse
    python scripts/after_game.py --goal defense --subtype "square tower" --session fabric-20260706-...

    # the heavy half without the label — what the rig automation runs unattended
    # (relocate + bank + gate + run_d1 --pixels + run_d2 --h3d + inspect; label later)
    python scripts/after_game.py --session fabric-... --evidence-only

    # a queue of labels staged by the labeling tool — ONE matcher run for all of them
    python scripts/after_game.py --batch after_session_tool/staged_labels.json

    # declare what you're building (newest capture, or --session) — NOT a label:
    # it feeds eval + material planning only, never the belief (D7)
    python scripts/after_game.py --declare infrastructure/bridge

    # catch up a stack of already-captured games (labels come from labels.json)
    python scripts/after_game.py --all

    # is the accumulated batch worth a cascade A yet?
    python scripts/after_game.py --status

A labeled run on a session whose evidence was already regenerated (its
session_report.json records a PASS gate, no structure quarantine, and the evidence
files exist) SKIPS the gate/run_d1/run_d2 steps — the label completes in seconds
instead of redoing minutes of GPU work. `--redo-evidence` forces the full regen.

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
# Every model weight is local; a pipeline step's load must never depend on (or
# die on) a HF Hub round-trip — a rate-limited ping once killed run_d1 mid-chain.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from migrate_raw_layout import _session_moves                       # noqa: E402
from organize_raw import adopt_agent_scans                          # noqa: E402
from mica.capture import session_store                              # noqa: E402
from mica.capture.discovery import newest_capture                   # noqa: E402

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_RAW = session_store.RAW_ROOT
_LABELS = os.path.join(_RAW, "labels.json")
_LEDGER = os.path.join(_RAW, "cascade_status.json")
_SOURCE_B_REPORT = os.path.join(os.path.dirname(_RAW), "scripted", "source_b_report.json")
# Declared build targets live in their OWN file, never labels.json: presence of a
# session id in labels.json means "labeled" to the whole backlog logic, and a
# declaration is not a label. It is also quarantined by design (D7): evidence
# builders, trainers, and heads never open this file — it feeds only eval,
# material planning, and the belief-vs-declaration diagnostic.
_DECLARED = os.path.join(_RAW, "declared_targets.json")


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
    # One spelling per subtype: "House" and "house" must count as the same style
    # in the label distribution (a real session was saved capitalized).
    entry.update({"goal": goal, "subtype": subtype.strip().lower(),
                  "mode": entry.get("mode", "deliberate"),
                  "source": entry.get("source", "free build (real, pixels + h3d); "
                            "processed by after_game.py"),
                  "note": note})
    labels[session_id] = entry
    # Write-then-rename: labels.json is every session's label — a crash mid-write
    # would corrupt the one file the whole labeling chain starts from.
    tmp = _LABELS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(labels, handle, indent=1)
    os.replace(tmp, _LABELS)


def load_declared() -> dict:
    """Every declared build target, by session id (empty when none declared yet)."""
    try:
        with open(_DECLARED, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def declare_target(session_id: str, goal: str, subtype: str) -> None:
    """Record what the builder SAYS they are building — before, during, or after
    play. Re-declaring overwrites. This never touches labels.json."""
    declared = load_declared()
    declared[session_id] = {"goal": goal, "subtype": subtype.strip().lower(),
                            "declared_when": time.strftime("%Y-%m-%d %H:%M")}
    tmp = _DECLARED + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(declared, handle, indent=1)
    os.replace(tmp, _DECLARED)
    print(f"declared target for {session_id}: {goal}/{subtype.strip().lower()}")


def _verdict_for(session_id: str) -> dict | None:
    """This session's matcher verdict, read from the freshly written labeling report.

    An entry the matcher SKIPPED (recording not found, no snapshots) carries no
    label — returning it would make the session look labeled with an empty verdict,
    so a skipped entry reads as no verdict; the caller reports the reason."""
    if not os.path.exists(_SOURCE_B_REPORT):
        return None
    with open(_SOURCE_B_REPORT, encoding="utf-8") as handle:
        report = json.load(handle)
    for entry in report.get("real", {}).get("labeled", []):
        if entry["session"] == session_id:
            return entry if entry.get("label") is not None else None
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

def _prior_report(session_id: str) -> dict | None:
    """The session_report.json a previous after_game run left, if any — the marker
    that says the offline evidence regen already happened for this session."""
    path = os.path.join(session_store.session_dir(session_id), "session_report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _evidence_regenerated(session_id: str) -> bool:
    """True when the offline regen is RECORDED for this session and its evidence
    files exist — the proof that labeling may proceed without redoing the GPU work.
    The marker matters, not bare file existence: live-written (possibly quarantined)
    evidence must never be labeled against by accident. Reports from before the
    marker existed qualify through their verdict: a matcher verdict can only have
    been computed after a full regen."""
    prior = _prior_report(session_id)
    regen_recorded = prior is not None and (
        prior.get("evidence_ready")
        or (prior.get("gate_file_checks") == "PASS"
            and not prior.get("structure_quarantined")
            and prior.get("verdict") is not None))
    return (regen_recorded
            and os.path.exists(session_store.session_file(session_id, ".evidence2d.jsonl"))
            and os.path.exists(session_store.session_file(session_id, ".evidence3d.jsonl")))


def _write_inspect(session_id: str, gate_ok: bool, quarantined: bool,
                   verdict: dict | None, awaiting_label: bool = False) -> None:
    directory = session_store.session_dir(session_id)
    summary = {"session": session_id, "gate_file_checks": "PASS" if gate_ok else "FAIL",
               "structure_quarantined": quarantined, "verdict": verdict,
               # evidence_ready is what lets a later LABELED run skip the GPU regen;
               # awaiting_label marks the sessions the automation leaves for the human.
               "evidence_ready": gate_ok and not quarantined,
               "awaiting_label": awaiting_label}
    report_path = os.path.join(directory, "session_report.json")
    tmp = report_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(tmp, report_path)

    lines = [f"# {session_id}", ""]
    lines.append(f"- B0 gate (file checks): **{'PASS' if gate_ok else 'FAIL'}**")
    if quarantined:
        lines.append("- **STRUCTURE QUARANTINE** — unexplained voxel divergence; not proof-grade, not labeled.")
    if awaiting_label:
        lines.append("- **AWAITING BUILDER LABEL** — evidence is ready; finish with "
                     f"`after_game.py --session {session_id} --goal <category> --subtype <style>`")
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


def process_one(session_id: str, goal: str | None, subtype: str | None, no_vlm: bool,
                evidence_only: bool = False, redo_evidence: bool = False) -> dict:
    what = "evidence only — label later" if evidence_only else f"{goal}/{subtype}"
    print(f"\n=== processing {session_id} ({what}) ===")
    if not _wait_for_manifest(session_id):
        # A provisional manifest means the game is still open (or the mod never
        # finalized): processing now would gate-fail a session that simply hasn't
        # ended. Refuse, instead of recording a misleading FAIL against it.
        print("  manifest still provisional after 30 s — is the game still open? "
              "Quit the session first, then rerun.")
        return {"session": session_id, "ok": False, "reason": "manifest provisional"}
    _relocate(session_id)
    adopt_agent_scans(session_id)          # this session's scans follow it home
    _bank_quarantined_live(session_id)
    jsonl = session_store.session_jsonl(session_id)

    prior = _prior_report(session_id)
    evidence_done = not redo_evidence and _evidence_regenerated(session_id)
    if evidence_done:
        print("  evidence already regenerated offline — skipping gate/run_d1/run_d2"
              " (--redo-evidence forces the full regen)")
    else:
        if _run("b0_gate.py", jsonl) != 0:
            print("  B0 gate FAILED — not proof-grade; skipping evidence + label")
            _write_inspect(session_id, gate_ok=False, quarantined=False, verdict=None)
            return {"session": session_id, "ok": False, "reason": "gate fail"}

        if _run("run_d1.py", jsonl, "--pixels", "--overwrite") != 0:
            # A dead d1 used to be swallowed here (only d2 was checked): the chain
            # sailed on, stamped evidence_ready over a STALE evidence2d file, and
            # shipped mismatched b1/b2 — found 2026-07-12 when the contested-pairs
            # writer refused two such sessions. No report is written: the session
            # stays honestly un-ready instead of ready-with-garbage.
            print("  run_d1 exited non-zero — evidence NOT regenerated; fix the "
                  "pixel pass (or the frames) and rerun")
            return {"session": session_id, "ok": False, "reason": "run_d1 failed"}
        d2 = _run("run_d2.py", jsonl, "--h3d", "--overwrite")
        if d2 != 0:
            print("  run_d2 exited non-zero — STRUCTURE QUARANTINE (unrecoverable); skipping label")
            _write_inspect(session_id, gate_ok=True, quarantined=True, verdict=None)
            return {"session": session_id, "ok": False, "reason": "structure quarantine"}

    if evidence_only:
        if prior is not None and prior.get("verdict") is not None:
            # Already fully labeled — never clobber a recorded verdict with an
            # awaiting-label report (the automation may re-meet an old session).
            print("  already labeled — nothing to do (evidence + verdict recorded)")
            return {"session": session_id, "ok": True, "already_labeled": True}
        _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=None,
                       awaiting_label=True)
        print(f"  evidence ready — finish with: after_game.py --session {session_id}"
              " --goal <category> --subtype <style>")
        return {"session": session_id, "ok": True, "awaiting_label": True}

    _upsert_label(session_id, goal, subtype, "processed by after_game.py; matcher verdict below")
    label_args = ["--no-vlm"] if no_vlm else []
    if _run("label_finished_builds.py", *label_args) != 0:
        # A failed matcher must not masquerade as a completed label: the report on
        # disk is stale (or torn), so reading a verdict from it would attach the
        # wrong run's numbers to this session. Leave it awaiting its label.
        print("  label_finished_builds.py FAILED — no verdict recorded; the label is "
              "saved, rerun after_game.py --session " + session_id + " to retry")
        _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=None,
                       awaiting_label=True)
        return {"session": session_id, "ok": False, "reason": "matcher failed"}
    verdict = _verdict_for(session_id)
    if verdict is None:
        # The matcher ran but skipped this session (recording or snapshots missing).
        print("  matcher SKIPPED this session (no verdict in the report) — still "
              "awaiting a usable label run")
        _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=None,
                       awaiting_label=True)
        return {"session": session_id, "ok": False, "reason": "matcher skipped session"}
    _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=verdict)
    return {"session": session_id, "ok": True, "verdict": verdict}


def process_batch(staged_path: str, no_vlm: bool) -> dict:
    """Label a queue of sessions with ONE matcher run. The staged file is a JSON
    list of {"session", "goal", "subtype"} (the labeling tool writes it).

    Every queued session must already have its evidence regenerated (the rig's
    evidence-only pass does that): a session that is not ready is REFUSED with a
    printed reason, never silently sent through minutes of GPU regen. All accepted
    labels go into labels.json first, then label_finished_builds.py runs once for
    the whole queue — one model load instead of one per session — and each session
    gets the same verdict + inspect report a single-session run would write."""
    with open(staged_path, encoding="utf-8") as handle:
        staged = json.load(handle)
    accepted, refused = [], []
    for entry in staged:
        session_id = entry.get("session", "")
        goal, subtype = entry.get("goal", ""), entry.get("subtype", "")
        if not session_id or not goal or not subtype:
            refused.append((session_id or "<missing id>",
                            "entry is missing session/goal/subtype"))
            continue
        if not _wait_for_manifest(session_id):
            refused.append((session_id, "manifest still provisional — is the game still open?"))
            continue
        _relocate(session_id)
        adopt_agent_scans(session_id)
        _bank_quarantined_live(session_id)
        if not _evidence_regenerated(session_id):
            refused.append((session_id, "evidence not regenerated yet — run "
                            "--evidence-only on it first (or it is quarantined/gate-failed)"))
            continue
        accepted.append((session_id, goal, subtype))
    for session_id, reason in refused:
        print(f"  REFUSED {session_id}: {reason}")
    if not accepted:
        print("batch: nothing labelable in the queue")
        return {"ok": False, "labeled": [], "refused": refused}

    for session_id, goal, subtype in accepted:
        _upsert_label(session_id, goal, subtype,
                      "processed by after_game.py --batch; matcher verdict below")
    print(f"\nbatch: {len(accepted)} labels saved — running the matcher ONCE for all of them")
    matcher_ok = _run("label_finished_builds.py", *(["--no-vlm"] if no_vlm else [])) == 0

    failures = list(refused)
    labeled = []
    for session_id, goal, subtype in accepted:
        verdict = _verdict_for(session_id) if matcher_ok else None
        if verdict is None:
            # Same honesty rule as the single-session path: a failed or skipping
            # matcher must not masquerade as a completed label. The label itself
            # is saved; the session stays awaiting a usable label run.
            _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=None,
                           awaiting_label=True)
            failures.append((session_id,
                             "matcher failed" if not matcher_ok else "matcher skipped it"))
        else:
            _write_inspect(session_id, gate_ok=True, quarantined=False, verdict=verdict)
            labeled.append(session_id)
    print(f"\nbatch done: {len(labeled)} labeled, {len(failures)} refused/failed")
    return {"ok": not failures, "labeled": labeled, "refused": failures}


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
            # Perception already ran — but an automation-processed session may still
            # be waiting for its builder label (evidence-only mode): surface it, and
            # complete it if the label has arrived in labels.json meanwhile.
            #
            # NO REPORT AT ALL also surfaces (fixed 2026-07-11): live-first sessions
            # carry evidence from their first minute (run_live writes it during
            # play), so "evidence but no report" just means the post-session chain
            # hasn't finished — or died partway. Hiding those made every fresh
            # session invisible to the tool until its report landed, and a session
            # whose chain crashed stayed invisible forever.
            report = _prior_report(session_id)
            if report is None or report.get("awaiting_label"):
                (processable if session_id in labels else needs_label).append(session_id)
            continue
        (processable if session_id in labels else needs_label).append(session_id)
    return sorted(processable), sorted(needs_label)


def main() -> int:
    if "--status" in sys.argv:
        print_readiness()
        return 0

    no_vlm = "--no-vlm" in sys.argv
    staged_path = _flag("--batch")
    if staged_path:
        result = process_batch(staged_path, no_vlm)
        print_readiness()
        return 0 if result["ok"] else 1

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

    declared = _flag("--declare")
    if declared is not None:
        # e.g. after_game.py --session fabric-... --declare infrastructure/bridge
        # (no --session declares on the newest capture — mid-play declaration).
        from mica.contracts.goals import GOALS
        parts = declared.split("/", 1)
        if len(parts) != 2 or parts[0] not in GOALS or not parts[1].strip():
            print(f"--declare wants <category>/<subtype> with a real category "
                  f"({', '.join(GOALS)}), e.g. habitation/house")
            return 1
        declare_target(target, parts[0], parts[1])
        return 0

    evidence_only = "--evidence-only" in sys.argv
    redo_evidence = "--redo-evidence" in sys.argv
    goal, subtype = _flag("--goal"), _flag("--subtype")
    if goal is None or subtype is None:
        existing = _load_labels().get(target, {})
        goal, subtype = goal or existing.get("goal"), subtype or existing.get("subtype")
    if not evidence_only and (not goal or not subtype):
        print(f"{target}: needs the builder's own label — pass --goal <category> "
              "--subtype <style> (the matcher compares against it), or run "
              "--evidence-only to do the heavy half now and label later")
        return 1

    result = process_one(target, goal, subtype, no_vlm, evidence_only, redo_evidence)
    print_readiness()
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
