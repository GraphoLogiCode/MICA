"""Audit gate_trace.jsonl files against the D5 rules — the gate's own proof-check.

    # every trace under capture/ (the default)
    python scripts/audit_gate_trace.py

    # one session's trace
    python scripts/audit_gate_trace.py capture/raw/2026-07-08/fabric-.../fabric-....gate_trace.jsonl

A gate trace is the gate's diary: one line per read, saying what the gate saw and
what it decided. This script re-reads that diary and checks that every decision
follows the frozen rules. It does NOT re-run the decoder or the belief — it only
checks that the numbers already in the trace are consistent with each other and
with the rules in models/gate_v1.json. That makes it fast and safe to run anywhere.

What it checks, in plain terms:

  parse      every line is valid JSON with the fields a trace row must have
  order      read numbers count 1, 2, 3, ... and game ticks never go backwards
  states     only the five allowed states appear; EXECUTE_CHUNK is banned outright;
             PLACE_LOW_RISK is banned under the demo config (pass --allow-place
             only for counterfactual artifacts, where placing is simulated)
  commits    committed_actions stays empty under the demo config, and if anything
             ever committed, no position ever fired below its own threshold
  snapshot   every full read carries the belief snapshot id that explains it, and
             (when the session's belief log is on disk) that id really is a
             correction tick in that log
  staircase  K_commit obeys the +1-per-read growth / instant-shrink rule, and the
             per-position numbers support the raw K the row claims
  fsm        the chosen state re-derives from the row's own inputs: the proximity
             veto always yields, an active human always observes, the confidence
             thresholds decide suggest/observe and preview/place, and the
             safety-lattice hysteresis (up slowly, down instantly) replays exactly
  demo pin   "placement disabled by config" is counted, proving the pin engaged

Exit code 0 when every trace passes; 1 when any check fails. WARNs don't fail the
run — they mark things worth a look (rounding-boundary reads, missing belief logs).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mica.contracts.b6 import GateState                              # noqa: E402
from mica.gate.fsm import LatticeHysteresis                          # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE_META = os.path.join(_ROOT, "models", "gate_v1.json")

# Rounding guard: the trace stores p_top_goal / p_z1 rounded to 4 decimals, so a
# recomputed confidence can differ from the live one by ~1e-4. A threshold
# comparison inside this band is inconclusive — WARN, never FAIL.
_EPS = 2e-3
# Same idea for the logged proximity (rounded to 2 decimals): a row that says
# exactly 4.0 may truly have been 4.004, just outside the veto radius.
_EPS_PROX = 0.005

_LIVE_STATES = {"observe", "suggest", "preview", "place_low_risk", "yield"}


class TraceAudit:
    """All findings for one trace file, collected then printed."""

    def __init__(self, path: str):
        self.path = path
        self.fails: list[str] = []
        self.warns: list[str] = []
        self.state_mix: dict[str, int] = {}
        self.disabled_placements = 0        # "placement disabled by config" reads
        self.degraded = 0                   # reads that could not gate

    def fail(self, k, message: str) -> None:
        self.fails.append(f"read {k}: {message}")

    def warn(self, k, message: str) -> None:
        self.warns.append(f"read {k}: {message}")


def _load_meta() -> dict:
    with open(_GATE_META, encoding="utf-8") as handle:
        return json.load(handle)


def _correction_ticks(trace_path: str) -> set[int] | None:
    """The correction ticks in the session's belief log, or None when no log is
    on disk to check against. A snapshot id in the trace must be one of these —
    that is what makes every decision replayable from the belief that caused it."""
    marker = ".gate_trace"
    if marker not in os.path.basename(trace_path):
        return None
    base = trace_path[: trace_path.index(marker)]
    belief_log = base + ".belief.jsonl"
    if not os.path.exists(belief_log):
        return None
    ticks = set()
    with open(belief_log, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") == "correction" and row.get("tick") is not None:
                ticks.add(int(row["tick"]))
    return ticks or None


def _segments(rows: list[dict]) -> list[list[dict]]:
    """Split a trace file into runs, because hysteresis memory never spans one.

    Two trace shapes exist. The POOLED counterfactual trace (run_gate.py) tags
    every row with its session — split when the session changes. A LIVE or
    replay trace has no session field; each runner counts its reads 1, 2, 3, ...,
    so the counter restarting at 1 mid-file means a SECOND run appended to the
    same file (the trace is opened in append mode)."""
    segments: list[list[dict]] = []
    if rows and "session" in rows[0]:
        current = object()                  # never equals a real session id
        for row in rows:
            if row.get("session") != current:
                current = row.get("session")
                segments.append([])
            segments[-1].append(row)
        return segments
    for row in rows:
        if row.get("k") == 1 or not segments:
            segments.append([])
        segments[-1].append(row)
    return segments


def _check_row_order(audit: TraceAudit, rows: list[dict], pooled: bool) -> None:
    last_tick = None
    for index, row in enumerate(rows):
        # The live runner counts reads 1, 2, 3, ...; the pooled trace's k is the
        # fused-record index and may step (train-seen sessions are thinned), so
        # only the live shape gets the exact-count check.
        expected_k = index + 1
        if not pooled and row.get("k") != expected_k:
            audit.fail(row.get("k"), f"read counter is {row.get('k')}, expected "
                       f"{expected_k} (one row per read, D5 S7)")
        tick = row.get("tick")
        if tick is not None and last_tick is not None and tick < last_tick:
            audit.fail(row.get("k"), f"tick went backwards ({last_tick} -> {tick})")
        if tick is not None:
            last_tick = tick


def _trace_thetas(rows: list[dict], meta: dict) -> tuple[float, float, bool]:
    """(theta_suggest, theta_place, stale). The thresholds the trace was actually
    written under, recovered from its own reason strings — an old trace must be
    judged by its own frozen config, not today's. `stale` says the two differ."""
    suggest = place = None
    for row in rows:
        reason = row.get("reason") or ""
        if suggest is None:
            match = re.search(r"theta_suggest ([0-9.]+)", reason)
            if match:
                suggest = float(match.group(1))
        if place is None:
            match = re.search(r"theta_place ([0-9.]+)", reason)
            if match:
                place = float(match.group(1))
        if suggest is not None and place is not None:
            break
    current_suggest = meta["fsm"]["theta_suggest"]
    current_place = meta["fsm"]["theta_place"]
    stale = ((suggest is not None and abs(suggest - current_suggest) > 1e-9)
             or (place is not None and abs(place - current_place) > 1e-9))
    return (suggest if suggest is not None else current_suggest,
            place if place is not None else current_place, stale)


def _check_states_and_commits(audit: TraceAudit, rows: list[dict],
                              allow_place: bool) -> None:
    for row in rows:
        k = row.get("k")
        state = row.get("chosen_state")
        audit.state_mix[state] = audit.state_mix.get(state, 0) + 1
        if state not in _LIVE_STATES:
            audit.fail(k, f"illegal state {state!r}")
        if state == "place_low_risk" and not allow_place:
            audit.fail(k, "PLACE_LOW_RISK under the demo config -- the S9 pin failed")
        if row.get("committed_actions"):
            # Nothing may commit in the demo config at all; and if a trace from a
            # placing config is audited, every committed action must be reversible
            # and traceable to the belief that caused it (D5 §7 pass criteria).
            if not allow_place:
                audit.fail(k, "committed_actions is non-empty under the demo config")
            if row.get("belief_snapshot_id") is None:
                audit.fail(k, "committed actions with no belief_snapshot_id "
                           "(untraceable commit)")
            for action in row["committed_actions"]:
                if not action.get("reversible", False):
                    audit.fail(k, "an irreversible action was committed (v1 bans this)")
        if "placement disabled by config" in (row.get("reason") or ""):
            audit.disabled_placements += 1


def _check_snapshots(audit: TraceAudit, rows: list[dict],
                     corrections: set[int] | None) -> None:
    for row in rows:
        k, snapshot = row.get("k"), row.get("belief_snapshot_id")
        degraded = "inputs_snapshot" not in row
        if snapshot is None:
            # A degraded read may predate the first correction; a full read
            # deciding without a snapshot would be an unexplainable decision.
            (audit.warn if degraded else audit.fail)(
                k, "no belief_snapshot_id" + (" (degraded read)" if degraded else ""))
            continue
        if corrections is not None and snapshot not in corrections:
            audit.fail(k, f"belief_snapshot_id {snapshot} is not a correction tick "
                       "in the session's belief log")


def _check_staircase(audit: TraceAudit, rows: list[dict], c_min: float) -> None:
    """Replay the K_commit growth rule and re-check each row's per-position math."""
    last_held = 0
    for row in rows:
        k = row.get("k")
        if "inputs_snapshot" not in row:
            continue                        # degraded read: the staircase never ran
        held = row.get("K_commit", 0)
        raw = row.get("raw_k_commit")
        positions = row.get("per_position") or []
        if not positions:
            # No proposal was decoded this read — the staircase (and its memory)
            # was not consulted, so the growth rule does not tick here.
            if held != 0:
                audit.fail(k, f"K_commit {held} with no per-position rows")
            continue
        if raw is not None:
            # The live shape logs the pre-hysteresis K too, so the rule replays
            # exactly: held = min(raw, previous + 1).
            expected = min(raw, last_held + 1)
            if held != expected:
                audit.fail(k, f"K_commit {held} breaks the staircase rule "
                           f"(raw {raw}, previous {last_held} -> expected {expected})")
        elif held > last_held + 1:
            # The pooled shape logs only the held K: growth beyond +1 per read is
            # still checkable, instant shrink is not (the raw value is gone).
            audit.fail(k, f"K_commit jumped {last_held} -> {held} "
                       "(authority may grow at most +1 per read)")
        last_held = held
        # The committed K must be an unbroken prefix of positions that passed
        # their own thresholds (materials may cap it shorter, never longer).
        ok_prefix = 0
        for entry in positions:
            if entry.get("ok"):
                ok_prefix += 1
            else:
                break
        claimed = raw if raw is not None else held
        if claimed > ok_prefix:
            audit.fail(k, f"claimed K {claimed} exceeds the passing prefix ({ok_prefix})")
        feasible = row.get("feasible_prefix")
        if raw is not None and raw < ok_prefix and (feasible is None or raw != feasible):
            audit.warn(k, f"raw_k_commit {raw} below the passing prefix ({ok_prefix}) "
                       "without a materials cap explaining it")
        for entry in positions[:held]:
            if entry.get("token_conf", 0.0) < c_min:
                audit.fail(k, f"position {entry.get('j')} committed with token_conf "
                           f"{entry.get('token_conf')} below c_min {c_min}")
            if entry.get("p_star", 0.0) < entry.get("theta", math.inf):
                audit.fail(k, f"position {entry.get('j')} committed with p* "
                           f"{entry.get('p_star')} below theta {entry.get('theta')}")


def _expected_candidate(row: dict, fsm_cfg: dict) -> tuple[str | None, str | None]:
    """Re-derive the candidate from the row's own inputs. Returns (candidate, None)
    when the inputs pin it down, or (None, why) when they genuinely can't."""
    snap = row["inputs_snapshot"]
    reason = row.get("reason") or ""
    conf = snap["p_top_goal"] * (1.0 - snap["p_z1"])

    # 1. the hard veto. The row logs the human-to-target distance; the workspace
    # half of the veto (target near the focus block) is only visible through the
    # reason text, so a yield with a workspace reason is accepted as-is. The
    # logged distance is rounded, so right at the radius it can't be judged.
    proximity = snap.get("proximity")
    if proximity is not None:
        margin = proximity - fsm_cfg["proximal_radius"]
        if margin <= -_EPS_PROX:
            return "yield", None
        if abs(margin) < _EPS_PROX and "proximity veto" not in reason:
            return None, "distance sits on the veto radius (rounding band)"
    if "proximity veto" in reason:
        return "yield", None

    # 2. the human is mid-action
    if row.get("idle_state") == "active":
        return "observe", None

    # 3. the safe window. K_commit == 0 (or a say-first proposal, which leaves no
    # target — visible only through the reason) splits observe/suggest on the
    # discounted confidence; K_commit >= 1 splits preview/place.
    k_commit = row.get("K_commit", 0)
    if "decoder proposes nothing" in reason:
        return "observe", None
    suggest_split = k_commit == 0 or "theta_suggest" in reason
    if suggest_split:
        margin = conf - fsm_cfg["theta_suggest"]
        if abs(margin) <= _EPS:
            return None, "confidence sits on the suggest threshold (rounding band)"
        return ("suggest" if margin > 0 else "observe"), None
    if snap.get("reversibility") is not True:
        return "preview", None
    margin = conf - fsm_cfg["theta_place"]
    if abs(margin) <= _EPS:
        return None, "confidence sits on the place threshold (rounding band)"
    if margin > 0:
        # reversible prefix + confidence clears theta_place: place when the config
        # allows it, preview when the demo pin holds it down.
        return ("place_low_risk" if fsm_cfg.get("place_low_risk_enabled") else "preview"), None
    return "preview", None


def _check_fsm(audit: TraceAudit, rows: list[dict], fsm_cfg: dict,
               m_consecutive: int) -> None:
    """Re-derive every candidate, then replay the safety-lattice hysteresis over
    the candidate column and demand the chosen column matches exactly."""
    hysteresis = LatticeHysteresis(m_consecutive, GateState.OBSERVE)
    for row in rows:
        k = row.get("k")
        if "inputs_snapshot" not in row:
            audit.degraded += 1
            continue                        # degraded reads never touch the FSM
        candidate = row.get("candidate_state")
        expected, undecidable = _expected_candidate(row, fsm_cfg)
        if undecidable:
            audit.warn(k, undecidable)
        elif expected is not None and candidate != expected:
            audit.fail(k, f"candidate {candidate!r} but the row's inputs say "
                       f"{expected!r} (reason logged: {row.get('reason')!r})")
        try:
            simulated = hysteresis.read(GateState(candidate))
        except ValueError:
            audit.fail(k, f"candidate state {candidate!r} is not a B6 state")
            continue
        if simulated.value != row.get("chosen_state"):
            audit.fail(k, f"hysteresis replay chose {simulated.value!r} but the "
                       f"trace says {row.get('chosen_state')!r}")


def audit_trace(path: str, meta: dict, allow_place: bool) -> TraceAudit:
    audit = TraceAudit(path)
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                audit.fail(f"line {number}", "not valid JSON")
    if not rows:
        audit.warn("-", "empty trace")
        return audit

    # A replay's beliefs are recomputed in memory and never written to disk, so
    # its snapshot ids can't be checked against the live belief log (which may
    # also have dropped moments the offline evidence kept).
    is_replay = ".replay." in os.path.basename(path)
    corrections = None if is_replay else _correction_ticks(path)

    pooled = "session" in rows[0]
    segments = _segments(rows)
    if len(segments) > 1 and not pooled:
        audit.warn("-", f"{len(segments)} runs appended into one trace file "
                   "(the read counter restarts) -- each run audited separately, "
                   "but the file should be one run per trace")
    for segment in segments:
        theta_suggest, theta_place, stale = _trace_thetas(segment, meta)
        if stale:
            audit.warn(segment[0].get("k"),
                       f"written under superseded thresholds (suggest {theta_suggest}, "
                       f"place {theta_place}) -- judged by its own config, but the "
                       "artifact is stale against models/gate_v1.json")
        fsm_cfg = dict(meta["fsm"])
        fsm_cfg["theta_suggest"] = theta_suggest
        fsm_cfg["theta_place"] = theta_place
        fsm_cfg["place_low_risk_enabled"] = allow_place
        _check_row_order(audit, segment, pooled)
        _check_states_and_commits(audit, segment, allow_place)
        _check_snapshots(audit, segment, corrections)
        _check_staircase(audit, segment, meta["thresholds"]["c_min"])
        _check_fsm(audit, segment, fsm_cfg, meta["thresholds"]["m_consecutive"])
    return audit


def main() -> int:
    allow_place = "--allow-place" in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not paths:
        paths = sorted(glob.glob(os.path.join(_ROOT, "capture", "**",
                                              "*.gate_trace*.jsonl"), recursive=True))
    if not paths:
        print("no gate traces found")
        return 1

    meta = _load_meta()
    any_fail = False
    for path in paths:
        audit = audit_trace(path, meta, allow_place)
        reads = sum(audit.state_mix.values())
        mix = " | ".join(f"{s}:{n}" for s, n in
                         sorted(audit.state_mix.items(), key=lambda kv: -kv[1]))
        verdict = "FAIL" if audit.fails else ("WARN" if audit.warns else "PASS")
        any_fail = any_fail or bool(audit.fails)
        print(f"{verdict}  {os.path.relpath(path, _ROOT)}")
        print(f"      {reads} reads   {mix}")
        if audit.degraded:
            print(f"      {audit.degraded} degraded read(s) (gate alive, could not decide)")
        if audit.disabled_placements:
            print(f"      demo pin engaged {audit.disabled_placements}x "
                  "(placement wanted, config said no)")
        for message in audit.fails:
            print(f"      FAIL  {message}")
        for message in audit.warns[:10]:
            print(f"      warn  {message}")
        if len(audit.warns) > 10:
            print(f"      ... {len(audit.warns) - 10} more warns")
    print(f"\n{'SOME TRACES FAILED' if any_fail else 'all traces pass'} "
          f"({len(paths)} audited)")
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
