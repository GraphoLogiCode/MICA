"""The suggestion-acceptance report (D9 §3): one command per session, regenerable.

    python scripts/suggestion_acceptance.py [session-id ...]

For each session (default: every labeled real session with a gate trace): join the
body's voicing log with the gate trace's proposal-bearing reads and the capture's
human PLACE events, classify each voiced suggestion (followed exact/near/type,
contradicted, ignored), compute the unvoiced-control base rate, and report the lift.
Writes <session>.suggestion_acceptance.json next to the session's other artifacts.

Sessions recorded before 2026-07-17 have no `proposal_first` in their trace rows and
no voiced log — they report n=0 rather than failing (the signal starts with the
first advisory session after the D9 landing).
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture import session_store                                # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource                     # noqa: E402
from mica.contracts.b0 import BlockOp, is_agent_actor                 # noqa: E402
from mica.validation import suggestion_acceptance as accept           # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        rows = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                break   # torn tail line (the writer may still be running)
        return rows


def _session_report(session_id: str) -> dict | None:
    directory = session_store.session_dir(session_id)
    trace_rows = _load_jsonl(os.path.join(directory, f"{session_id}.gate_trace.jsonl"))
    if not trace_rows:
        return None

    jsonl = os.path.join(directory, f"{session_id}.jsonl")
    manifest = os.path.join(directory, f"{session_id}.manifest.json")
    session = JsonlSource(jsonl, manifest).load()
    human_places = [(p.tick, (e.pos.x, e.pos.y, e.pos.z), e.block_type)
                    for p in session.packets for e in p.server.block_events
                    if e.op is BlockOp.PLACE and not is_agent_actor(e.actor)]

    # tick <-> wallclock, from the capture itself (one JVM, one clock on this rig)
    ticks = sorted((p.wallclock_ms, p.tick) for p in session.packets)

    def tick_of_ms(ms: float) -> int:
        best = ticks[0][1]
        for wall, tick in ticks:
            if wall <= ms:
                best = tick
            else:
                break
        return best

    voiced_rows = []
    for path in glob.glob(os.path.join(_RAW, "agent-*.voiced.jsonl")):
        voiced_rows.extend(_load_jsonl(path))
    session_span = (ticks[0][0], ticks[-1][0] + 60_000)
    voiced_rows = [v for v in voiced_rows if session_span[0] <= v["ts"] <= session_span[1]]

    paired, unmatched = accept.pair_voicings(voiced_rows, trace_rows, tick_of_ms)
    result = accept.report(trace_rows, paired, human_places)
    result["session"] = session_id
    result["voicings_unmatched"] = unmatched
    out = os.path.join(directory, f"{session_id}.suggestion_acceptance.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main() -> int:
    targets = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not targets:
        from run_tracker import real_labeled_sessions
        targets = sorted(real_labeled_sessions())
    print(f"suggestion acceptance (D9 §3, v{accept.ACCEPTANCE_VERSION}) "
          f"over {len(targets)} session(s):")
    reported = 0
    for session_id in targets:
        result = _session_report(session_id)
        if result is None:
            print(f"  {session_id}: no gate trace — skipped")
            continue
        reported += 1
        voiced, control = result["voiced"], result["control_unvoiced"]
        print(f"  {session_id}: voiced {voiced['n']} (followed {voiced['followed_rate']})"
              f"  control {control['n']} (followed {control['followed_rate']})"
              f"  lift {result['lift']}"
              + (f"  [{result['voicings_unmatched']} voicings unmatched]"
                 if result["voicings_unmatched"] else ""))
    print(f"-> {reported} report(s) written next to their sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
