"""B0 acceptance gate CLI: strict pass/fail on the newest (or a named) capture.

    python scripts/b0_gate.py [session.jsonl]

The checks live in mica.validation.b0_gate (so they're tested). A capture passes only if
it is clean AND actually exercises B0 — block events and undistorted frames, not just a
walk-around. Runtime checks (live stream, F9 pause, F8 HUD) are manual; see the runbook.
Exit code 0 = PASS, 1 = FAIL.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.discovery import newest_capture     # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource     # noqa: E402
from mica.validation.b0_gate import gate_checks       # noqa: E402

_RAW = os.path.join(os.path.dirname(__file__), "..", "capture", "raw")


def main() -> int:
    jsonl = sys.argv[1] if len(sys.argv) > 1 else newest_capture(_RAW)
    if not jsonl or not os.path.exists(jsonl):
        print("no capture found")
        return 1
    session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
    checks = gate_checks(session, os.path.dirname(os.path.abspath(jsonl)))

    print(f"B0 GATE  {os.path.basename(jsonl)}")
    overall = True
    for name, ok, detail in checks:
        overall = overall and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")
    print(f"\n  OVERALL (file checks): {'PASS' if overall else 'FAIL'}")
    if overall:
        print("  next: run the runtime checks (live probe, F9 pause, F8 HUD) to finish the gate.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
