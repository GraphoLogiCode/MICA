"""Rebuild the proof file from a pretend recording.

The project rule is simple: if there's no proof file, the step isn't done.
Run from the project root: python scripts/make_proof_logs.py
"""
from __future__ import annotations

import os
import sys

# Let this script find the project code when it's run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.validation.sync_report import build_sync_report, write_sync_report

_PROOF_DIR = os.path.join(os.path.dirname(__file__), "..", "proof_logs")


def main() -> None:
    session = generate_session(wall_row_build())
    report = build_sync_report(session)
    out_path = os.path.normpath(os.path.join(_PROOF_DIR, "sync_report.json"))
    write_sync_report(report, out_path)
    print(f"wrote {out_path} (passed={report.passed})")
    for result in report.coverage:
        status = "ok" if result.satisfied else f"MISSING {list(result.missing)}"
        print(f"  coverage [{result.reader}]: {status}")


if __name__ == "__main__":
    main()
