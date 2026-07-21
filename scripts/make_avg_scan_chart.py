"""The average agent-scan chart — one picture of how the agent's view fills in.

    python scripts/make_avg_scan_chart.py

Every finished session already has an agent_scan_report.json with a coverage
curve: after each scan sweep, how many of the builder's blocks the agent has
seen so far. Sessions have different lengths, so each curve is first put on a
shared 0-100% "session progress" axis, then all curves are averaged point by
point. Sessions whose capture was quarantined are left out, because their
block count (the bottom of the coverage fraction) cannot be trusted.

Writes report/avg_agent_scan.png (the chart) and report/avg_agent_scan.json
(the numbers behind it, so the chart can be checked or redrawn later).
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_OUT_PNG = os.path.join(_ROOT, "report", "avg_agent_scan.png")
_OUT_JSON = os.path.join(_ROOT, "report", "avg_agent_scan.json")

# The paper's colors, same as the poster and the evidence figures.
_BLUE = "#2E6EB4"
_GRAY = "#8C8C8C"

_GRID = np.linspace(0.0, 1.0, 101)   # shared progress axis: 0% .. 100%


def _quarantined(session_dir: str) -> bool:
    """A quarantined session had a broken capture — its block totals are not
    safe to divide by, so it stays out of the average."""
    path = os.path.join(session_dir, "session_report.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as handle:
        return bool(json.load(handle).get("structure_quarantined"))


def _load_curves() -> tuple[list[dict], list[str]]:
    """Read every per-session scan report and keep the usable coverage curves."""
    kept, skipped = [], []
    for path in sorted(glob.glob(os.path.join(_RAW, "*", "*", "*.agent_scan_report.json"))):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        sid = report.get("session_id", os.path.basename(path))
        if _quarantined(os.path.dirname(path)):
            skipped.append(f"{sid}: quarantined capture")
            continue
        curve = report.get("coverage_curve") or []
        total = report.get("built_total") or 0
        if total <= 0 or len(curve) < 5:
            skipped.append(f"{sid}: no build or too few sweeps ({len(curve)})")
            continue
        # Progress = how far along the sweep sequence we are (0..1).
        # Coverage = seen blocks / all blocks the builder placed (0..1).
        progress = np.linspace(0.0, 1.0, len(curve))
        coverage = np.array([point["built_seen"] / total for point in curve])
        kept.append({"session_id": sid,
                     "sweeps": len(curve),
                     "final_coverage": round(float(coverage[-1]), 4),
                     "on_grid": np.interp(_GRID, progress, coverage)})
    return kept, skipped


def main() -> int:
    kept, skipped = _load_curves()
    if not kept:
        print("no usable scan reports found")
        return 1
    stack = np.vstack([session["on_grid"] for session in kept])
    mean = stack.mean(axis=0)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    for session in kept:
        ax.plot(_GRID * 100, session["on_grid"] * 100,
                color=_GRAY, linewidth=0.9, alpha=0.35, zorder=1)
    ax.plot(_GRID * 100, mean * 100, color=_BLUE, linewidth=2.4, zorder=3)

    # Direct labels instead of a legend box, like the other paper figures.
    ax.annotate(f"mean of {len(kept)} sessions",
                xy=(100, mean[-1] * 100), xytext=(66, mean[-1] * 100 - 13),
                color=_BLUE, fontsize=11)
    ax.annotate("one line = one session",
                xy=(55, 30), color=_GRAY, fontsize=10)
    ax.annotate(f"{mean[-1] * 100:.0f}%",
                xy=(100, mean[-1] * 100), xytext=(96.5, mean[-1] * 100 + 3),
                color=_BLUE, fontsize=11, fontweight="bold")

    ax.set_xlabel("session progress (% of the agent's scan sweeps)", fontsize=11)
    ax.set_ylabel("build coverage (% of placed blocks seen)", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(_OUT_PNG)

    with open(_OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump({
            "sessions_used": [{key: session[key] for key in
                               ("session_id", "sweeps", "final_coverage")}
                              for session in kept],
            "sessions_skipped": skipped,
            "grid_progress": [round(float(x), 2) for x in _GRID],
            "mean_coverage": [round(float(y), 4) for y in mean],
            "mean_final_coverage": round(float(mean[-1]), 4),
        }, handle, indent=2)

    print(f"averaged {len(kept)} sessions, skipped {len(skipped)}:")
    for reason in skipped:
        print(f"  - {reason}")
    print(f"mean final coverage {mean[-1]:.3f}")
    print(f"chart -> {os.path.relpath(_OUT_PNG, _ROOT)}")
    print(f"data  -> {os.path.relpath(_OUT_JSON, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
