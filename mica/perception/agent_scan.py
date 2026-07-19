"""The agent-eye scan (h3d_scan): read what MICA_AI actually saw, split build from
terrain, and measure how much of the build the agent has covered.

The agent's body sweeps a ray grid from its own eyes four times a second and appends
every newly seen surface cell to a scan file (see the "Agent-Scan 3D Channel" design
note). That file is a raw sensor record — it holds terrain, the human's blocks, and
the agent's own blocks alike, because the sensor cannot know which is which. THIS
module is the mind side: it knows the session's pre-build base snapshot and the exact
built set, so it can say which scanned cells are part of the build (the channel's
cloud) and what fraction of the build the agent has seen (coverage). Nothing here
touches the D2 proof world — the scan is an evidence channel, never the verifier.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

Cell = tuple[int, int, int]

_AIR = "minecraft:air"


@dataclass(frozen=True)
class ScanSweep:
    """One 4 Hz sweep that saw something new: when, from where, and what."""

    ts: float
    tick: int | None
    pose: tuple[float, float, float, float, float]   # x, y, z, yaw, pitch
    cells: tuple[tuple[int, int, int, str], ...]     # x, y, z, block name


def _qualified(name: str) -> str:
    """mineflayer reports bare names ('oak_log'); the pipeline uses namespaced ids."""
    return name if ":" in name else f"minecraft:{name}"


def load_scan(path: str) -> list[ScanSweep]:
    """Every sweep in a scan file, in written order. A torn final line (the agent
    may still be writing) ends the read instead of failing it."""
    sweeps: list[ScanSweep] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except ValueError:
                break
            sweeps.append(ScanSweep(
                ts=float(raw.get("ts", 0)) / 1000.0,
                tick=raw.get("tick"),
                pose=tuple(raw.get("pose", (0, 0, 0, 0, 0))),
                cells=tuple((int(c[0]), int(c[1]), int(c[2]), _qualified(str(c[3])))
                            for c in raw.get("cells", ())),
            ))
    return sweeps


def accumulate(sweeps: Iterable[ScanSweep]) -> dict[Cell, str]:
    """Every cell the scan has seen, with the block it saw there — the LAST
    sighting wins. The body re-records a cell whenever its block changes
    (2026-07-19: the old once-forever rule made everything built on
    already-scanned ground invisible to the scan), so the newest record is the
    current truth and this accumulation must agree with it."""
    seen: dict[Cell, str] = {}
    for sweep in sweeps:
        for x, y, z, block in sweep.cells:
            seen[(x, y, z)] = block
    return seen


def scanned_build_cells(scanned: dict[Cell, str], built: dict[Cell, str]) -> dict[Cell, str]:
    """The channel's cloud: the scanned portion of the BUILT set — what the agent
    has seen of the build, in the exact cells-and-blocks form the shape embedding
    eats. Intersecting with the built set (not merely "differs from base") keeps
    the A7 rule intact: the agent's own blocks and world dynamics never enter the
    evidence cloud, exactly as they never enter the exact h3d channel's."""
    return {cell: built[cell] for cell in scanned.keys() & built.keys()}


def coverage(scanned: dict[Cell, str], built: dict[Cell, str]) -> tuple[int, int]:
    """(how many built cells the agent has seen, how many exist)."""
    return len(scanned.keys() & built.keys()), len(built)
