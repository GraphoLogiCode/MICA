"""The live loop's middle: one moment in, everything knowable out — while the game runs.

This wires the per-stage machines together exactly the way the offline scripts chain
them, but per tick instead of per finished file:

    packet -> LiveGate (streaming sanity)
           -> SnapshotMonitor (shadow world for the replay-vs-snapshot proof)
           -> Evidence3DStream.harvest (remember the tick's block events)
           -> Evidence2DStream.feed -> for each released record:
                scored  -> B1 log -> Evidence3DStream.on_correction -> B2 log
                        -> fuse() (the verified join, now a live invariant)
                        -> tracker predict+correct -> fused + belief logs
                context -> B1 log -> a display-only "drift" belief line

Two beliefs on purpose: the AUTHORITATIVE belief advances only at corrections, with
one predict spanning the whole gap — bit-identical to what run_tracker computes over
the same records. The ~1 Hz drift lines are a prediction applied to a throwaway copy,
for a human (or gate) watching between actions; they never feed back. Lazy prediction
makes the two mathematically equal; keeping them separate keeps the log bit-exact.

Every log line is written and flushed the moment it exists, so a crash loses at most
the record in flight and the logs stay valid JSONL mid-session.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, TextIO

from .contracts.b1 import Evidence2D
from .contracts.b3 import fuse
from .contracts.serialize import (
    belief_to_dict, evidence2d_to_dict, evidence3d_to_dict, fused_to_dict,
)
from .intent.heads_v0 import likelihood
from .intent.tracker import (
    TrackerParams, category_marginal, correct, entropy, mode_marginal, predict,
    uniform_belief,
)
from .perception.evidence2d import Evidence2DStream
from .perception.evidence3d import Evidence3DStream
from .perception.voxel_replay import SnapshotMonitor
from .validation.evidence2d_check import check_record
from .validation.live_checks import LiveGate

_TICKS_PER_SECOND = 20.0
_PROBLEM_CAP = 50          # keep the first N contract problems; count the rest


class PixelHead(Protocol):
    """Stage 2's seam: fills h2d/s_goal on a scored record from recent raw frames.
    Not implemented in v1 — the config slot exists so wiring it later touches nothing
    here. A head that isn't ready must return the record unchanged."""

    def enrich(self, record: Evidence2D, frames: tuple[tuple[int, bytes], ...]) -> Evidence2D: ...


@dataclass(frozen=True)
class RecordSinks:
    """Where the proof logs go — any writable text streams (files live, buffers in tests).
    A None sink skips that log (D1-only mode has no B2/fused/belief to write)."""

    evidence2d: TextIO | None = None
    evidence3d: TextIO | None = None
    fused: TextIO | None = None
    belief: TextIO | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the pipeline needs, bundled (sinks + stages + knobs)."""

    params: TrackerParams
    sinks: RecordSinks
    d2: Evidence3DStream | None = None          # None = D1-only (no base snapshot yet)
    monitor: SnapshotMonitor | None = None
    pixel_head: PixelHead | None = None
    frame_ring: int = 128                        # raw frames kept for the pixel head


@dataclass
class PipelineCounts:
    scored: int = 0
    context: int = 0
    corrections: int = 0        # scored records that made it through fusion + tracker
    contract_problems: int = 0
    gate_problem_lines: int = 0


class LivePipeline:
    """Feed moments in tick order (the live ingest guarantees that); read logs out."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._d1 = Evidence2DStream()
        self._d2 = config.d2
        self._monitor = config.monitor
        self._gate = LiveGate()
        self._belief = uniform_belief()
        self._previous_tick = 0                  # last correction's tick (run_tracker's clock)
        self._consumed: set[int] = set()
        self._frames: deque[tuple[int, bytes]] = deque(maxlen=config.frame_ring)
        self._problems: list[str] = []           # first N contract/gate problems, for the summary
        self.counts = PipelineCounts()
        self.last_tick: int | None = None
        # Gate-facing state (D5's B6 inputs): the human's last known position, the last
        # attention read, and the tick of the last correction — which doubles as the
        # belief snapshot id (unique per correction, replayable from the belief log).
        self._last_pos = None
        self._last_focus = None
        # Model-channel readouts (display only, never fed back): the L2 norms of the
        # latest h2d/h3d and the latest s_goal — the "live numbers" a human watches
        # to see that both model channels are actually producing evidence.
        self._h2d_norm: float | None = None
        self._sgoal_last: tuple[float, ...] | None = None
        self._h3d_norm: float | None = None
        self._h3d_records = 0
        self._h3d_cells: int | None = None

    # ------------------------------------------------------------------ per moment

    def on_moment(self, packet, frame: bytes | None) -> tuple[str, ...]:
        """Process one tick; returns any problems this tick made visible (for display)."""
        self.last_tick = packet.tick
        problems = self._gate.check(packet)
        if problems:
            self.counts.gate_problem_lines += len(problems)
            self._remember_problems(problems)
        if packet.server.player_pos is not None:
            self._last_pos = packet.server.player_pos
        if frame is not None:
            self._frames.append((packet.tick, frame))
        if self._monitor is not None:
            self._monitor.apply_packet(packet)
        if self._d2 is not None:
            self._d2.harvest(packet)
        for record in self._d1.feed(packet):
            self._on_record(record)
        return problems

    def on_snapshot(self, snap_tick: int, cells: dict, region=None) -> list:
        """A new mid-session snapshot appeared on disk: check the shadow world now.
        With a region attached (region v3), a growth snapshot also feeds BOTH worlds:
        the monitor adopts it inside on_snapshot; the feature world adopts it here,
        so built-vs-terrain stays exact in freshly grown territory."""
        if self._monitor is None:
            return []
        if region is not None and self._d2 is not None:
            self._d2.extend_world(region, cells)
        return self._monitor.on_snapshot(snap_tick, cells, region)

    def notice_region(self, region) -> None:
        """The manifest says the frame grew (region v3): widen both worlds' escape
        test now — the adopted cells' base arrives with the growth snapshot file."""
        if self._monitor is not None:
            self._monitor.notice_region(region)
        if self._d2 is not None:
            self._d2.notice_region(region)

    @property
    def events_consumed(self) -> int:
        """How many block events corrections have consumed — the reseed guard."""
        return len(self._consumed)

    def reseed_structure(self, d2, monitor) -> None:
        """Swap in a freshly seeded structure stream + monitor after the capture
        region moved. Mod 0.0.4's region is PROVISIONAL until the first block event
        (it follows the player, rewriting the base snapshot, then freezes) — so a
        region change can only arrive while no evidence exists yet, and re-seeding
        is exact, not approximate. Consuming an event and THEN seeing the region
        move would be a capture bug; the assertion keeps that impossible to miss.
        Events harvested but not yet consumed carry over to the new stream."""
        assert not self._consumed, "region changed after block events were consumed"
        if self._d2 is not None:
            d2.remember(self._d2._events.values())
        self._d2 = d2
        self._monitor = monitor

    def finish(self) -> dict:
        """Stream over: release what the segmenter still holds, then summarize."""
        for record in self._d1.finish():
            self._on_record(record)
        if self._monitor is not None:
            self._monitor.drain_pending()   # no more snapshots: the hold is over
        return self.summary()

    # ------------------------------------------------------------------ per record

    def _on_record(self, record: Evidence2D) -> None:
        if record.scored and self._config.pixel_head is not None:
            record = self._config.pixel_head.enrich(record, tuple(self._frames))
        if record.h2d is not None:
            self._h2d_norm = math.sqrt(sum(v * v for v in record.h2d))
        if record.s_goal is not None:
            self._sgoal_last = record.s_goal
        issues = check_record(record)
        if issues:
            self.counts.contract_problems += len(issues)
            self._remember_problems(issues)
        self._emit(self._config.sinks.evidence2d, evidence2d_to_dict(record))
        self._last_focus = record.focus
        if not record.scored:
            self.counts.context += 1
            self._drift_line(record)
            return
        self.counts.scored += 1
        for event_id in record.event_ids:
            if event_id in self._consumed:   # D1 guarantees single consumption; a repeat is a bug
                raise AssertionError(f"event id {event_id} consumed twice (tick {record.tick_range[1] + 1})")
            self._consumed.add(event_id)
        if self._d2 is None:
            return                            # D1-only: no structure, no fusion, no belief
        b2 = self._d2.on_correction(record)
        if b2.h3d is not None:
            self._h3d_norm = math.sqrt(sum(v * v for v in b2.h3d))
            self._h3d_records += 1
            # The embedding and built_count come from the same D2 cache, so this IS
            # the number of cells the last h3d was computed over — the point-cloud
            # view compares against it to show cloud/embedding sync.
            self._h3d_cells = b2.global_feats.built_count
        self._emit(self._config.sinks.evidence3d, evidence3d_to_dict(b2))
        fused = fuse(record, b2)              # raises on a join mismatch — a pipeline bug
        self._emit(self._config.sinks.fused, fused_to_dict(fused))

        # The tracker step, exactly as run_tracker does it: one predict over the whole
        # gap since the last correction, then correct — bit-identical belief offline.
        params = self._config.params
        dt = max(fused.tick - self._previous_tick, 1) / _TICKS_PER_SECOND
        self._previous_tick = fused.tick
        self._belief = predict(self._belief, dt, params)
        self._belief, normalizer = correct(self._belief, likelihood(fused, fused.a_hat), params)
        assert abs(sum(self._belief.values()) - 1.0) < 1e-9, "belief mass drifted"
        assert normalizer >= params.epsilon / params.action_count - 1e-15, "floor violated"
        self.counts.corrections += 1
        self._emit(self._config.sinks.belief,
                   belief_to_dict(fused.tick, "correction", self._belief, normalizer))

    def _drift_line(self, record: Evidence2D) -> None:
        """Display-only belief between corrections: predict on a throwaway, never stored."""
        if self._config.sinks.belief is None or self._d2 is None:
            return
        tick = record.tick_range[1]
        dt = max(tick - self._previous_tick, 1) / _TICKS_PER_SECOND
        drifted = predict(self._belief, dt, self._config.params)   # predict is pure
        self._emit(self._config.sinks.belief, belief_to_dict(tick, "drift", drifted))

    # ------------------------------------------------------------------ readouts

    def status(self) -> dict:
        """One line's worth of live state for the runner's ~1 Hz stdout readout —
        and the gate-facing features D5's B6 snapshot reads (idle/proximity inputs
        plus the belief snapshot id, added 2026-07-04 per the D0-D4 readiness review F2)."""
        marginal = category_marginal(self._belief)
        top = max(marginal, key=marginal.get)
        gate = self._gate.summary()
        behavior = self._d1.current_behavior
        pos = self._last_pos
        focus = self._last_focus
        return {
            "tick": self.last_tick,
            "top_goal": top,
            "p_top_goal": marginal[top],
            "p_z1": mode_marginal(self._belief)[1],
            "entropy": entropy(self._belief),
            "scored": self.counts.scored,
            "context": self.counts.context,
            "corrections": self.counts.corrections,
            "missing_ticks": gate["tick_gaps"]["missing_ticks"],
            "crop_escapes": self._d2.crop_escapes if self._d2 is not None else 0,
            "quarantined": bool(self._monitor.quarantined) if self._monitor else False,
            "current_behavior": behavior.value if behavior is not None else None,
            "player_pos": (pos.x, pos.y, pos.z) if pos is not None else None,
            "focus_block": ((focus.block.x, focus.block.y, focus.block.z)
                            if focus is not None and focus.block is not None else None),
            "focus_dwell_ticks": focus.dwell_ticks if focus is not None else 0,
            "belief_snapshot_id": self._previous_tick if self.counts.corrections else None,
            # The model channels' live readouts (None until each first produces):
            "h2d_norm": round(self._h2d_norm, 3) if self._h2d_norm is not None else None,
            "s_goal_last": [round(v, 4) for v in self._sgoal_last] if self._sgoal_last else None,
            "h3d_norm": round(self._h3d_norm, 3) if self._h3d_norm is not None else None,
            "h3d_records": self._h3d_records,
            "h3d_cells": self._h3d_cells,
        }

    def summary(self) -> dict:
        """The pipeline's half of live_run.json (the runner adds ingest + gate-on-disk)."""
        return {
            "gate_live": self._gate.summary(),
            "records": {"scored": self.counts.scored, "context": self.counts.context,
                        "corrections": self.counts.corrections},
            "consumed_event_ids": len(self._consumed),
            "unconsumed_event_ids": list(self._d2.pending_event_ids) if self._d2 else [],
            "contract_problems": self.counts.contract_problems,
            "first_problems": list(self._problems),
            "quarantined": bool(self._monitor.quarantined) if self._monitor else False,
            "divergences": self._monitor.divergence_count if self._monitor else 0,
            # Evidence blindness: events the FEATURE world had to discard because the
            # build left the capture region — D2 reads zeros while this is nonzero.
            "crop_escapes": self._d2.crop_escapes if self._d2 is not None else 0,
        }

    # ------------------------------------------------------------------ helpers

    def _emit(self, sink: TextIO | None, payload: dict) -> None:
        if sink is None:
            return
        sink.write(json.dumps(payload) + "\n")
        sink.flush()                       # crash-safe: the line is on disk now

    def _remember_problems(self, problems) -> None:
        room = _PROBLEM_CAP - len(self._problems)
        if room > 0:
            self._problems.extend(list(problems)[:room])
