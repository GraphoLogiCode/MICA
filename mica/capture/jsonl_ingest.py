"""Read a real recording written as JSONL (one captured moment per line) into a CapturedSession.

The Fabric capture mod writes this format, one line per game tick, and the live stream
sends the same per-tick shape. The mod fills every client field each tick (keys, crosshair,
hotbar, menu, look); a partial source (e.g. a server-only capture) may leave some as None,
and the coverage check then reports those as the gaps they are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..contracts.b0 import (
    BlockEvent,
    BlockOp,
    BlockPos,
    ClientObservation,
    CrosshairTarget,
    FrameRef,
    InputState,
    ObservationPacket,
    PlayerPos,
    ServerObservation,
)
from ..contracts.manifest import CapturedSession, SessionManifest


def _block_pos(raw) -> BlockPos | None:
    return None if raw is None else BlockPos(int(raw[0]), int(raw[1]), int(raw[2]))


def _player_pos(raw) -> PlayerPos | None:
    return None if raw is None else PlayerPos(float(raw[0]), float(raw[1]), float(raw[2]))


def _input_state(raw) -> InputState | None:
    if raw is None:
        return None
    return InputState(
        keys=tuple(raw.get("keys", ())),
        mouse_buttons=tuple(raw.get("mouse_buttons", ())),
        mouse_dx=float(raw.get("mouse_dx", 0.0)),
        mouse_dy=float(raw.get("mouse_dy", 0.0)),
    )


def _crosshair(raw) -> CrosshairTarget | None:
    if raw is None:
        return None
    return CrosshairTarget(
        block_pos=_block_pos(raw.get("block_pos")), face=raw.get("face"), entity=raw.get("entity")
    )


def _frame(raw) -> FrameRef | None:
    if raw is None:
        return None
    return FrameRef(path=raw["path"], width=int(raw["width"]), height=int(raw["height"]))


def _block_event(raw) -> BlockEvent:
    return BlockEvent(
        event_id=int(raw["event_id"]),
        pos=_block_pos(raw["pos"]),
        block_type=raw["block_type"],
        op=BlockOp(raw["op"]),
        actor=raw["actor"],
    )


def _client(raw) -> ClientObservation:
    return ClientObservation(
        capture_wallclock_ms=float(raw["capture_wallclock_ms"]),
        pov_frame=_frame(raw.get("pov_frame")),
        input_state=_input_state(raw.get("input_state")),
        yaw=float(raw["yaw"]),
        pitch=float(raw["pitch"]),
        crosshair_target=_crosshair(raw.get("crosshair_target")),
        held_item=raw.get("held_item"),
        hotbar=None if raw.get("hotbar") is None else tuple(raw["hotbar"]),
        gui_open=raw.get("gui_open"),
    )


def _server(raw) -> ServerObservation:
    return ServerObservation(
        player_pos=_player_pos(raw.get("player_pos")),
        block_events=tuple(_block_event(event) for event in raw.get("block_events", ())),
        inventory_delta=tuple((str(item), int(count)) for item, count in raw.get("inventory_delta", ())),
        dimension=raw.get("dimension", "unknown"),
        biome=raw.get("biome", "unknown"),
    )


def _packet(raw) -> ObservationPacket:
    return ObservationPacket(
        tick=int(raw["tick"]),
        wallclock_ms=int(raw["wallclock_ms"]),
        client=_client(raw["client"]),
        server=_server(raw["server"]),
    )


@dataclass(frozen=True)
class JsonlSource:
    """A recording read from a raw JSONL file plus its manifest sidecar."""

    jsonl_path: str
    manifest_path: str

    def load(self) -> CapturedSession:
        with open(self.manifest_path, encoding="utf-8") as handle:
            meta = json.load(handle)
        manifest = SessionManifest(
            session_id=meta["session_id"],
            session_start_ms=int(meta["session_start_ms"]),
            mc_version=meta.get("mc_version", "unknown"),
            mod_version=meta.get("mod_version", "unknown"),
            event_schema_version=str(meta.get("event_schema_version", "1")),
        )
        packets: list[ObservationPacket] = []
        with open(self.jsonl_path, encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    packets.append(_packet(json.loads(stripped)))
        packet_tuple = tuple(packets)
        # A provisional manifest (declared_event_count = -1, or missing) means the
        # session never closed cleanly — e.g. the game crashed. Fall back to the events
        # actually present so the recording still loads for inspection instead of failing.
        declared = int(meta.get("declared_event_count", -1))
        if declared < 0:
            declared = sum(len(packet.server.block_events) for packet in packet_tuple)
        return CapturedSession(
            manifest=manifest,
            packets=packet_tuple,
            declared_event_count=declared,
        )
