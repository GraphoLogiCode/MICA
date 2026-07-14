"""The yaw reset guard (debt F5/N-3, resolved by measurement 2026-07-13).

Real captures carry Minecraft's ACCUMULATING yaw (sessions span multiple full
turns; measured -1848°..559°), so raw differences are correct — the one artifact
is a rare single-tick client re-anchor (~±360·k on respawn/teleport). The guard
ignores that jump in the tick classifier and subtracts it from the window's
yaw_delta; every window without a reset computes the exact same float as before.
"""
from mica.contracts.b0 import (
    ClientObservation, CrosshairTarget, InputState, ObservationPacket, PlayerPos,
    ServerObservation,
)
from mica.perception.evidence2d import _camera_moved, _state_feats


def _packet(tick: int, yaw: float, pitch: float = 0.0) -> ObservationPacket:
    client = ClientObservation(
        capture_wallclock_ms=float(tick * 50),
        pov_frame=None,
        input_state=InputState(keys=(), mouse_buttons=(), mouse_dx=0.0, mouse_dy=0.0),
        yaw=yaw,
        pitch=pitch,
        crosshair_target=CrosshairTarget(block_pos=None, face=None, entity=None),
        held_item="minecraft:stone",
        hotbar=("minecraft:stone",),
        gui_open=False,
    )
    server = ServerObservation(
        player_pos=PlayerPos(0.0, 64.0, 0.0),
        block_events=(),
        inventory_delta=(),
        dimension="overworld",
        biome="plains",
    )
    return ObservationPacket(tick=tick, wallclock_ms=tick * 50, client=client, server=server)


def test_normal_turns_still_read_as_camera_motion():
    assert _camera_moved(_packet(2, yaw=-100.0).client, _packet(1, yaw=-110.0).client)
    # accumulating yaw far from zero behaves the same
    assert _camera_moved(_packet(2, yaw=-1240.0).client, _packet(1, yaw=-1250.0).client)


def test_a_reset_jump_is_not_a_look():
    # a respawn re-anchor: one tick, ~a full turn of "motion" that never happened
    assert not _camera_moved(_packet(2, yaw=-38.0).client, _packet(1, yaw=-398.0).client)
    # but pitch motion on the same tick still counts
    assert _camera_moved(_packet(2, yaw=-38.0, pitch=20.0).client,
                         _packet(1, yaw=-398.0, pitch=0.0).client)


def test_window_yaw_delta_is_bit_identical_without_a_reset():
    window = [_packet(t, yaw=-1236.4 + 7.3 * t) for t in range(1, 8)]
    feats = _state_feats(window, recent=())
    assert feats.yaw_delta == window[-1].client.yaw - window[0].client.yaw


def test_window_yaw_delta_subtracts_the_reset_jump():
    # turn +30° in small steps, then the client re-anchors by -360°, then +10° more:
    # the human turned 40°; the raw last-first says -320°.
    yaws = [0.0, 10.0, 20.0, 30.0, 30.0 - 360.0, 40.0 - 360.0]
    window = [_packet(t + 1, yaw=y) for t, y in enumerate(yaws)]
    feats = _state_feats(window, recent=())
    assert abs(feats.yaw_delta - 40.0) < 1e-9
