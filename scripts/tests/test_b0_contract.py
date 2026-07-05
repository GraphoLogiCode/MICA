import dataclasses

import pytest

from mica.contracts.b0 import (
    BlockEvent,
    BlockOp,
    BlockPos,
    ClientObservation,
    CrosshairTarget,
    InputState,
    ObservationPacket,
    PlayerPos,
    ServerObservation,
)


def _minimal_packet() -> ObservationPacket:
    client = ClientObservation(
        capture_wallclock_ms=0.0,
        pov_frame=None,
        input_state=InputState(keys=(), mouse_buttons=(), mouse_dx=0.0, mouse_dy=0.0),
        yaw=0.0,
        pitch=0.0,
        crosshair_target=CrosshairTarget(block_pos=None, face=None, entity=None),
        held_item="minecraft:oak_planks",
        hotbar=("minecraft:oak_planks",),
        gui_open=False,
    )
    event = BlockEvent(
        event_id=0,
        pos=BlockPos(0, 64, 0),
        block_type="minecraft:oak_planks",
        op=BlockOp.PLACE,
        actor="tester",
    )
    server = ServerObservation(
        player_pos=PlayerPos(0.0, 64.0, 0.0),
        block_events=(event,),
        inventory_delta=(("minecraft:oak_planks", -1),),
        dimension="overworld",
        biome="plains",
    )
    return ObservationPacket(tick=1, wallclock_ms=50, client=client, server=server)


def test_packet_constructs_with_all_fields():
    packet = _minimal_packet()
    assert packet.tick == 1
    assert packet.server.block_events[0].op is BlockOp.PLACE
    assert packet.client.held_item == "minecraft:oak_planks"


def test_block_op_values():
    assert {op.value for op in BlockOp} == {"place", "break"}


def test_packet_is_frozen():
    packet = _minimal_packet()
    with pytest.raises(dataclasses.FrozenInstanceError):
        packet.tick = 99
