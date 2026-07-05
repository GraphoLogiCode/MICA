import json

from mica.capture.jsonl_ingest import JsonlSource
from mica.contracts.b0 import BlockOp
from mica.validation.sync_report import build_sync_report


def _moment(tick, events):
    # A server-only capture's shape: client fields null. (No such recorder exists anymore —
    # the old bot is retired — but ingest must still tolerate the shape, and coverage
    # must flag it rather than crash.)
    return {
        "tick": tick,
        "wallclock_ms": tick * 50,
        "client": {
            "capture_wallclock_ms": tick * 50,
            "pov_frame": None,
            "input_state": None,
            "yaw": 0.0,
            "pitch": 0.0,
            "crosshair_target": None,
            "held_item": None,
            "hotbar": None,
            "gui_open": None,
        },
        "server": {
            "player_pos": [0, 64, 0],
            "block_events": events,
            "inventory_delta": [],
            "dimension": "minecraft:overworld",
            "biome": "unknown",
        },
    }


def _source(tmp_path, moments, declared):
    jsonl = tmp_path / "s.jsonl"
    manifest = tmp_path / "s.manifest.json"
    jsonl.write_text("\n".join(json.dumps(m) for m in moments), encoding="utf-8")
    manifest.write_text(
        json.dumps({"session_id": "t", "session_start_ms": 0, "declared_event_count": declared}),
        encoding="utf-8",
    )
    return JsonlSource(jsonl_path=str(jsonl), manifest_path=str(manifest))


def test_ingest_parses_block_events_and_keeps_unseen_client_fields_none(tmp_path):
    event = {"event_id": 0, "pos": [1, 64, 0], "block_type": "minecraft:oak_planks", "op": "place", "actor": "Steve"}
    session = _source(tmp_path, [_moment(0, []), _moment(1, [event])], declared=1).load()
    assert len(session.packets) == 2
    assert session.packets[0].client.input_state is None
    parsed = session.packets[1].server.block_events[0]
    assert parsed.op is BlockOp.PLACE
    assert parsed.block_type == "minecraft:oak_planks"


def test_ingested_server_capture_passes_structure_but_flags_client_half(tmp_path):
    event = {"event_id": 0, "pos": [1, 64, 0], "block_type": "minecraft:oak_planks", "op": "place", "actor": "Steve"}
    report = build_sync_report(_source(tmp_path, [_moment(0, []), _moment(1, [event])], declared=1).load())
    assert report.block_events.complete
    readers = {result.reader: result for result in report.coverage}
    assert readers["D2 3D structure stream"].satisfied
    assert not readers["D1 2D symbolic half"].satisfied


def test_provisional_manifest_loads_but_is_marked(tmp_path):
    # A crash leaves declared_event_count = -1. The recording must still load (for
    # inspection), fall back to the observed count, and carry the provisional mark.
    event = {"event_id": 0, "pos": [1, 64, 0], "block_type": "minecraft:oak_planks", "op": "place", "actor": "Steve"}
    session = _source(tmp_path, [_moment(0, []), _moment(1, [event])], declared=-1).load()
    assert session.declared_is_provisional
    assert session.declared_event_count == 1

    finalized = _source(tmp_path, [_moment(0, []), _moment(1, [event])], declared=1).load()
    assert not finalized.declared_is_provisional


def test_manifest_frame_settings_are_read(tmp_path):
    jsonl = tmp_path / "s.jsonl"
    manifest = tmp_path / "s.manifest.json"
    jsonl.write_text(json.dumps(_moment(0, [])), encoding="utf-8")
    manifest.write_text(
        json.dumps({"session_id": "t", "session_start_ms": 0, "declared_event_count": 0,
                    "frame_every": 1, "frame_width_px": 320}),
        encoding="utf-8",
    )
    session = JsonlSource(jsonl_path=str(jsonl), manifest_path=str(manifest)).load()
    assert session.manifest.frame_every == 1
    assert session.manifest.frame_width_px == 320
