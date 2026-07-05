"""The agent-eye scan channel (h3d_scan), mind side: parsing the body's scan file,
first-seen accumulation, the A7-clean channel selection (scanned ∩ built), coverage,
and the per-session report script end to end."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import run_agent_scan  # noqa: E402

from mica.capture.sample_builds import assisted_build
from mica.capture.synthetic import generate_session
from mica.contracts.serialize import packet_to_dict
from mica.perception.agent_scan import accumulate, coverage, load_scan, scanned_build_cells

_SID = "scantest-0001"
_REGION = (-8, 56, -8, 12, 72, 8)
_PLANKS = "minecraft:oak_planks"


def _write_scan(path, sweeps, torn_tail=False):
    lines = [json.dumps({"ts": 1000 + i, "tick": 10 * i, "pose": [0, 64, 5, 0, 0],
                         "cells": cells}) for i, cells in enumerate(sweeps)]
    text = "\n".join(lines) + "\n"
    if torn_tail:
        text += '{"ts": 9999, "tick": 99, "po'    # the agent caught mid-write
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_load_scan_tolerates_a_torn_tail_and_qualifies_names(tmp_path):
    path = _write_scan(tmp_path / "s.jsonl",
                       [[[0, 64, 0, "oak_planks"]], [[1, 64, 0, "minecraft:stone"]]],
                       torn_tail=True)
    sweeps = load_scan(path)
    assert len(sweeps) == 2                       # the torn line ends the read, no crash
    assert sweeps[0].cells[0][3] == "minecraft:oak_planks"   # bare names get the namespace
    assert sweeps[1].cells[0][3] == "minecraft:stone"


def test_accumulate_first_seen_wins(tmp_path):
    path = _write_scan(tmp_path / "s.jsonl",
                       [[[0, 64, 0, "oak_planks"]], [[0, 64, 0, "stone"], [1, 64, 0, "glass"]]])
    seen = accumulate(load_scan(path))
    assert seen[(0, 64, 0)] == "minecraft:oak_planks"        # the first sighting stands
    assert seen[(1, 64, 0)] == "minecraft:glass"


def test_channel_cloud_is_scanned_built_only():
    # the sensor saw terrain, an agent block, and two human blocks; the channel
    # (A7 rule) may carry only the scanned part of the BUILT set
    scanned = {(0, 60, 0): "minecraft:stone",         # terrain
               (0, 64, 3): "minecraft:stone",         # the agent's own block
               (0, 64, 0): _PLANKS, (1, 64, 0): _PLANKS}
    built = {(0, 64, 0): _PLANKS, (1, 64, 0): _PLANKS, (2, 64, 0): _PLANKS}
    channel = scanned_build_cells(scanned, built)
    assert channel == {(0, 64, 0): _PLANKS, (1, 64, 0): _PLANKS}
    assert coverage(scanned, built) == (2, 3)


def _lay_out_session(raw_dir):
    """A minimal D2-ready capture dir: jsonl + manifest + empty base snapshot +
    a scan file where the agent saw 3 of the human's 5 wall blocks, its own
    block, and terrain."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = generate_session(assisted_build(session_id=_SID))
    jsonl = raw_dir / f"{_SID}.jsonl"
    dicts = [packet_to_dict(p) for p in sorted(session.packets, key=lambda p: p.tick)]
    jsonl.write_text("".join(json.dumps(d) + "\n" for d in dicts), encoding="utf-8")
    (raw_dir / f"{_SID}.manifest.json").write_text(json.dumps({
        "session_id": _SID, "session_start_ms": 0, "mc_version": "1.16.5",
        "mod_version": "fabric-b0-0.0.5", "event_schema_version": "1",
        "frame_every": 1, "frame_width_px": 4, "snapshot_quiet_ticks": 40,
        "snapshot_region": list(_REGION),
        "declared_event_count": session.declared_event_count}), encoding="utf-8")
    snap_dir = raw_dir / _SID / "snapshots"
    snap_dir.mkdir(parents=True)
    volume = ((_REGION[3] - _REGION[0] + 1) * (_REGION[4] - _REGION[1] + 1)
              * (_REGION[5] - _REGION[2] + 1))
    (snap_dir / "0.json").write_text(json.dumps(
        {"tick": 0, "region": list(_REGION), "palette": ["minecraft:air"],
         "runs": [[volume, 0]]}), encoding="utf-8")
    _write_scan(raw_dir / "agent-MICA_AI.scan.jsonl",
                [[[0, 64, 0, "oak_planks"], [0, 60, 0, "stone"]],
                 [[1, 64, 0, "oak_planks"], [2, 64, 0, "oak_planks"], [0, 64, 3, "stone"]]])
    return jsonl


def test_report_script_end_to_end(tmp_path, monkeypatch):
    jsonl = _lay_out_session(tmp_path)

    monkeypatch.setattr(sys, "argv", ["run_agent_scan.py", str(jsonl)])
    assert run_agent_scan.main() == 0
    report = json.loads((tmp_path / f"{_SID}.agent_scan_report.json").read_text(encoding="utf-8"))
    assert report["built_total"] == 5
    assert report["built_seen"] == 3
    assert report["coverage"] == 0.6
    assert report["scanned_cells"] == 5                       # terrain + agent cells counted raw
    assert len(report["coverage_curve"]) == 2                 # one point per sweep
    assert report["coverage_curve"][0]["built_seen"] == 1     # first sweep saw one wall block


def test_session_report_renders_from_partial_artifacts(tmp_path, monkeypatch):
    # only the raw capture + scan exist (no belief/evidence logs yet): every
    # missing log must skip its graph, and the report still lands with the
    # clouds + summary — the tool may never refuse a session for being partial
    import make_session_report

    jsonl = _lay_out_session(tmp_path / "raw")
    monkeypatch.setattr(sys, "argv", ["make_session_report.py", str(jsonl)])
    assert make_session_report.main() == 0
    out_dir = tmp_path / "reports" / _SID
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["built_cells"] == 5
    assert summary["scanned_of_built"] == 3
    assert "clouds.png" in summary["graphs"]                 # drawn from raw + scan alone
    assert (out_dir / "clouds.png").exists()
    assert (out_dir / "agent_scan.jsonl").exists()           # the scan copy banks with it
