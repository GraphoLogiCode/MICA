"""run_d1 must refuse a session that fails the B0 gate — evidence built from an
unverified capture would poison the corpus quietly. Runs the real script end to end."""
import json
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_ROOT, "scripts", "run_d1.py")


def _moment(tick, events):
    # A server-only capture: client fields null, so the B0 gate fails on coverage + frames.
    return {
        "tick": tick,
        "wallclock_ms": tick * 50,
        "client": {
            "capture_wallclock_ms": tick * 50, "pov_frame": None, "input_state": None,
            "yaw": 0.0, "pitch": 0.0, "crosshair_target": None,
            "held_item": None, "hotbar": None, "gui_open": None,
        },
        "server": {
            "player_pos": [0, 64, 0], "block_events": events, "inventory_delta": [],
            "dimension": "minecraft:overworld", "biome": "unknown",
        },
    }


def _write_session(tmp_path):
    event = {"event_id": 0, "pos": [1, 64, 0], "block_type": "minecraft:oak_planks",
             "op": "place", "actor": "Steve"}
    moments = [_moment(t, [event] if t == 2 else []) for t in range(6)]
    jsonl = tmp_path / "s.jsonl"
    jsonl.write_text("\n".join(json.dumps(m) for m in moments), encoding="utf-8")
    (tmp_path / "s.manifest.json").write_text(
        json.dumps({"session_id": "t", "session_start_ms": 0, "declared_event_count": 1}),
        encoding="utf-8",
    )
    return str(jsonl)


def _run(*args):
    return subprocess.run([sys.executable, _SCRIPT, *args],
                          capture_output=True, text=True, cwd=_ROOT)


def test_gate_failing_session_is_refused(tmp_path):
    result = _run(_write_session(tmp_path))
    assert result.returncode == 1
    assert "REFUSED" in result.stdout and "B0 gate" in result.stdout


def test_allow_ungated_overrides_for_debugging(tmp_path):
    result = _run(_write_session(tmp_path), "--allow-ungated")
    assert result.returncode == 0
    assert "build-action fidelity" in result.stdout
