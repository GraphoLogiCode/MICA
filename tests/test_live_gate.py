"""The D5 live gate runner: demo config holds, degradation is a state, the human
cell tracking is actor-honest, and the Arm-0 neutral derivation does what its
docstring says."""
import json

import pytest

from decoder_fixtures import fused_record
from mica.capture.synthetic import ScriptedPlacement  # noqa: F401 (fixture parity)
from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos
from mica.gate import commit
from mica.gate.live_loop import LiveGateRunner, gate_ready
from mica.intent.tracker import uniform_belief


@pytest.fixture()
def runner(tmp_path):
    if not gate_ready():
        pytest.skip("no decoder/gate freeze on this machine")
    from mica.intent import heads_v1

    gate = LiveGateRunner(str(tmp_path / "trace.jsonl"), heads_v1.tracker_params(),
                          demo=True)
    yield gate
    gate.close()


def test_demo_config_disables_placement(runner):
    assert runner.fsm.config.place_low_risk_enabled is False
    assert runner.fsm.config.execute_chunk_enabled is False


def test_no_correction_yet_reads_as_observe(runner):
    block = runner.read(uniform_belief(), None, {})
    assert block["state"] == "observe" and "no corrections" in block["reason"]


def test_degraded_read_still_writes_a_trace_row(runner, tmp_path):
    runner.read(uniform_belief(), None, {"tick": 5})
    rows = [json.loads(line) for line in
            (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["reason"] == "no corrections yet"
    assert rows[-1]["chosen_state"] == "observe"
    assert rows[-1]["committed_actions"] == []


def test_between_correction_reads_use_a_time_advanced_belief(runner):
    # The belief only moves at corrections; a read a minute later must gate on
    # what the kernels say is true NOW, not on the frozen correction-time mass.
    belief = uniform_belief()
    peak = next(iter(belief))
    rest = 0.1 / (len(belief) - 1)
    peaked = {key: (0.9 if key == peak else rest) for key in belief}
    at_correction = runner._materialized(peaked, {"tick": 100, "belief_snapshot_id": 100})
    assert at_correction == peaked            # same tick: bit-identical, replay-safe
    minute_later = runner._materialized(
        peaked, {"tick": 100 + 20 * 60, "belief_snapshot_id": 100})
    assert minute_later[peak] < peaked[peak]  # the peak decayed toward uniform
    assert abs(sum(minute_later.values()) - 1.0) < 1e-9
    no_live_tick = runner._materialized(peaked, {"belief_snapshot_id": 100})
    assert no_live_tick == peaked             # replay drivers carry no live tick


def test_read_produces_a_trace_row_and_status_block(runner, tmp_path):
    status = {"current_behavior": "idle", "player_pos": (30.0, 64.0, 30.0),
              "focus_block": None, "belief_snapshot_id": 100}
    block = runner.read(uniform_belief(), fused_record(), status)
    assert block["state"] in ("observe", "suggest", "preview", "yield")
    assert block["state"] != "place_low_risk"
    runner.trace.flush()
    assert (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip()


def test_human_cell_tracking_ignores_the_agent(runner):
    runner.ingest_events([
        BlockEvent(event_id=0, pos=BlockPos(1, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.PLACE, actor="HumanBuilder"),
        BlockEvent(event_id=1, pos=BlockPos(2, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.PLACE, actor="MICA_AI"),
        BlockEvent(event_id=2, pos=BlockPos(1, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.BREAK, actor="HumanBuilder"),
    ])
    assert runner.human_cells == set()      # placed then broken; agent's never entered


def test_arm0_neutral_derivation_sides():
    assert commit.derive_arm0_neutral(0.7, theta_1=0.5) == 0.52
    assert commit.derive_arm0_neutral(0.1, theta_1=0.5) == 0.2
    assert commit.legacy_gate_mean([0.6, 0.6, 0.1], [0.2, 0.9, 0.2], 0.5) == pytest.approx(1 / 3)
    commit.set_arm0_neutral(0.52)
    try:
        assert commit.arm_p_star("arm0_zero", None, 0.9, 1,
                                 __import__("mica.intent.tracker",
                                            fromlist=["TrackerParams"]).TrackerParams(),
                                 1.0) == 0.52
    finally:
        commit.set_arm0_neutral(None)
