"""The materials constraint (D5 §4, 2026-07-06): stock depletes along the prefix,
substitution needs an explicit grant, the gate caps its prefix and reports, and
with no inventory snapshot the whole constraint is inactive."""
import json

import pytest

from mica.contracts.b5 import ProposalChunk
from mica.decoder.grammar import Move, Place
from mica.gate import materials
from mica.gate.live_loop import LiveGateRunner, gate_ready
from mica.intent.tracker import uniform_belief

from decoder_fixtures import fused_record


def _places(count: int, block: str = "minecraft:oak_planks"):
    return tuple(Place(dx=i, dy=0, dz=0, block=block) for i in range(count))


def test_stock_depletes_along_the_prefix():
    flags, missing, substituted = materials.feasibility_flags(
        _places(3), {"oak_planks": 2}, {})
    assert flags == [False, False, True]      # third placement has nothing to draw on
    assert missing == {"oak_planks": 1}
    assert substituted == {}


def test_moves_and_says_consume_nothing():
    actions = (Move(dx=0, dy=0, dz=0),) + _places(1)
    flags, missing, _ = materials.feasibility_flags(actions, {"oak_planks": 1}, {})
    assert flags == [False, False] and missing == {}


def test_a_grant_lets_the_substitute_stand_in():
    flags, missing, substituted = materials.feasibility_flags(
        _places(2), {"spruce_planks": 5}, {"oak_planks": "spruce_planks"})
    assert flags == [False, False]
    assert missing == {} and substituted == {"oak_planks": 2}


def test_namespace_normalization_matches_bot_inventory_names():
    flags, missing, _ = materials.feasibility_flags(
        _places(1), {"minecraft:oak_planks": 1}, {})
    assert flags == [False] and missing == {}


def test_substitute_proposals_stay_in_family():
    inventory = {"spruce_planks": 3, "cobblestone": 64, "oak_log": 2}
    assert materials.propose_substitute("oak_planks", inventory, {}) == "spruce_planks"
    assert materials.propose_substitute("oak_planks", inventory, {},
                                        denied=("spruce_planks",)) is None
    assert materials.propose_substitute("stone_bricks", inventory, {}) is None


def test_account_reports_shortfall_not_read_sums():
    account = materials.MaterialsAccount()
    for _ in range(60):   # the same shortage re-read at 1 Hz is ONE shortage
        account.observe(_places(3), {"oak_planks": 1}, {}, capped=True,
                        inventory={"oak_planks": 2}, grants={})
    report = account.report(committed_places=0)
    assert report["missing"] == {"oak_planks": 1}
    assert report["compromises"]["reads_with_shortened_prefix"] == 60
    assert report["inventory_remaining"] == {"oak_planks": 2}


def test_snapshot_reader_takes_the_newest_fresh_line(tmp_path):
    status = tmp_path / "agent-MICA_AI.status.jsonl"
    status.write_text(
        json.dumps({"ts": 1, "inventory": {"dirt": 1}, "material_grants": {}}) + "\n"
        + json.dumps({"ts": 2, "inventory": {"oak_planks": 7},
                      "material_grants": {"oak_planks": "spruce_planks"},
                      "pos": [10.5, 64.0, -3.5]}) + "\n",
        encoding="utf-8")
    snapshot = materials.read_agent_snapshot(str(tmp_path))
    # Third element: the agent's own position (D5 §10 Q1 live half); fourth: the
    # relayed placement consent (D5 §10, 2026-07-19) — both ride the same file
    # read that serves the materials constraint.
    assert snapshot == ({"oak_planks": 7}, {"oak_planks": "spruce_planks"},
                        (10.5, 64.0, -3.5), None)
    assert materials.read_agent_snapshot(str(tmp_path / "empty")) is None


def test_snapshot_reader_without_pos_returns_none_pos(tmp_path):
    status = tmp_path / "agent-MICA_AI.status.jsonl"
    status.write_text(
        json.dumps({"ts": 1, "inventory": {"dirt": 1}, "material_grants": {}}) + "\n",
        encoding="utf-8")
    assert materials.read_agent_snapshot(str(tmp_path)) == ({"dirt": 1}, {}, None, None)


@pytest.fixture()
def runner(tmp_path):
    if not gate_ready():
        pytest.skip("no decoder/gate freeze on this machine")
    from mica.intent import heads_v1

    gate = LiveGateRunner(str(tmp_path / "trace.jsonl"), heads_v1.tracker_params(),
                          demo=True)
    yield gate
    gate.close()


def _read_with(runner, monkeypatch, inventory, grants):
    """One gate read against a known proposal and a known inventory snapshot."""
    from mica.gate import live_loop

    proposal = ProposalChunk(actions=_places(2) + (Move(dx=1, dy=0, dz=1),),
                             token_conf=(1.0, 0.9, 0.9), rationale_goal="habitation")
    monkeypatch.setattr(live_loop.decoder_model, "propose",
                        lambda model, ctx, n: (proposal, None))
    runner.model = object()                    # already "loaded": skip the real torch load
    runner.origin = (0, 64, 0)                 # anchored: the decode branch runs
    runner.materials_source = lambda: (inventory, grants) if inventory is not None else None
    return runner.read(uniform_belief(), fused_record(),
                       {"belief_snapshot_id": 100, "player_pos": (50.0, 64.0, 50.0)})


def test_gate_caps_and_reports_then_a_grant_flips_it(runner, monkeypatch, tmp_path):
    short = _read_with(runner, monkeypatch, {"spruce_planks": 10}, {})
    assert short["materials"]["feasible_prefix"] == 0      # first placement infeasible
    assert short["materials"]["missing"] == {"oak_planks": 2}
    assert short["materials"]["ask"] == {"block": "oak_planks", "short": 2,
                                         "substitute": "spruce_planks"}

    granted = _read_with(runner, monkeypatch, {"spruce_planks": 10},
                         {"oak_planks": "spruce_planks"})
    assert granted["materials"]["missing"] == {}
    assert granted["materials"]["feasible_prefix"] == 3    # the whole chunk feasible
    assert granted["materials"]["ask"] is None

    runner.trace.flush()
    rows = [json.loads(line) for line in
            (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-2]["feasible_prefix"] == 0 and rows[-2]["missing"] == {"oak_planks": 2}
    assert rows[-1]["feasible_prefix"] == 3

    runner.close()
    report = json.loads((tmp_path / "trace.materials_report.json").read_text(encoding="utf-8"))
    assert report["proposed_blocks"] == {"oak_planks": 4}   # two reads, two places each
    assert report["missing"] == {"oak_planks": 2}
    assert report["compromises"]["substitution_grants_used"] == {"oak_planks": 2}
    assert report["committed_places"] == 0                  # demo config commits nothing


def test_no_snapshot_means_the_constraint_is_inactive(runner, monkeypatch):
    block = _read_with(runner, monkeypatch, None, {})
    assert block["materials"] is None
    assert runner.account.last_inventory is None            # close() writes no report
