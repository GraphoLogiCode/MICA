"""The B4/B5/B6 contracts hold their own rules, and contract C1 holds as an
import-graph fact: no filter-side module may read the decoder's outputs."""
import os

import pytest

from mica.contracts.b4 import ControlContext, SLOT_KINDS
from mica.contracts.b5 import ProposalChunk
from mica.contracts.b6 import GateDecision, GateState, InputsSnapshot
from mica.contracts.goals import GOALS
from mica.decoder.grammar import Move, Place

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _b4(**overrides):
    body = dict(tick=100, k=3, belief_snapshot_id=100, slot_kind="arm3",
                goal_marginal=(0.6, 0.1, 0.1, 0.1, 0.1), top_goal=GOALS[0],
                p_top=0.6, entropy_nats=1.2, p_z1=0.5, evidence=(0.0, 1.0))
    body.update(overrides)
    return ControlContext(**body)


def test_b4_accepts_a_well_formed_record():
    assert _b4().slot_kind == "arm3"


def test_b4_rejects_unknown_slot_kind():
    with pytest.raises(ValueError):
        _b4(slot_kind="arm7")


def test_b4_rejects_marginal_that_does_not_sum_to_one():
    with pytest.raises(ValueError):
        _b4(goal_marginal=(0.6, 0.1, 0.1, 0.1, 0.3))


def test_b4_arm0_must_be_all_zero_and_goal_less():
    ok = _b4(slot_kind="arm0_zero", goal_marginal=(0.0,) * len(GOALS),
             top_goal=None, p_top=0.0, entropy_nats=0.0)
    assert ok.top_goal is None
    with pytest.raises(ValueError):
        _b4(slot_kind="arm0_zero")   # carries a real marginal


def test_b4_rejects_non_finite_evidence():
    with pytest.raises(ValueError):
        _b4(evidence=(0.0, float("nan")))


def test_b5_conf_one_convention_is_enforced():
    with pytest.raises(ValueError):
        ProposalChunk(actions=(Move(0, 0, 0),), token_conf=(0.9,), rationale_goal=None)
    ok = ProposalChunk(actions=(Move(0, 0, 0), Place(1, 0, 0, "minecraft:oak_planks")),
                       token_conf=(1.0, 0.7), rationale_goal="defense")
    assert ok.token_conf[0] == 1.0


def test_b5_rejects_mismatched_lengths_and_bad_ranges():
    with pytest.raises(ValueError):
        ProposalChunk(actions=(Move(0, 0, 0),), token_conf=(1.0, 0.5), rationale_goal=None)
    with pytest.raises(ValueError):
        ProposalChunk(actions=(Move(0, 0, 0), Move(1, 0, 0)),
                      token_conf=(1.0, 1.5), rationale_goal=None)
    with pytest.raises(ValueError):
        ProposalChunk(actions=(), token_conf=(), rationale_goal=None)


def test_b6_schema_carries_the_pinned_states_and_features():
    states = {s.value for s in GateState}
    assert states == {"observe", "suggest", "preview", "place_low_risk",
                      "execute_chunk", "yield"}
    decision = GateDecision(
        state=GateState.OBSERVE,
        inputs_snapshot=InputsSnapshot(top_goal="defense", p_top_goal=0.4,
                                       belief_entropy=1.5, p_z1=0.5,
                                       proximity=None, reversibility=None),
        reason="below theta_suggest")
    assert decision.reason


def test_c1_no_filter_side_module_imports_the_decoder():
    """Contract C1 as an import-graph fact: the tracker, the likelihood heads, the
    fusion layer, and the perception streams must have NO import path to the
    decoder package or the B5 proposal type — decoder confidences can then never
    become likelihoods without this test failing first."""
    filter_side = []
    for package in ("intent", "perception", "data"):
        directory = os.path.join(_ROOT, "mica", package)
        filter_side.extend(os.path.join(directory, name)
                           for name in os.listdir(directory) if name.endswith(".py"))
    filter_side.append(os.path.join(_ROOT, "mica", "live_pipeline.py"))
    allowed = {"decoder_corpus.py"}   # corpus building CONSUMES the belief; one-way
    for path in filter_side:
        if os.path.basename(path) in allowed:
            continue
        source = open(path, encoding="utf-8").read()
        for line in source.splitlines():
            line = line.strip()
            if line.startswith(("import", "from")):
                assert "decoder" not in line, f"{path}: {line}"
                assert "b5" not in line, f"{path}: {line}"
