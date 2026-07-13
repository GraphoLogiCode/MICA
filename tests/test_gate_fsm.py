"""The D5 FSM: veto precedence, the safety lattice, the mode discount, hysteresis,
and the guarantee that EXECUTE_CHUNK cannot be emitted in v1."""
import itertools

import pytest

from mica.contracts.b6 import GateState
from mica.gate.fsm import FsmConfig, GateFsm, GateRead

_CONFIG = FsmConfig(theta_suggest=0.10, theta_place=0.20, m_consecutive=2)


def _read(**overrides) -> GateRead:
    body = dict(top_goal="defense", p_top=0.6, entropy_nats=1.0, p_z1=0.2,
                idle=True, player_pos=(20.0, 64.0, 20.0), focus_block=None,
                target_cell=(0, 64, 0), k_commit=2, prefix_fully_reversible=True,
                nothing_to_do=False)
    body.update(overrides)
    return GateRead(**body)


def _candidate(read, config=_CONFIG) -> GateState:
    _, candidate = GateFsm(config).read(read)
    return candidate


def test_theta_place_must_exceed_theta_suggest():
    with pytest.raises(ValueError):
        FsmConfig(theta_suggest=0.2, theta_place=0.2, m_consecutive=2)


def test_proximity_veto_beats_everything():
    # everything says place — but the human stands next to the target
    assert _candidate(_read(player_pos=(1.0, 64.0, 1.0))) is GateState.YIELD


def test_workspace_veto_fires_on_focus_alone():
    read = _read(player_pos=(30.0, 64.0, 30.0), focus_block=(1, 64, 0))
    assert _candidate(read) is GateState.YIELD


def test_active_human_means_observe():
    assert _candidate(_read(idle=False)) is GateState.OBSERVE


def test_safe_window_k_zero_splits_on_theta_suggest():
    assert _candidate(_read(k_commit=0, target_cell=None)) is GateState.SUGGEST
    low = _read(k_commit=0, target_cell=None, p_top=0.1)
    assert _candidate(low) is GateState.OBSERVE


def test_mode_discount_is_soft_but_real():
    # a fully heuristic read zeroes the confidence: silence, not a veto
    assert _candidate(_read(k_commit=0, target_cell=None,
                            p_top=0.9, p_z1=1.0)) is GateState.OBSERVE


def test_place_needs_reversible_prefix_and_confidence_and_config():
    assert _candidate(_read()) is GateState.PLACE_LOW_RISK
    assert _candidate(_read(prefix_fully_reversible=False)) is GateState.PREVIEW
    assert _candidate(_read(p_top=0.2)) is GateState.PREVIEW   # conf 0.16 < 0.20
    disabled = FsmConfig(theta_suggest=0.10, theta_place=0.20, m_consecutive=2,
                         place_low_risk_enabled=False)
    decision, candidate = GateFsm(disabled).read(_read())
    assert candidate is GateState.PREVIEW
    assert "disabled by config" in decision.reason


def test_nothing_to_do_reads_as_observe_with_its_own_reason():
    fsm = GateFsm(_CONFIG)
    decision, candidate = fsm.read(_read(k_commit=0, target_cell=None,
                                         nothing_to_do=True))
    assert candidate is GateState.OBSERVE
    assert "proposes nothing" in decision.reason


def test_hysteresis_holds_an_upward_candidate_for_m_reads():
    fsm = GateFsm(_CONFIG)
    first, candidate = fsm.read(_read())
    assert candidate is GateState.PLACE_LOW_RISK
    assert first.state is GateState.OBSERVE          # held: streak 1 of 2
    assert "held by hysteresis" in first.reason
    second, _ = fsm.read(_read())
    assert second.state is GateState.PLACE_LOW_RISK  # second consecutive read commits


def test_downward_moves_are_immediate():
    fsm = GateFsm(_CONFIG)
    fsm.read(_read())
    fsm.read(_read())                                # earn PLACE_LOW_RISK (M = 2)
    # the human steps close: YIELD on THIS read, no hysteresis hold (D5 §4)
    vetoed, _ = fsm.read(_read(player_pos=(1.0, 64.0, 1.0)))
    assert vetoed.state is GateState.YIELD
    assert "held by hysteresis" not in vetoed.reason


def test_confidence_dip_drops_authority_the_read_it_happens():
    fsm = GateFsm(_CONFIG)
    fsm.read(_read())
    fsm.read(_read())                                # earn PLACE_LOW_RISK
    dipped, _ = fsm.read(_read(p_top=0.2))           # conf 0.16 < theta_place 0.20
    assert dipped.state is GateState.PREVIEW         # not held at PLACE_LOW_RISK


def test_mixed_upward_flicker_never_accumulates_into_authority():
    fsm = GateFsm(_CONFIG)
    for _ in range(4):                               # SUGGEST/PREVIEW alternating
        held_suggest, c1 = fsm.read(_read(k_commit=0, target_cell=None))
        held_preview, c2 = fsm.read(_read(prefix_fully_reversible=False))
        assert c1 is GateState.SUGGEST and c2 is GateState.PREVIEW
        assert held_suggest.state is GateState.OBSERVE
        assert held_preview.state is GateState.OBSERVE


_PLACE_CONFIG = FsmConfig(theta_suggest=0.10, theta_place=0.20, m_consecutive=2,
                          declared_place_enabled=True)


def test_declared_target_clears_the_place_bar_when_armed():
    # conf 0.16 < theta_place 0.20 — confidence alone says PREVIEW...
    low_conf = _read(p_top=0.2, place_declared=True)
    decision, candidate = GateFsm(_PLACE_CONFIG).read(low_conf)
    assert candidate is GateState.PLACE_LOW_RISK      # ...the declaration licenses it
    assert "DECLARED target" in decision.reason


def test_declared_route_is_dead_unless_config_armed():
    # same read, default config: --place never ran, so the declaration is inert
    low_conf = _read(p_top=0.2, place_declared=True)
    assert _candidate(low_conf) is GateState.PREVIEW


def test_declared_route_needs_the_actual_declaration():
    assert _candidate(_read(p_top=0.2, place_declared=False),
                      _PLACE_CONFIG) is GateState.PREVIEW


def test_declared_route_never_overrides_reversibility_or_vetoes():
    declared = dict(p_top=0.2, place_declared=True)
    assert _candidate(_read(prefix_fully_reversible=False, **declared),
                      _PLACE_CONFIG) is GateState.PREVIEW
    assert _candidate(_read(player_pos=(1.0, 64.0, 1.0), **declared),
                      _PLACE_CONFIG) is GateState.YIELD
    assert _candidate(_read(idle=False, **declared),
                      _PLACE_CONFIG) is GateState.OBSERVE


def test_declared_route_still_pays_the_hysteresis_toll():
    fsm = GateFsm(_PLACE_CONFIG)
    held, candidate = fsm.read(_read(p_top=0.2, place_declared=True))
    assert candidate is GateState.PLACE_LOW_RISK
    assert held.state is GateState.OBSERVE            # streak 1 of 2: held
    second, _ = fsm.read(_read(p_top=0.2, place_declared=True))
    assert second.state is GateState.PLACE_LOW_RISK


def test_human_next_to_the_agent_body_vetoes():
    # target is far from the human, but the human walked up to the agent itself
    # (D5 §10 Q1 live half): YIELD, immediately.
    crowded = _read(player_pos=(20.0, 64.0, 20.0), agent_pos=(21.0, 64.0, 20.0))
    decision, candidate = GateFsm(_CONFIG).read(crowded)
    assert candidate is GateState.YIELD
    assert "from the agent" in decision.reason


def test_agent_pos_none_keeps_offline_reads_unchanged():
    assert _candidate(_read(agent_pos=None)) is GateState.PLACE_LOW_RISK


def test_execute_chunk_is_unreachable_in_v1():
    combos = itertools.product((True, False), (0, 3), (True, False, None),
                               (0.05, 0.9), (0.0, 1.0))
    fsm = GateFsm(_CONFIG)
    for idle, k, reversible, p_top, p_z1 in combos:
        _, candidate = fsm.read(_read(idle=idle, k_commit=k,
                                      prefix_fully_reversible=reversible,
                                      p_top=p_top, p_z1=p_z1))
        assert candidate is not GateState.EXECUTE_CHUNK


def test_snapshot_carries_the_six_pinned_features():
    decision, _ = GateFsm(_CONFIG).read(_read())
    snapshot = decision.inputs_snapshot
    assert snapshot.top_goal == "defense"
    assert snapshot.proximity is not None and snapshot.proximity > 4.0
    assert snapshot.reversibility is True
