"""The D5 action state machine: one gate read in, one B6 decision out.

This is D5 §3's pseudocode made executable, wrapping the D4 commit gate that
mica/gate/commit.py already computes. Order of authority inside one read:

  1. the single HARD veto — the human is close to the agent's target, close to
     the agent's own body (D5 §10 Q1's live half, wired 2026-07-12), or the
     target sits inside their workspace (the focus region) -> YIELD;
  2. the human is mid-action (not idle) -> OBSERVE;
  3. the safe window — the human is idle and not proximal:
       K_commit = 0:  SUGGEST if the discounted confidence clears theta_suggest,
                      else OBSERVE (uncertainty produces silence, not a guess)
       K_commit >= 1: PLACE_LOW_RISK only when the committed prefix is entirely
                      reversible AND the confidence bar is cleared AND the config
                      allows placement; otherwise PREVIEW. The bar (D5 §9
                      amendment 2026-07-12): conf >= theta_place, OR the human
                      DECLARED this session's build target — the declaration is
                      the human's own word, so it licenses authority the same way
                      it licenses gathering; it never shapes WHAT is proposed.

Reasoning mode is a soft feature, never a veto: conf = p* x (1 - P(z=1)) — a
heuristic-mode human carries no goal information, so confidence is discounted, not
blocked (D5 §3). This discounted form is the CANONICAL OBSERVE/SUGGEST semantics
by user decision 2026-07-06 (review P2-F1 resolved): D4's earlier raw-p* wording
is amended, and theta_suggest's single home is FsmConfig here — the staircase owns
K_commit only.

Hysteresis is asymmetric along the safety lattice (the corrected D5 §3 rule,
2026-07-06): a candidate at or below the current state takes effect on this read —
YIELD fires the moment the human is close, and a confidence dip drops authority the
read it happens — while a candidate above it must repeat, the SAME candidate, for
M consecutive reads. Authority is earned slowly and surrendered instantly, the same
asymmetry the D4 commit gate pins for K_commit. EXECUTE_CHUNK stays in the B6 enum
but no branch here can emit it while its config flag is False — and in v1 it
always is.
PLACE_LOW_RISK has its own flag: counterfactual runs may exercise it (nothing real
executes), the first live demo disables it (D5 §9 pin).

The proximity metric (D5 §10 question 1, pinned v1, 2026-07-05): a fixed Euclidean
radius around the human (PROXIMAL_RADIUS, matching the embodiment's 4-block
back-away band) plus a workspace radius around their focus block — the combination
answer, chosen because each half covers the other's blind spot (a human looking
away from where they stand; a target near their crosshair but far from their feet).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..contracts.b6 import GateDecision, GateState, InputsSnapshot

PROXIMAL_RADIUS = 4.0    # blocks, human body to agent target
WORKSPACE_RADIUS = 2.0   # blocks, human focus block to agent target

# The safety lattice, lowest (most defensive) first. Hysteresis compares ranks:
# moving down is free, moving up costs M consecutive identical reads. GATHER sits
# at PLACE_LOW_RISK's rank — both are autonomous world modification (the D5 §4
# amendment, 2026-07-10), so gathering costs the same slow-earned authority.
_LATTICE_RANK = {GateState.YIELD: 0, GateState.OBSERVE: 1, GateState.SUGGEST: 2,
                 GateState.PREVIEW: 3, GateState.PLACE_LOW_RISK: 4,
                 GateState.GATHER: 4, GateState.EXECUTE_CHUNK: 5}


class LatticeHysteresis:
    """D5 §3's ApplyHysteresis, corrected form (2026-07-06): downward moves are
    immediate; an upward move needs the SAME candidate on M consecutive reads —
    the pending variable is what keeps mixed-candidate flicker from ever
    accumulating into authority."""

    def __init__(self, m_consecutive: int, initial: GateState):
        self.m_consecutive = m_consecutive
        self.current = initial
        self.pending: GateState | None = None
        self.streak = 0

    def read(self, candidate: GateState) -> GateState:
        if _LATTICE_RANK[candidate] <= _LATTICE_RANK[self.current]:
            self.current = candidate
            self.pending, self.streak = None, 0
            return self.current
        if candidate is not self.pending:
            self.pending, self.streak = candidate, 0
        self.streak += 1
        if self.streak >= self.m_consecutive:
            self.current = candidate
            self.pending, self.streak = None, 0
        return self.current


@dataclass(frozen=True)
class FsmConfig:
    """The FSM's own knobs — frozen on validation, recorded in models/gate_v1.json."""

    theta_suggest: float
    theta_place: float
    m_consecutive: int
    proximal_radius: float = PROXIMAL_RADIUS
    workspace_radius: float = WORKSPACE_RADIUS
    place_low_risk_enabled: bool = True     # counterfactual: on; first live demo: OFF
    execute_chunk_enabled: bool = False     # v1: always off
    gather_enabled: bool = False            # D5 §4 amendment 2026-07-10: off by default,
                                            # so every banked behavior is bit-unchanged
    declared_place_enabled: bool = False    # D5 §9 amendment 2026-07-12: a declared
                                            # target may clear the place bar; armed only
                                            # by run_live --place, so banked behavior
                                            # is bit-unchanged everywhere else

    def __post_init__(self):
        if not self.theta_place > self.theta_suggest:
            raise ValueError("theta_place > theta_suggest by construction (D5 §9)")


@dataclass(frozen=True)
class GateRead:
    """Everything one gate read may look at. Positions are absolute world cells."""

    top_goal: str | None
    p_top: float
    entropy_nats: float
    p_z1: float
    idle: bool                                    # the human, this read
    player_pos: tuple[float, float, float] | None
    focus_block: tuple[int, int, int] | None
    target_cell: tuple[int, int, int] | None      # the first proposed action's cell
    k_commit: int
    prefix_fully_reversible: bool | None          # None when nothing is committed
    nothing_to_do: bool = False                   # the decoder's named "done" signal
    # The materials story (D7 §4/§6): the caller measured a shortfall for the target,
    # and whether that target was DECLARED by the human (a declaration authorizes
    # gathering outright; an inferred target must clear theta_place instead).
    gather_wanted: bool = False
    gather_declared: bool = False
    # The agent's own body (D5 §10 Q1 live half, 2026-07-12): the veto also fires
    # when the human stands close to the agent itself. None offline — bit-unchanged.
    agent_pos: tuple[float, float, float] | None = None
    # The human declared this session's build target (D5 §9 amendment 2026-07-12):
    # with declared_place_enabled, that word clears the placement confidence bar.
    place_declared: bool = False


def _distance(a, b) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))


def proximity_of(read: GateRead, config: FsmConfig) -> tuple[float | None, str | None]:
    """(logged proximity value, veto reason or None). The value is the human-to-
    target distance; the veto also fires when the target invades the workspace,
    or — when the caller supplies the agent's own position (live place mode) —
    when the human stands next to the agent itself."""
    distance = None
    if read.target_cell is not None and read.player_pos is not None:
        distance = _distance(read.player_pos, read.target_cell)
    # The agent-body half (D5 §10 Q1 live half, 2026-07-12): acting while the
    # human is crowded up against the agent is vetoed no matter where the target
    # is. Offline reads carry no agent_pos, so they are bit-unchanged.
    if read.agent_pos is not None and read.player_pos is not None:
        body = _distance(read.player_pos, read.agent_pos)
        if body <= config.proximal_radius:
            return distance, (f"human {body:.1f} blocks from the agent "
                              f"(radius {config.proximal_radius})")
    if read.target_cell is None:
        return None, None
    if distance is not None and distance <= config.proximal_radius:
        return distance, (f"human {distance:.1f} blocks from target "
                          f"(radius {config.proximal_radius})")
    if read.focus_block is not None:
        workspace = _distance(read.focus_block, read.target_cell)
        if workspace <= config.workspace_radius:
            return distance, (f"target {workspace:.1f} blocks from the human's focus "
                              f"(workspace {config.workspace_radius})")
    return distance, None


class GateFsm:
    """One session's gate: feed it reads in order, get B6 decisions out."""

    def __init__(self, config: FsmConfig):
        self.config = config
        self.hysteresis = LatticeHysteresis(config.m_consecutive, GateState.OBSERVE)

    def read(self, gate_read: GateRead) -> tuple[GateDecision, GateState]:
        """Returns (the decision, the raw candidate) — the trace logs both so a
        hysteresis hold is visible instead of looking like a slow gate."""
        config = self.config
        confidence = gate_read.p_top * (1.0 - gate_read.p_z1)
        proximity, veto = proximity_of(gate_read, config)

        if veto:
            candidate, why = GateState.YIELD, f"proximity veto: {veto}"
        elif not gate_read.idle:
            candidate, why = GateState.OBSERVE, "human active (macro-action in progress)"
        elif gate_read.nothing_to_do:
            candidate, why = GateState.OBSERVE, "decoder proposes nothing (build looks done)"
        elif (config.gather_enabled and gate_read.gather_wanted
              and (gate_read.gather_declared or confidence >= config.theta_place)):
            # The D5 §4 amendment (2026-07-10): fetch missing whitelisted materials.
            # Full autonomy removed the asking, not the thresholds — an inferred
            # target must clear the same bar placing does; a declared one is the
            # human's own word. Same lattice rank as placing, so hysteresis makes
            # gathering as slow to earn and as instant to surrender.
            candidate, why = GateState.GATHER, (
                "materials missing for the DECLARED target" if gate_read.gather_declared
                else f"materials missing; conf {confidence:.3f} >= theta_place "
                     f"{config.theta_place}")
        elif gate_read.k_commit == 0 or gate_read.target_cell is None:
            if confidence >= config.theta_suggest:
                candidate, why = GateState.SUGGEST, (
                    f"conf {confidence:.3f} >= theta_suggest {config.theta_suggest}")
            else:
                candidate, why = GateState.OBSERVE, (
                    f"conf {confidence:.3f} below theta_suggest {config.theta_suggest}")
        else:
            # The place bar (D5 §9 amendment 2026-07-12): confidence clears it, or
            # the human's own declared target does — the declaration is consent for
            # authority, never input to the proposal.
            by_confidence = confidence >= config.theta_place
            by_declaration = config.declared_place_enabled and gate_read.place_declared
            placeable = (gate_read.prefix_fully_reversible is True
                         and (by_confidence or by_declaration))
            if placeable and config.place_low_risk_enabled:
                candidate, why = GateState.PLACE_LOW_RISK, (
                    f"reversible prefix, conf {confidence:.3f} >= "
                    f"theta_place {config.theta_place}" if by_confidence else
                    f"reversible prefix, DECLARED target (conf {confidence:.3f})")
            elif placeable:
                candidate, why = GateState.PREVIEW, "placement disabled by config"
            elif gate_read.prefix_fully_reversible is not True:
                candidate, why = GateState.PREVIEW, "committed prefix not fully reversible"
            else:
                candidate, why = GateState.PREVIEW, (
                    f"conf {confidence:.3f} below theta_place {config.theta_place}")

        state = self.hysteresis.read(candidate)
        if state != candidate:
            why += f" | held by hysteresis (candidate {candidate.value})"
        snapshot = InputsSnapshot(
            top_goal=gate_read.top_goal, p_top_goal=gate_read.p_top,
            belief_entropy=gate_read.entropy_nats, p_z1=gate_read.p_z1,
            proximity=proximity, reversibility=gate_read.prefix_fully_reversible)
        return GateDecision(state=state, inputs_snapshot=snapshot, reason=why), candidate
