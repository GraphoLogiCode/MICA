"""The D4 commit gate: how many proposed actions the belief currently licenses.

The canonical staircase (D4, verifier-mandated form — the earlier whole-chunk
two-threshold and entropy-multiplicative gates are superseded):

  K_commit = max { i <= K : for all j <= i,
                   token_conf_j >= c_min  AND
                   p*(b_{k+j|k}) >= theta(j, irrev(a_j)) }

where b_{k+j|k} is the KERNEL-PROPAGATED belief — one tracker.predict call over
j * delta_hat seconds, exact Chapman-Kolmogorov, no ad hoc horizon discount — and
theta rises with position and with per-action irreversibility.

delta_hat is the expected inter-correction gap, a session statistic frozen on
validation data (the horizon map Delta t = j * delta_hat is part of the gate's
DEFINITION: token positions are action indices, kernels take seconds).

Reachability is checked A PRIORI (review 12-F6), before any threshold sweep:
p*(b_k) can never exceed p_star_max — the fixed point of one correction's odds
growth (capped by the humility bound M) against one gap's kernel decay — so any
theta above the propagated ceiling would make full-chunk execution structurally
unreachable for the belief arm while masked arms sail through. That asymmetry is
exactly the kind of silent bug the four-arm comparison cannot survive, which is why
it is a hard failure here, not a calibration surprise.

Masked-arm gate inputs (frozen mapping, D4 gate block): identical gate code runs in
every arm; what differs is only what each mechanism can natively supply. Arm 2's
stated confidence and Arm 1's classifier mass arrive as a CONSTANT p* across the
horizon (those mechanisms have no dynamics to propagate); Arm 0 supplies the uniform
floor. Only Arm 3 gets kernel propagation, because only Arm 3 has kernels.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from ..contracts.b5 import ProposalChunk
from ..contracts.goals import GOALS
from ..intent.tracker import Belief, TrackerParams, category_marginal, predict

THETA_CAP = 0.99   # theta never asks for more mass than a distribution can hold


@dataclass(frozen=True)
class GateThresholds:
    """Every STAIRCASE knob, swept POOLED on validation and then frozen — never
    per-arm, never tuned on the headline metric (threshold discipline, D4/D5).

    This owns K_commit only. State selection — including the OBSERVE/SUGGEST split
    and its theta_suggest — lives in fsm.py under D5's discounted-conf semantics,
    the canonical form by user decision 2026-07-06 (review P2-F1 resolved)."""

    c_min: float           # decoder-confidence floor (renamed from eps_tok, C5)
    theta_1: float         # top-mass demand at horizon position 1, reversible
    slope: float           # how much more mass each further position demands
    irrev_penalty: float   # extra demand for an irreversible action
    m_consecutive: int     # reads a candidate must hold before a state commits

    def theta(self, position: int, irreversible: bool) -> float:
        base = min(self.theta_1 + self.slope * (position - 1), THETA_CAP)
        return min(base + (self.irrev_penalty if irreversible else 0.0), THETA_CAP + 0.02)


def expected_gap(tick_sequences: list[list[int]]) -> dict:
    """delta_hat from validation sessions: the mean gap between consecutive
    corrections, in seconds. Reported per session too, so the freeze is auditable."""
    per_session = []
    for ticks in tick_sequences:
        gaps = [(after - before) / 20.0 for before, after in zip(ticks, ticks[1:])
                if after > before]
        if gaps:
            per_session.append(statistics.mean(gaps))
    if not per_session:
        raise ValueError("no correction gaps to freeze delta_hat from")
    return {"delta_hat_seconds": round(statistics.mean(per_session), 4),
            "per_session": [round(v, 4) for v in per_session]}


def _boost(p_star: float, humility: float) -> float:
    """One correction's best case: the top goal's odds against the field grow by at
    most the humility factor M (tracker Theorem 6)."""
    return humility * p_star / (humility * p_star + (1.0 - p_star))


def _decay(p_star: float, params: TrackerParams, delta_hat: float) -> float:
    """One inter-correction gap's kernel mixing, applied to the top-goal mass."""
    stay = math.exp(-params.lambda_g * delta_hat)
    return stay * p_star + (1.0 - stay) / len(GOALS)


def p_star_max(params: TrackerParams, delta_hat: float) -> float:
    """The ceiling on top-goal mass under this filter's own dynamics: the fixed
    point of boost-then-decay, iterated from uniform."""
    humility = params.humility_bound()
    p_star = 1.0 / len(GOALS)
    for _ in range(10_000):
        advanced = _decay(_boost(p_star, humility), params, delta_hat)
        if abs(advanced - p_star) < 1e-12:
            return advanced
        p_star = advanced
    return p_star


def propagated_bound(p_star: float, position: int, params: TrackerParams,
                     delta_hat: float) -> float:
    """D4's one-line prediction bound on the propagated top mass at horizon j."""
    stay = math.exp(-params.lambda_g * position * delta_hat)
    return stay * p_star + (1.0 - stay) / len(GOALS)


def reachability_report(thresholds: GateThresholds, params: TrackerParams,
                        delta_hat: float, horizon: int = 8) -> dict:
    """The a-priori check, run BEFORE any sweep freezes: every reversible-position
    theta must sit strictly under the ceiling's propagation, or the staircase can
    structurally never reach that position."""
    ceiling = p_star_max(params, delta_hat)
    positions = []
    for position in range(1, horizon + 1):
        bound = propagated_bound(ceiling, position, params, delta_hat)
        demand = thresholds.theta(position, irreversible=False)
        positions.append({"j": position, "bound": round(bound, 4),
                          "theta_reversible": round(demand, 4),
                          "reachable": demand < bound,
                          "margin": round(bound - demand, 4)})
    return {"p_star_max": round(ceiling, 4), "delta_hat": delta_hat,
            "lambda_g": params.lambda_g, "humility_bound": round(params.humility_bound(), 2),
            "positions": positions,
            "all_reachable": all(entry["reachable"] for entry in positions)}


def arm_p_star(slot_kind: str, belief: Belief | None, slot_p_top: float,
               position: int, params: TrackerParams, delta_hat: float) -> float:
    """The frozen masked-arm mapping: what each mechanism natively supplies as the
    horizon-j top mass. Identical gate code consumes the result everywhere."""
    if slot_kind == "arm3":
        if belief is None:
            raise ValueError("arm3 gate read without a belief")
        propagated = predict(belief, position * delta_hat, params)
        return max(category_marginal(propagated).values())
    if slot_kind == "arm0_zero":
        return 1.0 / len(GOALS)       # no mechanism: the uniform floor, every horizon
    return slot_p_top                 # arm1 / arm2: a constant — nothing to propagate


def k_commit(chunk: ProposalChunk, irreversible_flags: list[bool], slot_kind: str,
             belief: Belief | None, slot_p_top: float, params: TrackerParams,
             delta_hat: float, thresholds: GateThresholds) -> tuple[int, list[dict]]:
    """The staircase for one gate read: (K_commit, per-position trace rows).

    The trace rows are the proof-log's per_position block (D5 §7) — every number
    the decision was made from, recorded whether or not the position committed.
    """
    if len(irreversible_flags) != len(chunk.actions):
        raise ValueError("one irreversibility flag per action")
    committed = 0
    rows = []
    stopped = False
    for index, confidence in enumerate(chunk.token_conf):
        position = index + 1
        p_star = arm_p_star(slot_kind, belief, slot_p_top, position, params, delta_hat)
        demand = thresholds.theta(position, irreversible_flags[index])
        ok = confidence >= thresholds.c_min and p_star >= demand
        rows.append({"j": position, "token_conf": round(confidence, 4),
                     "p_star": round(p_star, 4), "theta": round(demand, 4),
                     "irrev": irreversible_flags[index], "ok": ok})
        if ok and not stopped:
            committed = position
        else:
            stopped = True
    return committed, rows


class CommitHysteresis:
    """K_commit may grow by at most +1 per gate read (D4). Shrinking is instant —
    caution never waits, only authority does."""

    def __init__(self):
        self.last = 0

    def read(self, raw_k_commit: int) -> int:
        allowed = min(raw_k_commit, self.last + 1)
        self.last = allowed
        return allowed


def thresholds_to_json(thresholds: GateThresholds) -> dict:
    return asdict(thresholds)
