"""The belief tracker: a recursive posterior over (goal category, reasoning mode).

This is Algorithm 1 of the Mathematical Foundations note, implemented line for line:
predict over the elapsed gap with the resample kernels, correct with an ε-floored
likelihood, normalize. Every guarantee the note proves is a checkable property here —
mass is preserved (Theorem 2), the normalizer can never reach zero (Theorem 4), lazy
prediction over one long gap equals many short ones (Theorem 5), one observation can
shift any pair of odds by at most the humility factor M (Theorem 6), and a likelihood
that is constant across goals leaves the goal belief alone (Proposition 7).

The belief runs over the five goal CATEGORIES times the two reasoning modes — ten
numbers. With the floor keeping everything strictly positive, plain floats are exact
enough; no log-space is needed at this size.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from ..contracts.goals import GOALS

MODES = (0, 1)   # 0 = deliberative (goal-directed), 1 = heuristic (shortcut/habit)

Belief = dict[tuple[str, int], float]


@dataclass(frozen=True)
class TrackerParams:
    """The model's three knobs, in the units the math defines.

    Rates are per second — never per step — so the kernels stay consistent no matter
    how irregular the correction cadence is. action_count is |A|, the emitted
    macro-action classes, which sets the likelihood floor ε/|A|.
    """

    lambda_g: float = 0.05   # goal-switching rate; goals persist
    lambda_z: float = 1.0    # mode-switching rate; modes flip freely
    epsilon: float = 0.01    # probability the human acts outside the model
    action_count: int = 5    # |A|: place, break, navigate, inspect, idle (v1 emits these)

    def humility_bound(self) -> float:
        """M: no single correction may multiply any pair of odds by more than this."""
        return 1.0 + self.action_count * (1.0 - self.epsilon) / self.epsilon


def uniform_belief() -> Belief:
    share = 1.0 / (len(GOALS) * len(MODES))
    return {(g, z): share for g in GOALS for z in MODES}


def _mix_goal_axis(belief: Belief, stay: float) -> Belief:
    # Resample kernel on the goal axis: keep the goal with probability `stay`, else
    # redraw uniformly — per mode column, that is a mix toward the column's mean.
    mixed = {}
    for z in MODES:
        column_mean = sum(belief[(g, z)] for g in GOALS) / len(GOALS)
        for g in GOALS:
            mixed[(g, z)] = stay * belief[(g, z)] + (1.0 - stay) * column_mean
    return mixed


def _mix_mode_axis(belief: Belief, stay: float) -> Belief:
    mixed = {}
    for g in GOALS:
        row_mean = sum(belief[(g, z)] for z in MODES) / len(MODES)
        for z in MODES:
            mixed[(g, z)] = stay * belief[(g, z)] + (1.0 - stay) * row_mean
    return mixed


def predict(belief: Belief, dt: float, params: TrackerParams) -> Belief:
    """Carry the belief across a gap of dt seconds with no observation.

    The two axis kernels commute, and each composes exactly over time (Chapman-
    Kolmogorov), so predicting one long gap equals predicting its pieces — which is
    why prediction-only steps never need to be stored.
    """
    belief = _mix_goal_axis(belief, math.exp(-params.lambda_g * dt))
    return _mix_mode_axis(belief, math.exp(-params.lambda_z * dt))


def floored(likelihood: Mapping[tuple[str, int], float], params: TrackerParams) -> Belief:
    """The model statement 'with probability ε the human acts outside the model':
    guarantees a positive normalizer and caps how hard one step can move the belief.

    The humility bound M needs every likelihood entry in [0, 1] (Theorem 6's proof
    caps the floored value at (1-ε)+ε/|A|). Both shipped heads are softmaxes, but a
    future head returning unnormalized scores would silently void M for the gate —
    so the precondition fails loudly here instead (review 13 F31)."""
    assert all(0.0 <= value <= 1.0 + 1e-9 for value in likelihood.values()), (
        "likelihood entries must lie in [0, 1] or the humility bound M is void")
    floor = params.epsilon / params.action_count
    return {key: (1.0 - params.epsilon) * value + floor for key, value in likelihood.items()}


def correct(
    belief: Belief, likelihood: Mapping[tuple[str, int], float], params: TrackerParams
) -> tuple[Belief, float]:
    """One observation: weight the predicted belief by the floored likelihood, normalize.
    Returns the new belief and the normalizer Z_k (provably >= ε/|A| > 0).

    Positivity of the OUTPUT relies on the belief coming in strictly positive — which
    predict() guarantees (the kernel re-mixes mass toward uniform). Calling correct()
    repeatedly with no predict between can drive entries to exact zero (measured:
    ~120 one-hot corrections), so the invariant is asserted instead of assumed
    (review 13 F19)."""
    assert min(belief.values()) > 0.0, (
        "correct() needs a strictly positive belief - call predict() first "
        "(a dead entry here means a predict step was skipped)")
    weighted = floored(likelihood, params)
    numerator = {key: weighted[key] * belief[key] for key in belief}
    normalizer = sum(numerator.values())
    return {key: value / normalizer for key, value in numerator.items()}, normalizer


def category_marginal(belief: Belief) -> dict[str, float]:
    return {g: sum(belief[(g, z)] for z in MODES) for g in GOALS}


def mode_marginal(belief: Belief) -> dict[int, float]:
    return {z: sum(belief[(g, z)] for g in GOALS) for z in MODES}


def entropy(belief: Belief) -> float:
    return -sum(p * math.log(p) for p in belief.values() if p > 0.0)
