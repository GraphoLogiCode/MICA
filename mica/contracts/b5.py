"""B5: one proposal chunk — what the decoder suggests doing next, with confidences.

CONTRACT C1, the one rule this module exists to hold: the decoder's token softmax
values and these per-action confidences are NEVER exported as likelihoods or filter
inputs. They are agreement scores (trained against the decoder's own one-step path,
FastSceneScript's recipe), not calibrated probabilities, and a filter fed its own
downstream's self-assessment would stop being a filter. Nothing in this module — and
nothing importing it — may hand these numbers to the tracker, the likelihood heads,
or the fusion layer. tests/test_decoder_contracts.py pins that as an import-graph
check.

rationale_goal is a faithfulness DIAGNOSTIC only (C3): at inference it is compared
against the belief's argmax; a mismatch is logged and the chunk rejected. It never
enters gate accept conditions, never enters the belief, is never scored as intent
accuracy, never surfaced to the human.

token_conf convention (D4): conf_1 := 1.0 — the first action is the plain next-token
prediction, accepted by default; truncation checks start at the second position.
Each action's confidence is the MINIMUM over its own tokens' agreement scores, the
conservative read: an action is only as trustworthy as its shakiest token.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..decoder.grammar import Action, validate


@dataclass(frozen=True)
class ProposalChunk:
    """K proposed actions, K confidences, and the goal the proposal serves."""

    actions: tuple[Action, ...]
    token_conf: tuple[float, ...]
    rationale_goal: str | None

    def __post_init__(self):
        if len(self.actions) != len(self.token_conf):
            raise ValueError("one confidence per action, no more, no less")
        if not self.actions:
            raise ValueError("an empty proposal is not a proposal")
        for action in self.actions:
            problem = validate(action)
            if problem:
                raise ValueError(problem)
        if any(not 0.0 <= c <= 1.0 for c in self.token_conf):
            raise ValueError("confidences live in [0, 1]")
        if self.token_conf[0] != 1.0:
            raise ValueError("conf_1 := 1 by convention (D4); got "
                             f"{self.token_conf[0]}")
