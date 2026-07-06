"""The agent action grammar: the five things the helper can do, as typed values.

D4 decision 3 pins the grammar — move / look / place / break / say, with numeric
arguments — and this module is its single definition. The decoder emits token
sequences that spell these actions; the tokenizer (tokenizer.py) converts between
the two forms; everything downstream (the B5 proposal chunk, the commit gate, the
coherence checks) works with the typed values, never with raw token ids.

Coordinates are integer offsets RELATIVE TO THE BUILD ORIGIN — the cell of the first
human placement in the session. That frame is observable in every arm from evidence
alone, carries no goal information, and keeps every number small: template builds
stay within a few dozen cells of where they start. Offsets are pinned to
[-COORD_RANGE, +COORD_RANGE]; anything outside is a malformed action.

Proposals also travel as strict JSON (one object per action) so a proposal can be
logged, replayed, and — later — sent to the embodiment. parse_proposal is the robust
door: malformed decoder output is rejected with a reason, never crashed on and never
silently repaired into something the decoder did not say.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..contracts.goals import GOALS

COORD_RANGE = 32   # offsets live in [-32, 32] on every axis

# What the helper is allowed to say, as a tiny fixed set. v1 corpus traces never use
# these (the scripted helper builds, it does not chat), but the grammar is complete so
# B5 does not need to change when the live agent starts talking.
UTTERANCES = ("suggest_next", "ask_goal", "announce_place", "announce_yield")

# Every block the decoder may name in a PLACE: the scripted palettes plus the concrete
# template blocks. "<other>" is the tokenizer's catch-all for block names outside this
# list (a parsed proposal keeps the real name; only the token form coarsens it).
BLOCK_VOCAB = (
    "<other>",
    "minecraft:cobblestone",
    "minecraft:oak_fence",
    "minecraft:oak_log",
    "minecraft:oak_planks",
    "minecraft:oak_slab",
    "minecraft:poppy",
    "minecraft:spruce_planks",
    "minecraft:stone_bricks",
    "minecraft:water",
)


@dataclass(frozen=True)
class Move:
    """Walk to stand near the target cell."""

    dx: int
    dy: int
    dz: int


@dataclass(frozen=True)
class Look:
    """Turn the head toward the target cell."""

    dx: int
    dy: int
    dz: int


@dataclass(frozen=True)
class Place:
    """Put one block of the named type at the target cell."""

    dx: int
    dy: int
    dz: int
    block: str


@dataclass(frozen=True)
class Break:
    """Remove the block at the target cell."""

    dx: int
    dy: int
    dz: int


@dataclass(frozen=True)
class Say:
    """Speak one of the pinned utterances in chat."""

    utterance: str


Action = Move | Look | Place | Break | Say

_WORDS = {Move: "move", Look: "look", Place: "place", Break: "break", Say: "say"}


def _in_range(*values: int) -> bool:
    return all(isinstance(v, int) and -COORD_RANGE <= v <= COORD_RANGE for v in values)


def validate(action: Action) -> str | None:
    """None when the action is well-formed, else the reason it is not."""
    if isinstance(action, Say):
        if action.utterance not in UTTERANCES:
            return f"unknown utterance {action.utterance!r}"
        return None
    if not _in_range(action.dx, action.dy, action.dz):
        return f"coordinate outside +/-{COORD_RANGE}: ({action.dx},{action.dy},{action.dz})"
    if isinstance(action, Place) and not isinstance(action.block, str):
        return "place without a block name"
    return None


def action_to_json(action: Action) -> dict:
    """One action as the strict JSON object form."""
    if isinstance(action, Say):
        return {"do": "say", "utterance": action.utterance}
    body: dict = {"do": _WORDS[type(action)], "pos": [action.dx, action.dy, action.dz]}
    if isinstance(action, Place):
        body["block"] = action.block
    return body


def action_from_json(body: dict) -> Action:
    """The reverse door. Raises ValueError on anything malformed."""
    if not isinstance(body, dict):
        raise ValueError("action is not an object")
    word = body.get("do")
    if word == "say":
        action: Action = Say(utterance=str(body.get("utterance", "")))
    elif word in ("move", "look", "place", "break"):
        pos = body.get("pos")
        if (not isinstance(pos, (list, tuple)) or len(pos) != 3
                or not all(isinstance(v, int) for v in pos)):
            raise ValueError(f"{word}: pos must be three integers, got {pos!r}")
        if word == "place":
            block = body.get("block")
            if not isinstance(block, str) or not block:
                raise ValueError("place: missing block name")
            action = Place(pos[0], pos[1], pos[2], block)
        else:
            kind = {"move": Move, "look": Look, "break": Break}[word]
            action = kind(pos[0], pos[1], pos[2])
    else:
        raise ValueError(f"unknown action word {word!r}")
    problem = validate(action)
    if problem:
        raise ValueError(problem)
    return action


def proposal_to_json(actions: list[Action], rationale_goal: str | None) -> str:
    """A whole proposal chunk in its strict JSON wire form."""
    return json.dumps({"actions": [action_to_json(a) for a in actions],
                       "rationale_goal": rationale_goal})


def parse_proposal(text: str | None) -> tuple[list[Action] | None, str | None, str | None]:
    """The robust parser: (actions, rationale_goal, reject_reason).

    Either actions is a list and reject_reason is None, or actions is None and
    reject_reason says exactly what was wrong. Nothing in between: a chunk with one
    malformed action is rejected whole, because executing "most of" a proposal the
    decoder half-garbled is how silent bugs become placed blocks.
    """
    if not text or not text.strip():
        return None, None, "empty proposal"
    try:
        body = json.loads(text)
    except ValueError:
        return None, None, "not valid JSON"
    if not isinstance(body, dict) or not isinstance(body.get("actions"), list):
        return None, None, "no actions list"
    rationale = body.get("rationale_goal")
    if rationale is not None and rationale not in GOALS:
        return None, None, f"rationale_goal {rationale!r} is not a goal category"
    actions = []
    for index, entry in enumerate(body["actions"]):
        try:
            actions.append(action_from_json(entry))
        except ValueError as error:
            return None, None, f"action {index}: {error}"
    if not actions:
        return None, None, "empty actions list"
    return actions, rationale, None
