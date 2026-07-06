"""Between typed actions and the decoder's token ids — both directions, losslessly.

The token language is deliberately tiny and structured (the FastSceneScript regime
D4 bets on): each action is one action-word token followed by its argument tokens,
where every integer coordinate is ONE token (65 values cover the pinned +/-32 range)
and every block name is one token from the pinned vocabulary. A chunk is
<bos> action tokens ... <eos>. No byte-pair pieces, no digits glued together — the
grammar's structure is visible to the model at the token level.

The vocabulary is built here, in one fixed order, so a token id means the same thing
in every checkpoint trained against this module version. Training provenance records
the vocabulary; loading a model checks it.

agreement() is D4's tau rule for the MTP confidence targets and the S2 verifier:
symbolic tokens must match exactly (tau = 0), numeric tokens may differ by up to
NUMERIC_TOLERANCE (tau = 2) — a placement two cells off is "the same idea", a
different action word never is.
"""
from __future__ import annotations

from .grammar import (
    BLOCK_VOCAB, COORD_RANGE, UTTERANCES, Action, Break, Look, Move, Place, Say, validate,
)

NUMERIC_TOLERANCE = 2   # tau for numeric tokens; symbolic tokens use exact match

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
_WORDS = ("<break>", "<look>", "<move>", "<place>", "<say>")
_NUMBERS = tuple(f"n{value}" for value in range(-COORD_RANGE, COORD_RANGE + 1))
_BLOCKS = tuple(f"b:{name}" for name in BLOCK_VOCAB)
_UTTERANCES = tuple(f"u:{name}" for name in UTTERANCES)

VOCAB: tuple[str, ...] = (PAD, BOS, EOS) + _WORDS + _NUMBERS + _BLOCKS + _UTTERANCES
_INDEX = {token: index for index, token in enumerate(VOCAB)}

PAD_ID, BOS_ID, EOS_ID = _INDEX[PAD], _INDEX[BOS], _INDEX[EOS]
_NUM_BASE = _INDEX[_NUMBERS[0]]
_NUM_END = _INDEX[_NUMBERS[-1]]
_BLOCK_BASE = _INDEX[_BLOCKS[0]]
# the numeric tokens' id range, inclusive — for vectorized tau checks
NUMERIC_ID_RANGE = (_NUM_BASE, _NUM_END)


def vocab_size() -> int:
    return len(VOCAB)


def is_numeric(token_id: int) -> bool:
    """True for the coordinate-value tokens — the ones the tau tolerance applies to."""
    return _NUM_BASE <= token_id <= _NUM_END


def numeric_value(token_id: int) -> int:
    return token_id - _NUM_BASE - COORD_RANGE


def _number(value: int) -> int:
    return _NUM_BASE + value + COORD_RANGE


def _block(name: str) -> int:
    # Unknown block names coarsen to the catch-all token; the typed action keeps the
    # real name, only the token form loses it.
    return _BLOCK_BASE + (BLOCK_VOCAB.index(name) if name in BLOCK_VOCAB else 0)


def encode_action(action: Action) -> list[int]:
    problem = validate(action)
    if problem:
        raise ValueError(problem)
    if isinstance(action, Say):
        return [_INDEX["<say>"], _INDEX[f"u:{action.utterance}"]]
    word = {Move: "<move>", Look: "<look>", Place: "<place>", Break: "<break>"}[type(action)]
    ids = [_INDEX[word], _number(action.dx), _number(action.dy), _number(action.dz)]
    if isinstance(action, Place):
        ids.append(_block(action.block))
    return ids


def encode_chunk(actions: list[Action]) -> list[int]:
    ids = [BOS_ID]
    for action in actions:
        ids.extend(encode_action(action))
    ids.append(EOS_ID)
    return ids


def decode_chunk(ids: list[int]) -> tuple[list[Action] | None, str | None]:
    """Token ids back to typed actions: (actions, None) or (None, reject reason).

    Same all-or-nothing rule as the JSON parser: one malformed stretch rejects the
    chunk. The decoder's sampler may hand us anything; this is where anything that
    is not exactly the grammar stops.
    """
    body = list(ids)
    if body and body[0] == BOS_ID:
        body = body[1:]
    if EOS_ID in body:
        body = body[:body.index(EOS_ID)]
    actions: list[Action] = []
    at = 0
    while at < len(body):
        token = VOCAB[body[at]] if 0 <= body[at] < len(VOCAB) else None
        if token == "<say>":
            if at + 1 >= len(body) or not VOCAB[body[at + 1]].startswith("u:"):
                return None, f"say without an utterance token at position {at}"
            actions.append(Say(utterance=VOCAB[body[at + 1]][2:]))
            at += 2
            continue
        if token not in ("<move>", "<look>", "<place>", "<break>"):
            return None, f"expected an action word at position {at}, got {token!r}"
        need = 4 if token != "<place>" else 5
        if at + need > len(body):
            return None, f"{token} truncated at position {at}"
        coords = body[at + 1: at + 4]
        if not all(is_numeric(c) for c in coords):
            return None, f"{token} with a non-numeric coordinate at position {at}"
        dx, dy, dz = (numeric_value(c) for c in coords)
        if token == "<place>":
            block_token = VOCAB[body[at + 4]] if 0 <= body[at + 4] < len(VOCAB) else ""
            if not block_token.startswith("b:"):
                return None, f"place without a block token at position {at}"
            actions.append(Place(dx, dy, dz, block_token[2:]))
        else:
            kind = {"<move>": Move, "<look>": Look, "<break>": Break}[token]
            actions.append(kind(dx, dy, dz))
        at += need
    if not actions:
        return None, "no actions between <bos> and <eos>"
    return actions, None


def agreement(token_a: int, token_b: int) -> bool:
    """D4's tau rule: exact for symbolic tokens, +/-NUMERIC_TOLERANCE for numeric."""
    if is_numeric(token_a) and is_numeric(token_b):
        return abs(numeric_value(token_a) - numeric_value(token_b)) <= NUMERIC_TOLERANCE
    return token_a == token_b
