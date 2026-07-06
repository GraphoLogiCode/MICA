"""The action grammar and tokenizer: round-trips are lossless, malformed input is
rejected with a reason, and the tau agreement rule does exactly what D4 pins."""
import json

from mica.decoder import tokenizer
from mica.decoder.grammar import (
    BLOCK_VOCAB, COORD_RANGE, Break, Look, Move, Place, Say,
    action_from_json, action_to_json, parse_proposal, proposal_to_json,
)

_EVERY_KIND = [
    Move(1, 0, -3), Look(-2, 5, 0), Place(4, 1, 4, "minecraft:oak_planks"),
    Break(0, -1, 2), Say("ask_goal"),
]


def test_json_round_trip_every_action_kind():
    for action in _EVERY_KIND:
        assert action_from_json(action_to_json(action)) == action


def test_token_round_trip_every_action_kind():
    ids = tokenizer.encode_chunk(_EVERY_KIND)
    assert ids[0] == tokenizer.BOS_ID and ids[-1] == tokenizer.EOS_ID
    decoded, reason = tokenizer.decode_chunk(ids)
    assert reason is None
    assert decoded == _EVERY_KIND


def test_proposal_round_trip_with_rationale():
    text = proposal_to_json(_EVERY_KIND, "defense")
    actions, rationale, reason = parse_proposal(text)
    assert reason is None and rationale == "defense" and actions == _EVERY_KIND


def test_parser_rejects_each_malformation_with_a_reason():
    bad = [
        "",                                                    # empty
        "not json",                                            # not JSON
        json.dumps({"rationale_goal": "defense"}),             # no actions list
        json.dumps({"actions": []}),                           # empty actions
        json.dumps({"actions": [{"do": "fly", "pos": [0, 0, 0]}]}),      # unknown word
        json.dumps({"actions": [{"do": "place", "pos": [0, 0]}]}),       # short pos
        json.dumps({"actions": [{"do": "place", "pos": [0, 0.5, 0],
                                 "block": "b"}]}),                       # float coord
        json.dumps({"actions": [{"do": "place", "pos": [0, 0, 0]}]}),    # no block
        json.dumps({"actions": [{"do": "move",
                                 "pos": [COORD_RANGE + 1, 0, 0]}]}),     # out of range
        json.dumps({"actions": [{"do": "say", "utterance": "yo"}]}),     # unknown say
        json.dumps({"actions": [{"do": "move", "pos": [0, 0, 0]}],
                    "rationale_goal": "castle"}),              # rationale not a goal
    ]
    for text in bad:
        actions, _, reason = parse_proposal(text)
        assert actions is None and reason, f"should reject: {text!r}"


def test_one_bad_action_rejects_the_whole_chunk():
    text = json.dumps({"actions": [action_to_json(Move(0, 0, 0)),
                                   {"do": "move", "pos": [99, 0, 0]}]})
    actions, _, reason = parse_proposal(text)
    assert actions is None and "action 1" in reason


def test_vocab_is_unique_and_covers_the_coordinate_range():
    assert len(set(tokenizer.VOCAB)) == len(tokenizer.VOCAB)
    assert tokenizer.vocab_size() == len(tokenizer.VOCAB)
    low, high = tokenizer.NUMERIC_ID_RANGE
    assert high - low + 1 == 2 * COORD_RANGE + 1
    assert tokenizer.numeric_value(low) == -COORD_RANGE
    assert tokenizer.numeric_value(high) == COORD_RANGE


def test_agreement_tau_rule():
    low, _ = tokenizer.NUMERIC_ID_RANGE
    zero = low + COORD_RANGE                      # the token for value 0
    assert tokenizer.agreement(zero, zero + 2)    # numeric, within tau = 2
    assert not tokenizer.agreement(zero, zero + 3)
    assert tokenizer.agreement(tokenizer.BOS_ID, tokenizer.BOS_ID)
    assert not tokenizer.agreement(tokenizer.BOS_ID, tokenizer.EOS_ID)
    assert not tokenizer.agreement(zero, tokenizer.BOS_ID)  # numeric vs symbolic


def test_decode_rejects_malformed_token_streams():
    place_word = tokenizer.VOCAB.index("<place>")
    truncated = [tokenizer.BOS_ID, place_word, tokenizer.EOS_ID]
    actions, reason = tokenizer.decode_chunk(truncated)
    assert actions is None and reason
    empty = [tokenizer.BOS_ID, tokenizer.EOS_ID]
    actions, reason = tokenizer.decode_chunk(empty)
    assert actions is None and reason


def test_unknown_block_coarsens_to_the_catch_all_token():
    ids = tokenizer.encode_action(Place(0, 0, 0, "minecraft:diamond_block"))
    decoded, reason = tokenizer.decode_chunk([tokenizer.BOS_ID] + ids + [tokenizer.EOS_ID])
    assert reason is None
    assert decoded[0].block == BLOCK_VOCAB[0]     # "<other>"
