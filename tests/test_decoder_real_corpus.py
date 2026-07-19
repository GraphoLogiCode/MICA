"""The label-free real-session door into the decoder corpus (pre-registered
2026-07-19): human events become plan-shaped targets with A7 at the source, no
goal anywhere in a real row, and the trainer's masking honors that."""
import random
import sys
import os
import types

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos
from mica.data import decoder_corpus


def _session(events):
    packets = []
    for tick, cell, actor, op in events:
        event = BlockEvent(event_id=len(packets), pos=BlockPos(*cell),
                           block_type="minecraft:stone", op=op, actor=actor)
        packets.append(types.SimpleNamespace(
            tick=tick, server=types.SimpleNamespace(block_events=[event])))
    return types.SimpleNamespace(packets=packets)


def test_real_placements_are_a7_filtered_at_the_source():
    session = _session([
        (10, (1, 64, 1), "HumanBuilder", BlockOp.PLACE),
        (20, (2, 64, 1), "MICA_AI", BlockOp.PLACE),        # the agent's own block
        (30, (3, 64, 1), "MICA_AI_2", BlockOp.PLACE),      # a replica's block
        (40, (4, 64, 1), "HumanBuilder", BlockOp.PLACE),
    ])
    placements, dropped = decoder_corpus.real_session_placements(session)
    assert dropped == 2
    assert [(p.pos.x, p.tick) for p in placements] == [(1, 10), (4, 40)]
    # the human's break-corrects-place pattern feeds the SAME mistake pairing
    assert decoder_corpus.skipped_place_ids(placements) == set()


def test_out_of_range_events_are_dropped_and_counted():
    session = _session([
        (10, (1, 64, 1), "HumanBuilder", BlockOp.PLACE),
        (20, (500, 64, 1), "HumanBuilder", BlockOp.PLACE),   # wandered far away
    ])
    placements, _ = decoder_corpus.real_session_placements(session)
    kept, dropped = decoder_corpus.in_range_placements(placements, (0, 64, 0))
    assert dropped == 1
    assert [p.pos.x for p in kept] == [1]


def test_real_rows_carry_no_goal_and_the_trainer_masks_them():
    from train_decoder import _draw_slot, _prepare

    row = {"target": [], "evidence": [0.0] * 8, "goal": None,
           "slots": {"arm3": {"goal_marginal": [0.2] * 5, "top_goal": "habitation",
                              "p_top": 0.2, "entropy_nats": 1.6, "p_z1": 0.5},
                     "arm1": None, "arm2": None}}
    sample = _prepare([row])[0]
    assert sample["goal_index"] == -1
    rng = random.Random(7)
    for _ in range(200):    # every draw: never rationale-active, never sharpened
        vector, _, rationale_active = _draw_slot(sample, rng)
        assert rationale_active is False
        assert float(max(vector[:5])) < 1.0     # no point-mass sharpening possible


def test_load_rows_never_routes_real_groups_into_the_standard_split(tmp_path):
    import json

    labels = {"scripted-a": {"goal": "habitation", "group": "train"},
              "scripted-b": {"goal": "defense", "group": "holdout"},
              "fabric-real-1": {"goal": None, "group": "real_pretrain"},
              "fabric-real-2": {"goal": None, "group": "real_holdout"}}
    with open(tmp_path / "decoder_labels.json", "w", encoding="utf-8") as handle:
        json.dump(labels, handle)
    for sid in labels:
        with open(tmp_path / f"{sid}.decoder.jsonl", "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"session": sid, "target": []}) + "\n")
    train, evaluation, _ = decoder_corpus.load_rows(str(tmp_path))
    assert {r["session"] for r in train} == {"scripted-a"}
    assert {r["session"] for r in evaluation} == {"scripted-b"}
    assert [r["session"] for r in decoder_corpus.load_group(
        "real_pretrain", str(tmp_path))] == ["fabric-real-1"]
    assert [r["session"] for r in decoder_corpus.load_group(
        "real_holdout", str(tmp_path))] == ["fabric-real-2"]
