"""The decoder network on a tiny config: shapes hold, proposals respect the B5
conventions, the per-arm slot projections exist, and the C2 rationale mask
actually zeroes the loss."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")

from mica.contracts.b4 import SLOT_KINDS, ControlContext
from mica.decoder import model as decoder_model
from mica.decoder import tokenizer
from mica.decoder.context import EVIDENCE_DIM
from mica.decoder.grammar import Place

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import train_decoder  # noqa: E402

_TINY = decoder_model.DecoderConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64,
                                    dropout=0.0, soft_tokens=2, mtp_horizon=3,
                                    max_target_tokens=16)


def _context(slot_kind="arm3"):
    return ControlContext(
        tick=100, k=1, belief_snapshot_id=100, slot_kind=slot_kind,
        goal_marginal=(0.6, 0.1, 0.1, 0.1, 0.1) if slot_kind != "arm0_zero"
        else (0.0,) * 5,
        top_goal="habitation" if slot_kind != "arm0_zero" else None,
        p_top=0.6 if slot_kind != "arm0_zero" else 0.0,
        entropy_nats=1.2 if slot_kind != "arm0_zero" else 0.0, p_z1=0.5,
        evidence=tuple(0.1 for _ in range(EVIDENCE_DIM)))


def test_every_slot_kind_has_its_own_projection():
    model = decoder_model.build_model(_TINY)
    assert set(model.slot_proj.keys()) == set(SLOT_KINDS)


def test_context_tensors_shapes():
    evidence, slot, kind = decoder_model.context_tensors(_context())
    assert evidence.shape == (EVIDENCE_DIM,)
    assert slot.shape == (decoder_model.SLOT_DIM,)
    assert kind.dim() == 0


def test_propose_respects_b5_or_rejects_with_a_reason():
    torch.manual_seed(0)
    model = decoder_model.build_model(_TINY).eval()
    for slot_kind in SLOT_KINDS:
        proposal, reason = decoder_model.propose(model, _context(slot_kind),
                                                 max_actions=3)
        # an untrained model may emit garbage — then the parser must say why
        if proposal is None:
            assert reason
        else:
            assert proposal.token_conf[0] == 1.0
            assert len(proposal.actions) <= 3
            if slot_kind != "arm3":
                assert proposal.rationale_goal is None   # C2: untrained => unread


def test_finalize_chunk_names_all_three_outcomes():
    """The F6 fix: 'the build needs nothing' travels under its own name, a
    truncated tail trims to the complete prefix, and garbage stays rejected."""
    place = Place(1, 0, 0, "minecraft:oak_planks")
    place_ids = tokenizer.encode_action(place)

    done, reason = decoder_model._finalize_chunk([tokenizer.BOS_ID], [], None, 8)
    assert done is None and reason == decoder_model.NOTHING_TO_DO

    dangling = [tokenizer.BOS_ID] + place_ids + [tokenizer.VOCAB.index("<place>")]
    chunk, reason = decoder_model._finalize_chunk(
        dangling, [1.0] * (len(dangling) - 1), None, 8)
    assert reason is None and len(chunk.actions) == 1 and chunk.actions[0] == place

    low, _ = tokenizer.NUMERIC_ID_RANGE
    garbage, reason = decoder_model._finalize_chunk(
        [tokenizer.BOS_ID, low, low], [1.0, 1.0], None, 8)
    assert garbage is None and reason and reason != decoder_model.NOTHING_TO_DO


def test_c2_rationale_loss_is_zero_on_masked_slots():
    torch.manual_seed(0)
    model = decoder_model.build_model(_TINY)
    ids = tokenizer.encode_chunk([Place(1, 0, 0, "minecraft:oak_planks"),
                                  Place(2, 0, 0, "minecraft:oak_planks")])
    batch = {
        "ids": torch.tensor([ids], dtype=torch.long),
        "evidence": torch.zeros((1, EVIDENCE_DIM)),
        "slot": torch.zeros((1, decoder_model.SLOT_DIM)),
        "kinds": torch.tensor([SLOT_KINDS.index("arm0_zero")]),
        "rationale_active": torch.tensor([False]),
        "goal_index": torch.tensor([0]),
    }
    _, _, rationale_loss = train_decoder._losses(model, batch, torch.device("cpu"),
                                                 horizons=2)
    assert rationale_loss == 0.0
    batch["rationale_active"] = torch.tensor([True])
    batch["kinds"] = torch.tensor([SLOT_KINDS.index("arm3")])
    _, _, active_loss = train_decoder._losses(model, batch, torch.device("cpu"),
                                              horizons=2)
    assert active_loss > 0.0


def test_losses_backpropagate_through_stage_b():
    torch.manual_seed(0)
    model = decoder_model.build_model(_TINY)
    ids = tokenizer.encode_chunk([Place(1, 0, 0, "minecraft:oak_planks"),
                                  Place(2, 0, 0, "minecraft:oak_planks"),
                                  Place(3, 0, 0, "minecraft:oak_planks")])
    batch = {
        "ids": torch.tensor([ids], dtype=torch.long),
        "evidence": torch.rand((1, EVIDENCE_DIM)),
        "slot": torch.rand((1, decoder_model.SLOT_DIM)),
        "kinds": torch.tensor([SLOT_KINDS.index("arm3")]),
        "rationale_active": torch.tensor([True]),
        "goal_index": torch.tensor([2]),
    }
    loss, nll, _ = train_decoder._losses(model, batch, torch.device("cpu"),
                                         horizons=_TINY.mtp_horizon)
    assert nll is not None and nll > 0.0
    loss.backward()
    assert model.conf_head.weight.grad is not None      # the confidence trained
    assert model.token_head.weight.grad is not None
