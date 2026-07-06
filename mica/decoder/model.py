"""The from-scratch action decoder D_psi: a small transformer with an MTP retrofit.

Backbone (pinned by user decision 2026-07-05): a from-scratch decoder-only
transformer in the 15-30M range — the FastSceneScript regime of small, structured,
low-entropy token languages, NOT the billion-parameter code regime where MTP hurts
small models. The bet is stated in the D4 note; the OQ1 check in the eval script
measures it instead of assuming it.

How c_k enters (D4 decision 1): the context becomes a PREFIX —
  m learned soft tokens        projected from the arm-invariant evidence vector
  one intent-slot token        the slot's numbers through a PER-ARM projection plus
                               a slot-kind embedding (the maskable, intervenable slot;
                               per-arm projections are frozen with the checkpoint, C6)
then <bos> and the action tokens follow. The decoder reads the human only through
this prefix (C4).

MTP retrofit (D4 decision 2, FastSceneScript recipe): ONE shared projection block
(two FFN blocks, conditioned on the horizon offset through a small offset embedding)
feeding ONE shared token head and ONE shared confidence head. Stage A trains only
offset 1 (plain next-token prediction); Stage B turns on offsets 2..n and the
agreement-trained confidence. The confidence head scores agreement with the model's
own one-step path — it is NOT a calibrated probability, and by contract C1 neither
it nor the token softmax ever reaches the filter.

rationale_goal (B5) reads off the intent-slot token's final hidden state. Its CE
loss is active ONLY on unmasked arm3-slot samples (C2) and at inference it is a
faithfulness diagnostic only (C3).

Per-token confidence at generation time: the token generated at chunk position j
gets the confidence the model computed for it ONE STEP EARLIER (the offset-2 read —
the shortest trained horizon). The first token has no earlier read and is the NTP
token itself: conf_1 := 1 by the D4 convention. An action's confidence is the MIN
over its tokens' confidences (the conservative read B5 pins).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from ..contracts.b4 import SLOT_KINDS, ControlContext
from ..contracts.b5 import ProposalChunk
from ..contracts.goals import GOALS
from . import tokenizer
from .context import EVIDENCE_DIM
from .grammar import Action

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WEIGHTS_PATH = os.path.join(_ROOT, "models", "decoder_v1.pt")
META_PATH = os.path.join(_ROOT, "models", "decoder_v1.json")
STAGE_A_PATH = os.path.join(_ROOT, "models", "decoder_v1_stage_a.pt")

SLOT_DIM = len(GOALS) + 3   # goal marginal + p_top + entropy_nats + p_z1

# The decoder's "build looks finished" signal: it closed the chunk before emitting
# a single action. B5 rightly forbids empty chunks, so this travels as a named
# rejection reason the gate reads as OBSERVE — never to be confused with garbage
# output (review F6: the two used to be indistinguishable).
NOTHING_TO_DO = "nothing to do: the decoder closed the chunk with no actions"


@dataclass(frozen=True)
class DecoderConfig:
    """Everything that fixes the network's shape. Ships in the provenance json."""

    vocab: int = tokenizer.vocab_size()
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    soft_tokens: int = 4          # m: learned soft prefix tokens from the evidence
    mtp_horizon: int = 8          # n: parallel prediction depth in TOKENS (n <= 8)
    max_target_tokens: int = 64   # room for TARGET_ACTIONS place actions + <eos>
    evidence_dim: int = EVIDENCE_DIM

    @property
    def prefix_len(self) -> int:
        return self.soft_tokens + 1   # soft tokens + the intent-slot token


def slot_tensor(goal_marginal, p_top: float, entropy_nats: float, p_z1: float):
    """The slot's numbers in the fixed order the per-arm projections expect."""
    import torch

    return torch.tensor(list(goal_marginal) + [p_top, entropy_nats, p_z1],
                        dtype=torch.float32)


def build_model(config: DecoderConfig):
    """The network. Defined in a function so torch stays out of import time."""
    import torch
    from torch import nn

    class ActionDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            d = config.d_model
            self.token_embed = nn.Embedding(config.vocab, d)
            self.pos_embed = nn.Embedding(
                config.prefix_len + 1 + config.max_target_tokens + 1, d)
            # the evidence half of c_k -> m soft prefix tokens
            self.evidence_proj = nn.Sequential(
                nn.Linear(config.evidence_dim, 512), nn.GELU(),
                nn.Linear(512, config.soft_tokens * d))
            # the intent slot: one projection PER ARM (frozen with the checkpoint)
            # plus a slot-kind embedding so the model knows which mechanism spoke
            self.slot_proj = nn.ModuleDict(
                {kind: nn.Linear(SLOT_DIM, d) for kind in SLOT_KINDS})
            self.kind_embed = nn.Embedding(len(SLOT_KINDS), d)
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=config.n_heads, dim_feedforward=config.d_ff,
                dropout=config.dropout, activation="gelu", batch_first=True,
                norm_first=True)
            self.trunk = nn.TransformerEncoder(layer, num_layers=config.n_layers,
                                               enable_nested_tensor=False)
            self.final_norm = nn.LayerNorm(d)
            # the MTP retrofit: offset embedding + two FFN blocks, shared across
            # every horizon position — then ONE token head and ONE confidence head
            self.offset_embed = nn.Embedding(config.mtp_horizon, d)
            self.mtp_norm1 = nn.LayerNorm(d)
            self.mtp_ffn1 = nn.Sequential(nn.Linear(d, config.d_ff), nn.GELU(),
                                          nn.Linear(config.d_ff, d))
            self.mtp_norm2 = nn.LayerNorm(d)
            self.mtp_ffn2 = nn.Sequential(nn.Linear(d, config.d_ff), nn.GELU(),
                                          nn.Linear(config.d_ff, d))
            self.token_head = nn.Linear(d, config.vocab)
            self.conf_head = nn.Linear(d, 1)
            self.rationale_head = nn.Linear(d, len(GOALS))

        def prefix(self, evidence, slot, slot_kinds):
            """(batch, evidence_dim) + (batch, SLOT_DIM) + kind ids -> the prefix."""
            batch = evidence.shape[0]
            soft = self.evidence_proj(evidence).view(
                batch, self.config.soft_tokens, self.config.d_model)
            slot_vecs = torch.stack([
                self.slot_proj[SLOT_KINDS[int(kind)]](slot[row])
                for row, kind in enumerate(slot_kinds)])
            slot_vecs = slot_vecs + self.kind_embed(slot_kinds)
            return torch.cat([soft, slot_vecs.unsqueeze(1)], dim=1)

        def trunk_hidden(self, evidence, slot, slot_kinds, token_ids):
            """Causal trunk pass. Returns hidden states for every position."""
            prefix = self.prefix(evidence, slot, slot_kinds)
            tokens = self.token_embed(token_ids)
            sequence = torch.cat([prefix, tokens], dim=1)
            positions = torch.arange(sequence.shape[1], device=sequence.device)
            sequence = sequence + self.pos_embed(positions)
            mask = nn.Transformer.generate_square_subsequent_mask(
                sequence.shape[1], device=sequence.device)
            hidden = self.trunk(sequence, mask=mask, is_causal=True)
            return self.final_norm(hidden)

        def project(self, hidden, offset_index):
            """The shared projection block at one horizon offset (0 = next token)."""
            h = hidden + self.offset_embed.weight[offset_index]
            h = h + self.mtp_ffn1(self.mtp_norm1(h))
            return h + self.mtp_ffn2(self.mtp_norm2(h))

        def token_logits(self, hidden, offset_index=0):
            return self.token_head(self.project(hidden, offset_index))

        def confidence(self, hidden, offset_index):
            return torch.sigmoid(self.conf_head(self.project(hidden, offset_index)))

        def rationale_logits(self, hidden):
            """Read the goal the proposal serves off the slot token's hidden state."""
            return self.rationale_head(hidden[:, self.config.soft_tokens, :])

    return ActionDecoder()


def available() -> bool:
    return os.path.exists(WEIGHTS_PATH) and os.path.exists(META_PATH)


def load(path: str = WEIGHTS_PATH, meta_path: str = META_PATH):
    """A shipped checkpoint plus its config, vocabulary-checked."""
    import torch

    with open(meta_path, encoding="utf-8") as handle:
        meta = json.load(handle)
    if tuple(meta["vocab_tokens"]) != tokenizer.VOCAB:
        raise ValueError("decoder was trained against a different token vocabulary")
    config = DecoderConfig(**meta["config"])
    model = build_model(config)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, config, meta


def save_meta(config: DecoderConfig, extra: dict, path: str = META_PATH) -> None:
    body = {"config": asdict(config), "vocab_tokens": list(tokenizer.VOCAB),
            "goals": list(GOALS), "slot_kinds": list(SLOT_KINDS)}
    body.update(extra)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(body, handle, indent=2)


def context_tensors(ctx: ControlContext):
    """One B4 record -> the (evidence, slot, kind id) tensors the model reads."""
    import torch

    evidence = torch.tensor(ctx.evidence, dtype=torch.float32)
    slot = slot_tensor(ctx.goal_marginal, ctx.p_top, ctx.entropy_nats, ctx.p_z1)
    kind = torch.tensor(SLOT_KINDS.index(ctx.slot_kind), dtype=torch.long)
    return evidence, slot, kind


def propose(model, ctx: ControlContext, max_actions: int = 8,
            ) -> tuple[ProposalChunk | None, str | None]:
    """Greedy chunk generation for one context: (proposal, None) or (None, reason).

    Autoregressive over the next-token head; each generated token carries the
    confidence the previous position's offset-2 read gave it. Confidence per action
    is the min over that action's tokens; the first action is 1.0 by convention.
    """
    import torch

    model.eval()
    config = model.config
    device = next(model.parameters()).device
    evidence, slot, kind = context_tensors(ctx)
    evidence = evidence.unsqueeze(0).to(device)
    slot = slot.unsqueeze(0).to(device)
    kinds = kind.unsqueeze(0).to(device)

    ids = [tokenizer.BOS_ID]
    confs: list[float] = []
    budget = min(config.max_target_tokens, max_actions * 5 + 1)
    with torch.no_grad():
        for _ in range(budget):
            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.trunk_hidden(evidence, slot, kinds, token_ids)
            last = hidden[:, -1:, :]
            next_id = int(model.token_logits(last, 0).argmax(dim=-1))
            # the confidence for THIS token was computed one step earlier (offset 2);
            # for the very first generated token there is no earlier read
            if len(ids) == 1:
                confs.append(1.0)
            else:
                prev = hidden[:, -2:-1, :]
                confs.append(float(model.confidence(prev, 1)))
            if next_id == tokenizer.EOS_ID:
                break
            ids.append(next_id)
        rationale = None
        if ctx.slot_kind == "arm3":
            token_ids = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.trunk_hidden(evidence, slot, kinds, token_ids)
            logits = model.rationale_logits(hidden)
            rationale = GOALS[int(logits.argmax(dim=-1))]

    return _finalize_chunk(ids, confs, rationale, max_actions)


def _finalize_chunk(ids: list[int], token_confs: list[float], rationale: str | None,
                    max_actions: int) -> tuple[ProposalChunk | None, str | None]:
    """Generated tokens -> (chunk, None), (None, NOTHING_TO_DO), or (None, reason).

    An immediate close (<eos> before any action token) is the decoder saying the
    build needs nothing — a real answer, returned under its own name. The token
    budget is a control-side cap, not a grammar rule: generation often stops
    mid-action (the corpus trains longer traces than one chunk holds), so the
    stream is trimmed to its longest complete-action prefix — the gate consumes
    prefixes anyway. Only a stream with no decodable action at all is garbage.
    """
    body = ids[1:]                       # drop <bos>
    if not body:
        return None, NOTHING_TO_DO
    actions, reason = None, None
    while body:
        actions, reason = tokenizer.decode_chunk(
            [tokenizer.BOS_ID] + body + [tokenizer.EOS_ID])
        if actions is not None:
            break
        body = body[:-1]
    if actions is None:
        return None, reason
    actions = actions[:max_actions]
    action_confs = _per_action_confidence(actions, token_confs)
    return ProposalChunk(actions=tuple(actions), token_conf=tuple(action_confs),
                         rationale_goal=rationale), None


def _per_action_confidence(actions: list[Action], token_confs: list[float]) -> list[float]:
    """Fold per-token confidences into per-action ones: min over each action's own
    tokens, first action pinned to 1.0 (the D4 conf_1 convention)."""
    per_action = []
    at = 0
    for index, action in enumerate(actions):
        width = len(tokenizer.encode_action(action))
        window = token_confs[at:at + width] or [0.0]
        per_action.append(1.0 if index == 0 else max(0.0, min(1.0, min(window))))
        at += width
    return per_action
