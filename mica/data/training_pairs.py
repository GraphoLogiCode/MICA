"""Training samples for the Phase-E adapter and likelihood heads.

The heads learn one thing: given the evidence available strictly BEFORE an action,
predict which action the human took. The samples come from the two licensed sources:

  Source A — the scripted corpus. The generator wrote each build, so every sample
             carries an exact goal label AND an exact reasoning-mode label (deliberate
             vs shortcut). This is the only source that may train the heuristic head,
             because hindsight labels carry no mode ground truth.
  Source B — real captures labeled in hindsight from their finished structure
             (<session>.source_b.jsonl, written by scripts/label_finished_builds.py
             under the contest rule). Deliberative supervision only.

Every sample is re-verified here before it may train anything (09-F1, the project's
named most-dangerous silent failure: heads trained on evidence that already contains
the action learn to READ the action instead of predicting it, and the filter goes
inert while every normalization test passes):

  1. the pair joins — fuse_dicts refuses a tick or event-id mismatch;
  2. the window ends strictly before the action tick (the snapshot rule, asserted
     explicitly even though the join implies it);
  3. the leak detector: the evidence's built-cell count can never exceed the number
     of block events consumed by EARLIER records — if a record's own placement leaked
     into its evidence, the very first placement of a session would show a built cell
     before any event had been consumed.

A violation raises SnapshotViolation and the sample never enters training.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ..contracts.b3 import FusedEvidence, fuse_dicts

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPTED = os.path.join(_ROOT, "capture", "scripted")
_RAW = os.path.join(_ROOT, "capture", "raw")

# The pinned validation split: the LAST scripted session per goal (sorted session ids,
# deterministic) plus this one pairs-eligible real capture — the cleanest template
# session (fit 1.00 comp 1.00), chosen because it costs training the fewest pairs.
VALIDATION_REAL = "fabric-20260705-134615"


class SnapshotViolation(AssertionError):
    """A training pair whose evidence is not strictly pre-action."""


@dataclass(frozen=True)
class Sample:
    """One (pre-action evidence -> next action) training example."""

    session: str
    source: str               # "A" (scripted, mode known) or "B" (real, hindsight label)
    goal: str                 # category label the pair trains toward
    mode: int | None          # 0 deliberate / 1 shortcut (Source A only; None for B)
    progress: float           # fraction of the build placed BEFORE this action
    fused: FusedEvidence      # the verified evidence bundle; fused.a_hat is the target


def _verify_and_fuse(pair: dict, consumed_before: int) -> FusedEvidence:
    """The per-sample 09-F1 assertions. Returns the fused evidence or raises."""
    b1, b2 = pair["b1"], pair["b2"]
    fused = fuse_dicts(b1, b2)                       # join check; raises on mismatch
    if b1["tick_range"][1] + 1 != b2["tick"]:
        raise SnapshotViolation(
            f"{pair['session']}: window ends at {b1['tick_range'][1]} but the action "
            f"tick is {b2['tick']} — evidence is not strictly pre-action")
    built = b2["global"]["built_count"]
    if built > consumed_before:
        raise SnapshotViolation(
            f"{pair['session']}: evidence shows {built} built cells but only "
            f"{consumed_before} events were consumed before this record — the action's "
            "own effects leaked into its evidence")
    return fused


def _load_session_pairs(path: str, source: str, mode: int | None) -> list[Sample]:
    samples = []
    consumed_before = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        pair = json.loads(line)
        fused = _verify_and_fuse(pair, consumed_before)
        consumed_before += len(pair["b1"]["event_ids"])
        samples.append(Sample(session=pair["session"], source=source,
                              goal=pair["label"]["goal"], mode=mode,
                              progress=pair["progress"], fused=fused))
    return samples


def load_source_a() -> list[Sample]:
    """Every scripted session's pairs, with the generator's mode label attached.
    The matcher's labels on this corpus measured 1.0 accurate (source_b_report.json),
    so the pair label and the generator truth coincide."""
    with open(os.path.join(_SCRIPTED, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    samples = []
    for session_id, label in sorted(labels.items()):
        path = os.path.join(_SCRIPTED, f"{session_id}.source_b.jsonl")
        if not os.path.exists(path):
            continue
        mode = 1 if label["mode"] == "shortcut" else 0
        samples.extend(_load_session_pairs(path, "A", mode))
    return samples


def load_source_b() -> list[Sample]:
    """Every real capture's pairs. Only sessions the labeler KEPT with an uncontested
    label have a source_b file at all, so the contest rule is already enforced.
    Recursive so the pairs are found whether the raw tree is flat (legacy) or
    organized by date/session (the session_store layout)."""
    import glob

    samples = []
    for path in sorted(glob.glob(os.path.join(_RAW, "**", "*.source_b.jsonl"),
                                 recursive=True)):
        samples.extend(_load_session_pairs(path, "B", None))
    return samples


def split(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """(train, validation) under the pinned rule. Validation holds out whole sessions —
    never records — so no session's evidence appears on both sides."""
    by_goal: dict[str, set[str]] = {}
    for s in samples:
        if s.source == "A":
            by_goal.setdefault(s.goal, set()).add(s.session)
    holdout = {sorted(sessions)[-1] for sessions in by_goal.values()}
    holdout.add(VALIDATION_REAL)
    train = [s for s in samples if s.session not in holdout]
    validation = [s for s in samples if s.session in holdout]
    return train, validation


def held_item_vocab(samples: list[Sample]) -> tuple[str, ...]:
    """The held-item vocabulary the state encoder embeds, built from training data
    only. Index 0 is reserved for unknown items at inference time."""
    return ("<unk>",) + tuple(sorted({s.fused.state_feats.held_item for s in samples}))
