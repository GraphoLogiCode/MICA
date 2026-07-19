"""R-6 (2026-07-18 rig review): a voiced suggestion must name WHAT and WHERE.
proposal_summary was gated on held_k >= 1, but SUGGEST is by construction the
K_commit == 0 state — so every voiced suggestion in the first real advisory
session said the generic fallback and logged summary: null."""
from dataclasses import dataclass, field

from mica.decoder.grammar import Place, Say
from mica.gate.live_loop import proposal_summary


@dataclass
class _Proposal:
    actions: list = field(default_factory=list)


def test_a_k0_placement_proposal_names_block_and_cell():
    # the 2026-07-18 voiced row: stone_bricks at (282, 70, 188), K_commit 0
    proposal = _Proposal([Place(0, 0, 0, "minecraft:stone_bricks")])
    summary = proposal_summary(proposal, (282, 70, 188), 0, "production")
    assert summary == "a stone_bricks block at (282, 70, 188)"


def test_a_committed_prefix_adds_the_commit_context():
    proposal = _Proposal([Place(0, 0, 0, "oak_planks")])
    summary = proposal_summary(proposal, (1, 2, 3), 3, "habitation")
    assert summary == "a oak_planks block at (1, 2, 3) (3 action(s) committed toward habitation)"


def test_non_placement_proposals_keep_the_old_commit_line():
    proposal = _Proposal([Say("shall I help?")])
    assert proposal_summary(proposal, None, 2, "defense") == \
        "2 action(s) toward defense, first at None"
    # ...and stay silent below the old threshold, as before
    assert proposal_summary(proposal, None, 0, "defense") is None


def test_no_proposal_is_no_summary():
    assert proposal_summary(None, (1, 2, 3), 5, "production") is None
