"""The static reversibility table (D5 §4): can a proposed action be undone cheaply?

Each proposed action is classified ONCE, by this table, before the commit gate may
count it. The classes and their gate treatment come straight from the D5 note:

  reversible     placing a common solid block on empty space; the agent's own
                 movement, gaze, and chat — eligible for PLACE_LOW_RISK
  irreversible   breaking a HUMAN-placed block; placing lava, water, fire, or TNT;
                 placing gravity blocks (sand, gravel) that can fall somewhere else;
                 anything that destroys, buries, or floods human work — never
                 PLACE_LOW_RISK, at most PREVIEW

Two v1 notes, stated instead of hidden. First: "on empty space" is a live-world
check; offline, the decoder corpus's hygiene rule already guarantees proposed cells
are empty at proposal time, and the live embodiment re-checks before acting (part 2).
Second: breaking the human's own misplaced block is still classified irreversible —
the table protects human work even when the helper trace says the block was a
mistake; cleaning up a human mistake is at most a PREVIEW/SUGGEST, never a silent
break.
"""
from __future__ import annotations

from ..decoder.grammar import Action, Break, Place

# Block types whose placement cannot be cleanly undone: they flow, burn, explode,
# or fall. Matched by substring on the block id so waterlogged variants count too.
RISKY_BLOCK_WORDS = ("lava", "water", "fire", "tnt", "sand", "gravel")


def is_reversible(action: Action, human_cells: frozenset[tuple[int, int, int]]) -> bool:
    """One action against the table. human_cells holds the HUMAN-placed cells, in
    the same origin-relative frame the action's coordinates use."""
    if isinstance(action, Place):
        return not any(word in action.block for word in RISKY_BLOCK_WORDS)
    if isinstance(action, Break):
        return (action.dx, action.dy, action.dz) not in human_cells
    return True   # move / look / say never change the world


def prefix_flags(actions, human_cells: frozenset[tuple[int, int, int]]) -> list[bool]:
    """Per-action irreversibility flags for a chunk, in order: True = IRREVERSIBLE.
    (The staircase's theta(j, irrev) wants the risk flag, not the safety flag.)"""
    return [not is_reversible(action, human_cells) for action in actions]
