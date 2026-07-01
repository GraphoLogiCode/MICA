"""The game runs in fixed steps called ticks, 20 per second.

That tick count is our official clock. The player's own clock can drift, so we
always check it against the server's clock instead of trusting it.
"""
from __future__ import annotations

TICKS_PER_SECOND = 20
MS_PER_TICK = 1000 // TICKS_PER_SECOND  # 50 ms

# The player's clock may be off by at most one step (50 ms); more than that and we don't trust it.
DEFAULT_ALIGNMENT_BOUND_MS = MS_PER_TICK


def alignment_residual_ms(server_wallclock_ms: float, capture_wallclock_ms: float) -> float:
    """How far the player's clock is from the server's clock for the same moment.

    We compare the two real clocks, not an ideal schedule, so that a slow server
    (which makes both clocks late together) doesn't look like a sync problem.
    """
    return capture_wallclock_ms - server_wallclock_ms


def ends_before_action(window_end_tick: int, action_tick: int) -> bool:
    """True if we finished gathering clues before the action happened.

    We have to guess an action from earlier clues. If a clue already shows the
    action, we'd just be copying the answer instead of predicting it.

    Currently unused: D1 enforces this inline (it builds windows that end the tick
    before an action) and sync_report checks each action has a prior tick. Kept as
    the contract's snapshot-rule helper. (ISSUES.md C-6.)
    """
    return window_end_tick < action_tick
