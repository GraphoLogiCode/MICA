"""One step of 2D behavior evidence: what the human is doing, read from a window of B0.

This is B1 (Evidence2D), the output of D1. The belief tracker scores goal hypotheses
against it, so it is **evidence, never intent**. Every scored record's window ends
strictly BEFORE the action it describes — the snapshot rule — so a later head can
never read the action off its own evidence (which would make the goal-likelihoods flat
and the filter inert). Unscored context records (~1 Hz inside long runs) carry drifting
state/attention between corrections; they score nothing and train nothing.

The two pixel channels, h2d (VPT) and s_goal (MineCLIP), stay None until those frozen
models are wired in (Layer 2). The symbolic channels below carry the signal until then.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .b0 import BlockPos
# The goal taxonomy lives in contracts/goals.py: GOALS is its five top-level categories —
# what s_goal scores against and what the belief's first proof target runs over. The
# style subtypes underneath belong to D2's template instances and to state_feats cues;
# they never widen the s_goal vector.
from .goals import GOALS   # noqa: F401  (re-exported: every consumer imports GOALS from here)


class MacroAction(Enum):
    """The few things a builder does, abstracted up from raw 20-per-second key/mouse input.

    The tracker reasons over these, not over single key presses — a goal is consistent with
    "placing a wall", not with "right-click at tick 412".
    """

    PLACE = "place"
    BREAK = "break"
    NAVIGATE = "navigate"
    INSPECT = "inspect"      # camera sweep with no block change — looking around
    SCAFFOLD = "scaffold"    # temporary support (place-then-remove); v1 reports these as PLACE
    IDLE = "idle"


@dataclass(frozen=True)
class StateFeats:
    """Player state taken straight from the logs, around no pixel encoder.

    Held item and recent actions are the load-bearing parts: what the player can do right
    now, and what they were just doing. The deltas say how they moved and looked over the
    window.
    """

    held_item: str
    hotbar: tuple[str, ...]
    pos_delta: tuple[float, float, float]   # net movement across the window
    yaw_delta: float                        # net turn (left/right)
    pitch_delta: float                      # net look (up/down)
    recent_actions: tuple[str, ...]         # the macro-actions just before this one


@dataclass(frozen=True)
class Focus:
    """Where the player's attention rested — the crosshair's block, and for how long.

    A single targeted block is a 1x1x1 region; dwell_ticks separates a glance from a
    fixation. `block` is None when the player was aiming at open air.
    """

    block: BlockPos | None
    dwell_ticks: int


@dataclass(frozen=True)
class Evidence2D:
    """One B1 record: the evidence for a single macro-action, all of it pre-action.

    Field contract (this is what D3 consumes — see D0's B3 box):
      tick_range  (int, int)         evidence window [t0, t1], t0 <= t1. For a scored record the
                                     action starts at t1 + 1, so the window ends strictly before
                                     it (snapshot rule). For an unscored context record t1 is the
                                     context tick itself — nothing is scored, so nothing can leak.
      a_hat       MacroAction        the discrete action (element of A): the scored action on a
                                     correction, the ongoing run's label on a context record.
      a_hat_conf  float in [0, 1]    1.0 for PLACE/BREAK (event-backed) — and for IDLE, which is
                                     inferred, a v1 choice pending the idle-precision measurement
                                     (ISSUES B1-F7); 0.9/0.7 for the inferred motion classes.
      idle        bool               True iff a_hat is IDLE — the FSM's safe-assist signal.
      state_feats StateFeats         RAW structured state. D3's heads read it raw (lossless held
                                     item, per D0 B3); D3's adapter also embeds it to float[ds].
      focus       Focus              the crosshair's target block (a 1x1x1 region) and dwell.
      scored      bool               True at a macro-action boundary (a correction step); False
                                     for the ~1 Hz periodic context records emitted inside long
                                     runs. Context is read by the tracker and gate between
                                     corrections — never scored, never trained on.
      event_ids   tuple[int]         the block events this action consumed — non-empty for scored
                                     PLACE/BREAK only, always empty on context records. Each
                                     event id appears in exactly one record.
      h2d         tuple[float]|None  VPT behavior embedding, length d2 (1024 for vpt-1x). The
                                     adapter input. None when the window held no saved frames.
      s_goal      tuple[float]|None  per-goal MineCLIP similarity, length |G| = len(GOALS); each a
                                     cosine in [-1, 1]. None when no frames. h2d and s_goal are
                                     always filled together or None together (one frame source).
    """

    tick_range: tuple[int, int]
    a_hat: MacroAction
    a_hat_conf: float
    idle: bool
    state_feats: StateFeats
    focus: Focus
    scored: bool
    event_ids: tuple[int, ...]
    h2d: tuple[float, ...] | None = None
    s_goal: tuple[float, ...] | None = None
