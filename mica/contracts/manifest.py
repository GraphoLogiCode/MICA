"""A result only makes sense if you know the exact setup that produced it, so we record that here."""
from __future__ import annotations

from dataclasses import dataclass

from .b0 import ObservationPacket
from .timebase import TICKS_PER_SECOND


@dataclass(frozen=True)
class SessionManifest:
    """The setup for one B0 recording: version numbers and when its clock started.

    B0-only provenance. The VPT/MineCLIP checkpoint hashes are D1's, not B0's — D1 records them in a
    `<session>.d1_provenance.json` sidecar when it runs (scripts/run_d1.py), so they don't belong here.
    """

    session_id: str
    session_start_ms: int
    ticks_per_second: int = TICKS_PER_SECOND
    mc_version: str = "unknown"
    mod_version: str = "unknown"
    event_schema_version: str = "1"   # the B0 block-event schema this recording was written with
    # The frame settings the recording ran with (0 = an older recording that didn't say).
    # Width 256+ and the mod's 160 height floor let both B1 models shrink, never enlarge.
    frame_every: int = 0
    frame_width_px: int = 0
    # Region snapshots (D2's ground truth). quiet_ticks > 0 means this recording takes
    # them (one at start, one after each building burst, at a quiet moment); 0 = an older
    # recording that can't have any. The region is the fixed box every snapshot frames,
    # as (x0, y0, z0, x1, y1, z1) inclusive — empty until the first in-world tick fixed it.
    snapshot_quiet_ticks: int = 0
    snapshot_region: tuple[int, ...] = ()


@dataclass(frozen=True)
class CapturedSession:
    """One whole recording: its setup, every captured moment, and how many block changes it should contain.

    The recorder states the count separately so we can tell if any block change
    went missing — a lost change makes the captured changes fall short of the count.

    A recording whose count was never finalized (the game crashed before the clean
    stop that writes the real total) is marked provisional. It can still be loaded
    and inspected, but the missing-change check means nothing for it — the end of
    the recording may be gone without any way to tell — so the gate rejects it.
    """

    manifest: SessionManifest
    packets: tuple[ObservationPacket, ...]
    declared_event_count: int
    declared_is_provisional: bool = False
