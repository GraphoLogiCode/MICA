"""The materials constraint (D5 §4, added 2026-07-06): the agent builds only from
what it is actually given.

The agent's own in-game inventory is the hard constraint — what MICA_AI holds is
what it may place. The inventory travels mind-ward in the agent status file the
embodiment already writes; the gate reads the newest snapshot once per read. A
proposed prefix stays committable only up to the first placement whose block ran
out (stock depletes as the prefix consumes it), so a shortage SHRINKS the build
instead of substituting behind the human's back. Substitution happens only through
an explicit chat grant ("yes" to the agent's one question), which arrives back
here in the same status file.

When no inventory snapshot exists — counterfactual runs, replays, agent absent —
the constraint is inactive by design: every banked behavior stays bit-unchanged.
"""
from __future__ import annotations

import glob
import json
import os
import time

from ..decoder.grammar import Action, Place

# A snapshot older than this means the agent is gone (crashed, world closed) —
# gating on its stale inventory would be acting on a body that no longer exists.
_SNAPSHOT_FRESH_S = 10.0
_TAIL_BYTES = 1 << 14           # the status file's last lines live in the final 16 KB


def normalize(block: str) -> str:
    """One name for both worlds: the grammar says 'minecraft:oak_planks', the
    bot's inventory says 'oak_planks'."""
    return block.split(":")[-1]


def feasibility_flags(actions, inventory: dict, grants: dict) -> tuple[list[bool], dict, dict]:
    """Per-action infeasibility flags for a chunk, in order: True = CANNOT PLACE.

    Mirrors reversibility.prefix_flags (True = the problem case). Stock depletes
    as the prefix consumes it, so the same block twice needs two in stock. A
    granted substitute (block -> substitute, both bare names) is drawn on only
    when the real block is out. Returns (flags, missing, substituted):
    missing[block] = how many placements had nothing to draw on;
    substituted[block] = how many drew on the granted substitute instead.
    """
    stock = {normalize(name): count for name, count in (inventory or {}).items()}
    flags: list[bool] = []
    missing: dict[str, int] = {}
    substituted: dict[str, int] = {}
    for action in actions:
        if not isinstance(action, Place):
            flags.append(False)         # move / look / say consume nothing
            continue
        block = normalize(action.block)
        substitute = grants.get(block)
        if stock.get(block, 0) > 0:
            stock[block] -= 1
            flags.append(False)
        elif substitute and stock.get(substitute, 0) > 0:
            stock[substitute] -= 1
            substituted[block] = substituted.get(block, 0) + 1
            flags.append(False)
        else:
            missing[block] = missing.get(block, 0) + 1
            flags.append(True)
    return flags, missing, substituted


def propose_substitute(block: str, inventory: dict, grants: dict,
                       denied: tuple[str, ...] = ()) -> str | None:
    """A same-family stand-in the agent could ASK about: shares the block's last
    name part (oak_planks -> spruce_planks, both 'planks'), is in stock, isn't
    the block itself, isn't already granted, and wasn't already refused."""
    block = normalize(block)
    family = block.rsplit("_", 1)[-1]
    stock = {normalize(name): count for name, count in (inventory or {}).items()}
    for candidate in sorted(stock):
        if (candidate != block and stock[candidate] > 0
                and candidate.rsplit("_", 1)[-1] == family
                and candidate not in denied and grants.get(block) != candidate):
            return candidate
    return None


def read_agent_snapshot(directory: str) -> tuple | None:
    """The newest agent status line's (inventory, material_grants, agent position,
    consent) — or None when no agent is writing (no file, unreadable, or stale).
    None means the materials constraint stays inactive, which is exactly right for
    replays and tests. The position rides along (D5 §10 Q1 live half) so one file
    read per gate read serves both the materials constraint and the agent-proximity
    veto; consent (D5 §10, 2026-07-19) is the human's relayed "yes" to a voiced
    suggestion — {ts, cell} or None. Consumers index with length guards, so older
    test doubles handing back shorter tuples keep working."""
    candidates = glob.glob(os.path.join(directory, "agent-*.status.jsonl"))
    newest, newest_mtime = None, -1.0
    for path in candidates:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime
    if newest is None or time.time() - newest_mtime > _SNAPSHOT_FRESH_S:
        return None
    try:
        with open(newest, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - _TAIL_BYTES))
            lines = handle.read().decode("utf-8", errors="replace").strip().splitlines()
        for line in reversed(lines):
            row = json.loads(line)
            if "inventory" in row:
                pos = tuple(row["pos"]) if row.get("pos") else None
                return (row.get("inventory") or {}, row.get("material_grants") or {},
                        pos, row.get("consent"))
        return None                      # an agent from before the inventory field
    except (OSError, ValueError):
        return None


class MaterialsAccount:
    """The session's running material ledger — what the report is written from."""

    def __init__(self):
        self.proposed: dict[str, int] = {}       # every block the decoder asked for
        self.missing: dict[str, int] = {}        # placements with nothing to draw on
        self.substituted: dict[str, int] = {}    # placements served by a grant
        self.capped_reads = 0                    # reads whose prefix was shortened
        self.last_inventory: dict | None = None
        self.grants_seen: dict[str, str] = {}

    def observe(self, actions, missing: dict, substituted: dict,
                capped: bool, inventory: dict, grants: dict) -> None:
        for action in actions:
            if isinstance(action, Place):
                block = normalize(action.block)
                self.proposed[block] = self.proposed.get(block, 0) + 1
        for block, count in missing.items():
            self.missing[block] = max(self.missing.get(block, 0), count)
        for block, count in substituted.items():
            self.substituted[block] = max(self.substituted.get(block, 0), count)
        self.capped_reads += 1 if capped else 0
        self.last_inventory = dict(inventory or {})
        self.grants_seen.update(grants or {})

    def report(self, committed_places: int) -> dict:
        """The D5 materials-report contract: usage, remaining, missing, compromises.
        `missing` is per-read maxima, not sums — the same shortfall re-read at 1 Hz
        is one shortage, not sixty."""
        return {
            "proposed_blocks": dict(sorted(self.proposed.items())),
            "committed_places": committed_places,      # 0 under the demo config
            "inventory_remaining": self.last_inventory,
            "missing": dict(sorted(self.missing.items())),
            "compromises": {
                "substitution_grants_used": dict(sorted(self.substituted.items())),
                "grants": dict(sorted(self.grants_seen.items())),
                "reads_with_shortened_prefix": self.capped_reads,
            },
        }
