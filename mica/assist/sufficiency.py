"""Material sufficiency: can MICA realistically help build the target, and if not,
what is missing and what should it gather (D7 §4).

Pure functions over plain data — the report script and the live gather behavior both
call these; neither re-derives material math. Reuses the gate's materials primitives
(normalize) and the template set: a target's needs come straight from its template's
cells, so "enough materials" always means the same thing the matcher's shapes mean.

Honest approximations, stated once:
  * A subtype's needs are read from its SMALLEST template instance — the least
    material a recognizable build of that style requires. A grander build needs more;
    this is the floor, not a promise.
  * REQ_SOLID cells accept any placeable block, so they draw on pooled "block-ish"
    stock. What counts as placeable is a suffix heuristic (tools, weapons, buckets
    and food are excluded); item ids alone cannot say for sure without the game's
    registry, and this module never has the game.
  * Definitions-only subtypes (mining, moats, ...) have no template, so sufficiency
    for them is honestly None — "cannot say", never a guess.
"""
from __future__ import annotations

from ..gate.materials import normalize
from ..perception.templates import REQ_SOLID, TEMPLATES

# One errand's hard cap (D5 amendment 2026-07-10, guardrail 3): one stack.
GATHER_CAP = 64

# What the agent may ever mine on its own (guardrail 1): natural, abundant blocks.
GATHER_WHITELIST = ("oak_log", "spruce_log", "birch_log", "stone", "cobblestone",
                    "dirt", "sand", "gravel", "poppy", "dandelion")

# Crafted blocks the templates pin, mapped to the natural resource they come from —
# the agent gathers RAW material; crafting stays the human's (v1 keeps it simple).
_RAW_SOURCE = {
    "oak_fence": "oak_log", "oak_slab": "oak_log", "oak_planks": "oak_log",
    "oak_log": "oak_log", "poppy": "poppy", "cobblestone": "cobblestone",
    "stone": "stone", "dirt": "dirt", "sand": "sand",
}

# Items that are never building material — the placeable-stock heuristic.
_NON_PLACEABLE_SUFFIXES = ("sword", "pickaxe", "axe", "shovel", "hoe", "bucket",
                           "bow", "arrow", "shield", "helmet", "chestplate",
                           "leggings", "boots", "apple", "bread", "beef", "porkchop",
                           "seeds", "stick", "flint")


def _placeable(item: str) -> bool:
    name = normalize(item)
    return not any(name.endswith(suffix) for suffix in _NON_PLACEABLE_SUFFIXES)


def template_requirements(subtype: str) -> dict[str, int] | None:
    """Block needs for one v3 subtype, from its smallest template instance:
    {exact block -> count, REQ_SOLID -> any-placeable-block budget}. None when the
    subtype is definitions-only (no template can say what it needs)."""
    instances = [template for templates in TEMPLATES.values() for template in templates
                 if template.subtype == subtype]
    if not instances:
        return None
    smallest = min(instances, key=lambda template: len(template.cells))
    needs: dict[str, int] = {}
    for required in smallest.cells.values():
        key = required if required == REQ_SOLID else normalize(required)
        needs[key] = needs.get(key, 0) + 1
    return needs


def _stock(inventory) -> dict[str, int]:
    """(item, count) pairs or a dict -> bare-name counts. None -> empty."""
    pairs = inventory.items() if isinstance(inventory, dict) else (inventory or ())
    stock: dict[str, int] = {}
    for item, count in pairs:
        stock[normalize(item)] = stock.get(normalize(item), 0) + int(count)
    return stock


def sufficiency(requirements: dict[str, int] | None,
                player_inventory=None, agent_inventory=None) -> dict | None:
    """Have vs need for one target, over BOTH inventories pooled (the build is
    collaborative; what the pair holds together is what can go into it), plus the
    agent-only view that answers "can MICA help with its own stock".

    Returns {needed, have, missing, can_help, agent_share, gather_next} — or None
    when requirements is None (definitions-only subtype: cannot say)."""
    if requirements is None:
        return None
    player, agent = _stock(player_inventory), _stock(agent_inventory)
    pooled = dict(player)
    for item, count in agent.items():
        pooled[item] = pooled.get(item, 0) + count

    have: dict[str, int] = {}
    missing: dict[str, int] = {}
    remaining = dict(pooled)
    # Exact-block needs draw first (only that block satisfies them)...
    for block, needed in requirements.items():
        if block == REQ_SOLID:
            continue
        got = min(remaining.get(block, 0), needed)
        have[block] = got
        remaining[block] = remaining.get(block, 0) - got
        if got < needed:
            missing[block] = needed - got
    # ...then the any-solid budget draws on whatever placeable stock is left.
    solid_needed = requirements.get(REQ_SOLID, 0)
    if solid_needed:
        solid_stock = sum(count for item, count in remaining.items()
                          if count > 0 and _placeable(item))
        got = min(solid_stock, solid_needed)
        have[REQ_SOLID] = got
        if got < solid_needed:
            missing[REQ_SOLID] = solid_needed - got

    total_needed = sum(requirements.values())
    agent_placeable = sum(count for item, count in agent.items() if _placeable(item))
    return {
        "needed": dict(requirements),
        "have": have,
        "missing": missing,
        # The agent can help when its OWN stock covers at least one needed block —
        # and agent_share says how much of the whole build it could carry alone.
        "can_help": agent_placeable > 0 and total_needed > 0,
        "agent_share": round(min(agent_placeable / total_needed, 1.0), 3) if total_needed else 0.0,
        "gather_next": gather_plan(missing),
    }


def gather_plan(missing: dict[str, int]) -> list[dict]:
    """The missing blocks as gather errands, biggest shortfall first: whitelisted
    raw resources only, one stack apiece (the D5 amendment's guardrails). A missing
    block with no gatherable source is reported as such, never silently dropped."""
    errands = []
    for block, count in sorted(missing.items(), key=lambda kv: -kv[1]):
        if block == REQ_SOLID:
            source = "cobblestone"           # the default bulk block: mineable anywhere
        else:
            source = _RAW_SOURCE.get(normalize(block))
        if source in GATHER_WHITELIST:
            errands.append({"gather": source, "count": min(count, GATHER_CAP),
                            "for": block})
        else:
            errands.append({"gather": None, "count": count, "for": block,
                            "note": "no gatherable source — needs the human"})
    return errands
