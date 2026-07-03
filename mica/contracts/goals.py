"""The goal taxonomy: five functional categories, each realized by named style subtypes.

WHY A HIERARCHY (decided 2026-07-02, prompted by a real session): a builder's treehouse
was outside the old flat goal set, yet its purpose — somewhere to live — was never in
doubt. Intent lives at the category level ("making a place to live"); the style ("as a
treehouse") is how that intent is being realized. A flat set of narrow build classes
forces early commitment to exactly the stylistic variation real builders produce;
categories let the belief firm up early while the style stays open — coarse-to-fine.

WHY THE MATH IS UNTOUCHED: the belief b(g, z) runs over the FIVE CATEGORIES, so every
reviewed theorem (finite G, kernels, floor, geometric forgetting) and every |G| = 5
contract (s_goal length, the B1 box) is unchanged. The subtype layer is evidence-side:
each subtype is one template instance, and the best-fitting instance in registration
names the style. No new latent. A leaf-level belief (subtypes as filter states,
categories as marginals) is the recorded upgrade path — it changes the kernel constants,
so it waits for its own math review.

Division of labor across the streams, on purpose:
  - s_goal (MineCLIP) stays CATEGORY-level — 5 scores, each pooled over its subtypes'
    prompts. Video-text similarity cannot tell a cabin from a longhouse anyway.
  - Subtype evidence comes from D2 (the winning template instance) and from state_feats
    (a hand full of fences vs logs is a style cue the pixel models never see crisply).

TAXONOMY lists the template-backed subtypes; templates.py carries one instance per name
and a test asserts the coupling. The full conceptual reach of each category — the
roadmap for future template versions — is: habitation (houses, cabins, treehouses,
tower homes, underground bases); infrastructure (bridges, roads, paths, stairs, docks,
entrances, rail hubs); production (crop farms, animal pens, storage barns, trading
halls, smelters, redstone utilities); defense (watchtowers, walls, gatehouses, guard
posts, fortified bases); decorative/civic (gardens, statues, plazas, fountains,
gazebos, parks, monuments).

The taxonomy is contract data: changing it changes what every artifact means, so it is
versioned, and additions bump the version rather than editing silently.
"""
from __future__ import annotations

TAXONOMY_VERSION = "2"   # v1 was the flat five build types promoted to categories 1:1

# category -> its style subtypes. What makes each category recognizable to the 3D stream:
# habitation is enclosed and roofed; infrastructure is elongated, open, and low; production
# is flat ground work or fenced ground; defense is thin and vertical; decorative is small,
# ornamental, and symmetric.
TAXONOMY: dict[str, tuple[str, ...]] = {
    "habitation": ("cabin", "longhouse", "treehouse"),
    "infrastructure": ("railed bridge", "flat span", "road"),
    "production": ("crop field", "bordered plot", "fence pen", "post-and-rail pen"),
    "defense": ("square tower", "pillar tower", "perimeter wall"),
    "decorative": ("flower garden", "fountain"),
}

GOALS: tuple[str, ...] = tuple(TAXONOMY)          # the five categories — the filter's G
LEAVES: tuple[tuple[str, str], ...] = tuple(
    (category, subtype) for category, subtypes in TAXONOMY.items() for subtype in subtypes
)

_CATEGORY_OF = {subtype: category for category, subtype in LEAVES}


def category_of(subtype: str) -> str:
    """Which category a style belongs to — the coarse read of a fine label."""
    return _CATEGORY_OF[subtype]
