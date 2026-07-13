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
each subtype names a style; the best-fitting template instance in registration names
the style being observed. No new latent. A leaf-level belief (subtypes as filter
states, categories as marginals) is the recorded upgrade path — it changes the kernel
constants, so it waits for its own math review.

v3 (2026-07-10, the user's own list — design note "Taxonomy v3 - Subtype Expansion"):
the subtype layer widens from the 15 template-backed styles to 25 subcategories, five
per category, with written definitions and boundary rules in the vault note. Not every
subtype has a template: templates stay placement-only and <= 16^3, so break-dominant
styles (mining, moats, underground dwellings) are labelable but not matcher-confirmable.
Template instances now carry their subtype separately from their instance name
(templates.py), so several instances can realize one subtype (cabin and longhouse are
both ways of building a "house"). Existing labels migrate by the vault note's table
(scripts/migrate_labels_v3.py).

Division of labor across the streams, on purpose:
  - s_goal (MineCLIP) stays CATEGORY-level — 5 scores, each pooled over its subtypes'
    prompt phrases (PHRASES below turns snake_case names into readable prompt text).
  - Subtype evidence comes from D2 (the winning template instance) and from state_feats
    (a hand full of fences vs logs is a style cue the pixel models never see crisply).

The taxonomy is contract data: changing it changes what every artifact means, so it is
versioned, and additions bump the version rather than editing silently.
"""
from __future__ import annotations

TAXONOMY_VERSION = "3"   # v2 was the 15 template-backed styles; v1 the flat five types

# category -> its style subtypes (definitions + boundaries: the Taxonomy v3 vault note).
TAXONOMY: dict[str, tuple[str, ...]] = {
    "habitation": ("starter_shelter", "house", "base", "settlement",
                   "underground_dwelling"),
    "infrastructure": ("path_road", "bridge", "portal_hub", "rail_transit",
                       "vertical_transit"),
    "production": ("crop_farm", "mining_excavation", "animal_husbandry", "mob_farm",
                   "smelting_processing"),
    "defense": ("perimeter_wall", "gatehouse", "watchtower", "mob_trap", "moat"),
    "decorative": ("landscaping_garden", "interior_decor", "terraforming_aesthetic",
                   "pixel_art", "fountain_water_feature"),
}

# Readable prompt text per subtype — snake_case names make bad CLIP prompts
# ("a smelting_processing"). Prompt builders substitute these phrases.
PHRASES: dict[str, str] = {
    "starter_shelter": "starter shelter",
    "house": "house",
    "base": "home base with storage and crafting",
    "settlement": "village of several buildings",
    "underground_dwelling": "underground home dug into the ground",
    "path_road": "road",
    "bridge": "bridge",
    "portal_hub": "nether portal",
    "rail_transit": "railway with rails",
    "vertical_transit": "staircase connecting heights",
    "crop_farm": "crop farm",
    "mining_excavation": "mine shaft",
    "animal_husbandry": "animal pen",
    "mob_farm": "mob farm with drop chute",
    "smelting_processing": "furnace room",
    "perimeter_wall": "defensive wall",
    "gatehouse": "fortified gatehouse entrance",
    "watchtower": "watchtower",
    "mob_trap": "defensive mob trap",
    "moat": "moat ditch around a base",
    "landscaping_garden": "flower garden",
    "interior_decor": "furnished room interior",
    "terraforming_aesthetic": "sculpted landscape",
    "pixel_art": "pixel art mural",
    "fountain_water_feature": "fountain",
}

GOALS: tuple[str, ...] = tuple(TAXONOMY)          # the five categories — the filter's G
LEAVES: tuple[tuple[str, str], ...] = tuple(
    (category, subtype) for category, subtypes in TAXONOMY.items() for subtype in subtypes
)

_CATEGORY_OF = {subtype: category for category, subtype in LEAVES}


def category_of(subtype: str) -> str:
    """Which category a style belongs to — the coarse read of a fine label."""
    return _CATEGORY_OF[subtype]


assert set(PHRASES) == set(_CATEGORY_OF), "every subtype needs its prompt phrase"
