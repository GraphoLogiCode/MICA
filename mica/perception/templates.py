"""The hand-authored template instances — one buildable shape per taxonomy subtype.

A template is the structure channel's analog of a model weight: it decides what a build
"looks like progress toward", so the set carries a version and every evidence artifact
records it. Each instance IS one subtype of the goal taxonomy (contracts/goals.py) —
the winning instance in registration is the style read, which is how coarse-to-fine
inference gets its fine layer without widening s_goal beyond the five categories.

Every instance fits the ~16x16x16 cap and is buildable purely by placing blocks (the
capture records placements; tilling and buckets are outside its scope, so no template
requires farmland or water — a tilled-ground field style waits on a HoeItem capture hook).

Cell requirements: a specific block id must match exactly; REQ_SOLID accepts any
non-air block — pre-existing terrain can satisfy it (the D2 ground rules: completion
reads the world, fit reads the player). Styles that terrain could fake (rings, flat
beds) pin exact blocks (fences, poppies) precisely so a landscape cannot be one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..contracts.goals import GOALS, TAXONOMY

TEMPLATE_SET_VERSION = "3"   # v3 (2026-07-02): the functional-category taxonomy, 15 subtypes
REQ_SOLID = "#solid"         # any non-air block satisfies this cell

_FENCE = "minecraft:oak_fence"
_LOG = "minecraft:oak_log"
_SLAB = "minecraft:oak_slab"
_FLOWER = "minecraft:poppy"


@dataclass(frozen=True)
class Template:
    """One style's shape: required blocks at offsets from its local origin."""

    name: str      # equals its taxonomy subtype
    goal: str      # the category it belongs to
    cells: Mapping[tuple[int, int, int], str]   # (dx, dy, dz) -> required block / REQ_SOLID


def rotate_offset(offset: tuple[int, int, int], rot: int) -> tuple[int, int, int]:
    """Turn an offset by a yaw rotation about the vertical axis (90-degree steps)."""
    dx, dy, dz = offset
    if rot == 0:
        return (dx, dy, dz)
    if rot == 90:
        return (-dz, dy, dx)
    if rot == 180:
        return (-dx, dy, -dz)
    return (dz, dy, -dx)   # 270


def footprint(cells: Mapping[tuple[int, int, int], str]) -> frozenset[tuple[int, int]]:
    """The template's ground-plane shadow — what the registration search slides around."""
    return frozenset((dx, dz) for dx, _, dz in cells)


def _rect_perimeter(size_x: int, size_z: int, y: int, required: str) -> dict:
    ring = {}
    for dx in range(size_x):
        ring[(dx, y, 0)] = required
        ring[(dx, y, size_z - 1)] = required
    for dz in range(size_z):
        ring[(0, y, dz)] = required
        ring[(size_x - 1, y, dz)] = required
    return ring


def _slab_layer(size_x: int, size_z: int, y: int, required: str) -> dict:
    return {(dx, y, dz): required for dx in range(size_x) for dz in range(size_z)}


# ---- habitation: enclosed and roofed --------------------------------------------------

def _cabin() -> Template:
    cells = {}
    cells.update(_rect_perimeter(5, 5, 0, REQ_SOLID))
    cells.update(_rect_perimeter(5, 5, 1, REQ_SOLID))
    cells.update(_slab_layer(5, 5, 2, REQ_SOLID))
    return Template(name="cabin", goal="habitation", cells=cells)


def _longhouse() -> Template:
    cells = {}
    cells.update(_rect_perimeter(7, 5, 0, REQ_SOLID))
    cells.update(_rect_perimeter(7, 5, 1, REQ_SOLID))
    cells.update(_slab_layer(7, 5, 2, REQ_SOLID))
    return Template(name="longhouse", goal="habitation", cells=cells)


def _treehouse() -> Template:
    # The style a real session taught us to expect: a trunk column carrying a platform
    # high up — habitation that grows upward before it grows outward.
    cells = {(2, y, 2): REQ_SOLID for y in range(5)}
    cells.update(_slab_layer(5, 5, 5, REQ_SOLID))
    return Template(name="treehouse", goal="habitation", cells=cells)


# ---- infrastructure: elongated, open, low ---------------------------------------------

def _railed_bridge() -> Template:
    cells = _slab_layer(7, 3, 0, REQ_SOLID)
    for dx in range(7):
        cells[(dx, 1, 0)] = REQ_SOLID
        cells[(dx, 1, 2)] = REQ_SOLID
    return Template(name="railed bridge", goal="infrastructure", cells=cells)


def _flat_span() -> Template:
    return Template(name="flat span", goal="infrastructure", cells=_slab_layer(9, 3, 0, REQ_SOLID))


def _road() -> Template:
    # Long and narrow, nothing above it — movement, not crossing.
    return Template(name="road", goal="infrastructure", cells=_slab_layer(11, 2, 0, REQ_SOLID))


# ---- production: flat ground work or fenced ground ------------------------------------

def _crop_field() -> Template:
    cells = _slab_layer(7, 7, 0, REQ_SOLID)
    del cells[(3, 0, 3)]   # the classic center water gap
    return Template(name="crop field", goal="production", cells=cells)


def _bordered_plot() -> Template:
    cells = _rect_perimeter(5, 5, 0, _LOG)
    cells.update({(dx, 0, dz): REQ_SOLID for dx in range(1, 4) for dz in range(1, 4)})
    return Template(name="bordered plot", goal="production", cells=cells)


def _fence_pen() -> Template:
    # Fences: the one ring a landscape can never fake.
    return Template(name="fence pen", goal="production", cells=_rect_perimeter(5, 5, 0, _FENCE))


def _post_and_rail_pen() -> Template:
    posts = {(dx, 0, dz): _LOG
             for dx, dz in ((0, 0), (0, 4), (4, 0), (4, 4), (2, 0), (2, 4), (0, 2), (4, 2))}
    rails = {(dx, 1, dz): _SLAB for (dx, _, dz) in _rect_perimeter(5, 5, 0, _SLAB)}
    posts.update(rails)
    return Template(name="post-and-rail pen", goal="production", cells=posts)


# ---- defense: thin and vertical --------------------------------------------------------

def _square_tower() -> Template:
    cells = {}
    for y in range(6):
        cells.update(_rect_perimeter(3, 3, y, REQ_SOLID))
    cells.update(_slab_layer(3, 3, 6, REQ_SOLID))
    return Template(name="square tower", goal="defense", cells=cells)


def _pillar_tower() -> Template:
    cells = {}
    for y in range(8):
        cells.update(_slab_layer(2, 2, y, REQ_SOLID))
    return Template(name="pillar tower", goal="defense", cells=cells)


def _perimeter_wall() -> Template:
    # A single straight run, three high, one thick — territory control, not shelter.
    cells = {(dx, y, 0): REQ_SOLID for dx in range(9) for y in range(3)}
    return Template(name="perimeter wall", goal="defense", cells=cells)


# ---- decorative/civic: small, ornamental, symmetric ------------------------------------

def _flower_garden() -> Template:
    # A checkerboard of planted flowers — exact blocks, so a meadow is not a garden.
    cells = {(dx, 0, dz): _FLOWER for dx in range(5) for dz in range(5) if (dx + dz) % 2 == 0}
    return Template(name="flower garden", goal="decorative", cells=cells)


def _fountain() -> Template:
    cells = _rect_perimeter(5, 5, 0, REQ_SOLID)
    cells[(2, 0, 2)] = REQ_SOLID
    cells[(2, 1, 2)] = REQ_SOLID
    return Template(name="fountain", goal="decorative", cells=cells)


TEMPLATES: dict[str, tuple[Template, ...]] = {
    "habitation": (_cabin(), _longhouse(), _treehouse()),
    "infrastructure": (_railed_bridge(), _flat_span(), _road()),
    "production": (_crop_field(), _bordered_plot(), _fence_pen(), _post_and_rail_pen()),
    "defense": (_square_tower(), _pillar_tower(), _perimeter_wall()),
    "decorative": (_flower_garden(), _fountain()),
}


def instance(subtype: str) -> Template:
    """The template that realizes one taxonomy subtype."""
    for templates in TEMPLATES.values():
        for template in templates:
            if template.name == subtype:
                return template
    raise KeyError(subtype)


assert set(TEMPLATES) == set(GOALS), "every category needs template instances"
assert all({t.name for t in TEMPLATES[goal]} == set(TAXONOMY[goal]) for goal in GOALS), \
    "template instances and taxonomy subtypes must be the same names"
