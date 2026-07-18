"""Hand-coded likelihood heads v0 — the "belief tracker with hand-coded likelihoods"
baseline from the evaluation plan. NOT the trained D3 heads: no parameters were fit to
anything; these are transparent forms chosen so the tracker runs end to end with the
FULL fused evidence and its machinery can be proven before any training exists.

The heads consume B3 (contracts/b3.py), and the fusion is real: both streams shape the
likelihood. From D1 — the held item (does the hand match the winning style's signature
materials?), the recent-action streak and crosshair dwell (shared action-tendency
context), and s_goal when pixels exist. From D2 — the per-goal (fit, comp) pair and the
progress delta. The D1 with/without probe is a first-class ablation: strip_behavior()
blanks exactly the 2D channels, so the behavior stream's contribution is measurable.

The four D3 head constraints, honored by construction:
  1. Goal symmetry — the heuristic head takes NO goal argument; it reads only the
     goal-free dwell channel (the placing streak is a deliberative-side GAIN only —
     it never reaches this head; review 13 F29 corrected the older claim). s_goal
     and per_goal reach the deliberative head only.
  2. a_hat_conf never multiplies into a likelihood (it is not read at all).
  3. s_goal arrives as raw cosines; a small weight scales it.
  4. comp enters only jointly with fit (the product), never alone.

The softmax balance law these numbers obey (learned from a 0/30 inversion): pauses
outnumber placements and a softmax couples all actions, so net evidence per correction
f*bump - log(1 + p*(e^bump - 1)) is positive only when the base place probability p sits
well below the realized build fraction f (~0.4). Optimized near p ~ 0.06, bump ~ 2.2.
Progress saturates gently — a hot gain lets every advancing category max the term out.
"""
from __future__ import annotations

import dataclasses
import math

from ..contracts.b1 import GOALS, MacroAction
from ..contracts.b3 import FusedEvidence
from ..perception.templates import REQ_SOLID, TEMPLATES
from . import h3d_readout
from .tracker import Belief

# The emitted macro-action classes (SCAFFOLD folds into PLACE in v1) — |A| = 5.
ACTIONS = (MacroAction.PLACE, MacroAction.BREAK, MacroAction.NAVIGATE,
           MacroAction.INSPECT, MacroAction.IDLE)

_PROGRESS_GAIN = 3.0
_BUMP_PROGRESS = 1.0
_BUMP_ANCHOR = 1.6
_BUMP_MATERIAL = 0.8    # the hand holds a style-signature block (fences, poppies, ...)
_H3D_WEIGHT = 1.0       # learned Uni3D shape readout, centered + anchoring-gated
                        # (probe_h3d_fusion measures it; see the gate note in deliberative)
_SGOAL_WEIGHT = 1.5
_STREAK_WEIGHT = 0.8    # recent placing predicts more placing — goal-free, both heads
_DWELL_WEIGHT = 0.3     # a fixated crosshair leans toward acting on the spot

# Which exact blocks signal each style — a hand full of oak fences is pen evidence the
# pixel models never see crisply. Styles built from any solid block have no signature.
# Keyed by taxonomy-v3 SUBTYPE (what the evidence's style read carries), pooling every
# instance that realizes it — so "animal_husbandry" unions the fence pen's fences with
# the post-and-rail pen's logs and slabs.
_STYLE_MATERIALS: dict[str, frozenset[str]] = {}
for _templates in TEMPLATES.values():
    for _template in _templates:
        signature = frozenset(required for required in _template.cells.values()
                              if required != REQ_SOLID)
        _STYLE_MATERIALS[_template.subtype] = (
            _STYLE_MATERIALS.get(_template.subtype, frozenset()) | signature)
# Category-level union, for when no style read exists (the structure-stripped arm):
# knowing a category's material vocabulary is static goal knowledge, not runtime 3D.
_CATEGORY_MATERIALS: dict[str, frozenset[str]] = {
    goal: frozenset(block for template in templates for block in
                    _STYLE_MATERIALS[template.subtype])
    for goal, templates in TEMPLATES.items()
}


def _softmax(logits: dict) -> dict:
    peak = max(logits.values())
    exps = {action: math.exp(value - peak) for action, value in logits.items()}
    total = sum(exps.values())
    return {action: value / total for action, value in exps.items()}


def _behavior_context(evidence: FusedEvidence) -> tuple[float, float]:
    """The goal-free 2D signals both heads may read: placing streak and crosshair dwell."""
    recent = evidence.state_feats.recent_actions
    streak = (sum(1 for a in recent if a == "place") / len(recent)) if recent else 0.0
    dwell = min(evidence.focus.dwell_ticks, 20) / 20.0 if evidence.focus.block else 0.0
    return streak, dwell


def _base_logits(evidence: FusedEvidence) -> dict:
    """Action tendencies before any goal enters — shared by both heads, so the goal
    axis is moved only by the deliberative bump and the mode axis stays a fair fight.

    The base PLACE rate stays LOW and constant: an earlier version let the placing
    streak raise it, which broke the balance law exactly when placements arrived and
    made the fused arm WORSE than the stripped one (measured: 0.800 vs 0.933). The
    streak instead gates the deliberative bump below; dwell nudges INSPECT only.
    """
    _, dwell = _behavior_context(evidence)
    return {
        MacroAction.PLACE: -0.6,
        MacroAction.BREAK: -0.9,
        MacroAction.NAVIGATE: 0.8,
        MacroAction.INSPECT: 0.9 + _DWELL_WEIGHT * dwell,
        MacroAction.IDLE: 1.0,
    }


def _material_match(evidence: FusedEvidence, goal: str) -> float:
    """Does the held item say this category's winning style? Fences say pen, poppies say
    garden; a style built from any solid block gets a weak nod for holding one at all.
    Without a style read (structure stripped), fall back to the category's whole
    material vocabulary — static goal knowledge, not a 3D runtime output."""
    subtype = evidence.per_goal[goal].subtype
    signature = _STYLE_MATERIALS.get(subtype, _CATEGORY_MATERIALS[goal])
    held = evidence.state_feats.held_item
    if signature:
        return 1.0 if held in signature else 0.0
    return 0.3 if held != "minecraft:air" else 0.0


def deliberative(evidence: FusedEvidence, goal: str) -> dict:
    """P(a | e, g, z=0): a builder pursuing g places when g is advancing, anchored,
    and the right materials are in hand."""
    feats = evidence.per_goal[goal]
    progress = max(0.0, min(1.0, feats.delta_comp * _PROGRESS_GAIN))
    anchored = feats.fit * feats.comp
    bump = (_BUMP_PROGRESS * progress + _BUMP_ANCHOR * anchored
            + _BUMP_MATERIAL * _material_match(evidence, goal))
    shape = h3d_readout.scores(evidence.h3d)
    if shape is not None:
        # The learned Uni3D shape readout — what the geometry itself says the build is
        # becoming. g-indexed (trained on goal labels), so deliberative-only; CENTERED
        # so it is zero-INFORMATION in logit space: it discriminates between goals.
        # Honesty note (review 13 F16): centering does NOT make it rate-preserving —
        # softmax is convex, so even pure zero-sum noise here lifts the goal-averaged
        # P(PLACE) second-order (measured 0.0657 -> 0.0747 at sigma=1), reading as
        # spurious deliberative-mode evidence. The ramp_in x unanchored gates below
        # bound the term; a noisy readout still tilts P(z) slightly and that residual
        # is accepted and recorded, not denied.
        #
        # Its measured reliability window (h3d_linear_probe.json) sets two goal-free
        # gates: near-chance on the smallest clouds (0.3 at the 10% bin — a couple of
        # blocks say nothing), so it ramps IN with built size; and it trails once a
        # template locks on late, so it fades OUT as the best fit x comp anchors.
        # Both scalars are the same for every g, so the zero-sum property survives.
        ramp_in = min(1.0, evidence.global_feats.built_count / 8.0)
        unanchored = 1.0 - max(f.fit * f.comp for f in evidence.per_goal.values())
        bump += _H3D_WEIGHT * ramp_in * unanchored * (shape[goal] - 1.0 / len(GOALS))
    # The placing streak is goal-free, so it may not shift the base rate (balance law);
    # its legal role is a gain control — mid-burst, the structure evidence is more
    # diagnostic of what the burst is building. Applied uniformly across g.
    streak, _ = _behavior_context(evidence)
    bump *= 0.7 + _STREAK_WEIGHT * streak
    if evidence.s_goal is not None:
        bump += _SGOAL_WEIGHT * evidence.s_goal[GOALS.index(goal)]
    logits = _base_logits(evidence)
    logits[MacroAction.PLACE] += bump
    logits[MacroAction.BREAK] += 0.3 * progress   # fixing mistakes rides on activity
    return _softmax(logits)


def heuristic(evidence: FusedEvidence) -> dict:
    """P(a | e, z=1): autopilot. Reads only the goal-free dwell channel (via
    _base_logits) plus a constant placing bump — the signature has no goal argument,
    which is the identifiability guarantee."""
    logits = _base_logits(evidence)
    logits[MacroAction.PLACE] += 0.9   # habitual placing, unconditioned on any goal
    return _softmax(logits)


def likelihood(evidence: FusedEvidence, action: MacroAction) -> Belief:
    """The full L_k(g, z) = P(a_k | e_k, g, z) the tracker corrects with."""
    if action == MacroAction.SCAFFOLD:
        # v1 folds SCAFFOLD into PLACE (features.action_index); this arm must agree,
        # or a future segmenter emitting SCAFFOLD crashes one arm of the comparison
        # while the other keeps running (review 13 F28).
        action = MacroAction.PLACE
    heuristic_p = heuristic(evidence)[action]
    result = {}
    for goal in GOALS:
        result[(goal, 0)] = deliberative(evidence, goal)[action]
        result[(goal, 1)] = heuristic_p   # constant in g, by construction
    return result


def strip_behavior(evidence: FusedEvidence) -> FusedEvidence:
    """The D1 with/without probe: blank exactly the 2D behavior channels, keep the 3D
    structure — what remains is the structure-plus-action-label ablation arm."""
    return dataclasses.replace(
        evidence,
        state_feats=dataclasses.replace(
            evidence.state_feats, held_item="minecraft:air", recent_actions=()),
        focus=dataclasses.replace(evidence.focus, block=None, dwell_ticks=0),
        s_goal=None,
        h2d=None,
    )


def strip_structure(evidence: FusedEvidence) -> FusedEvidence:
    """The D2 with/without probe, mirror of strip_behavior: blank exactly the 3D
    structure channels — per-goal features zeroed, the style read removed, global shape
    facts emptied — keeping the behavior stream. What remains is the 2D-only arm."""
    zeroed = {
        goal: dataclasses.replace(evidence.per_goal[goal], comp=0.0, edit_distance=0,
                                  fit=0.0, subtype="", delta_comp=0.0)
        for goal in GOALS
    }
    empty_global = dataclasses.replace(
        evidence.global_feats, built_count=0, bbox=None, centroid=None, planar_runs=0,
        has_enclosure=False, symmetry=0.0, symmetry_support=0)
    return dataclasses.replace(evidence, per_goal=zeroed, global_feats=empty_global, h3d=None)
