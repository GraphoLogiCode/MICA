"""Arm 2 — the LLM-PROMPTED intent mechanism of the four-arm comparison.

Per D0's arm table: a language model is shown the SYMBOLIC scene (the same evidence
fields every other arm reads — built count, shape facts, the per-goal template reads,
held item, recent macro-actions) and names the goal category plus a stated confidence,
each step judged alone. No filter, no learning, no pixels — the mechanism under test
is "ask a general model to read intent from a described scene".

The stated answer becomes a distribution by the mapping pinned in D4 (S1, arm
weights): mass = stated confidence on the named category, the residual spread
uniformly over all five, so the distribution always sums to one and an unusable
answer degrades to uniform (recorded, never dropped). Confidence is clamped to [0,1].

Every reply is cached append-only in capture/raw/arm2_cache.jsonl keyed by a hash of
the prompt, so reruns of the comparison are free and deterministic; the model id that
produced each verdict rides along for provenance. The local Ollama door
(mica/data/vlm.py) serves the calls — nothing leaves the machine.
"""
from __future__ import annotations

import hashlib
import json
import os

from ..contracts.b1 import GOALS
from ..contracts.b3 import FusedEvidence
from ..data import vlm

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CACHE_PATH = os.path.join(_ROOT, "capture", "raw", "arm2_cache.jsonl")

_PROMPT = """You are watching a Minecraft player build. Based only on the facts below, which ONE category is the player most likely building: habitation, infrastructure, production, defense, or decorative?

{scene}

Answer as JSON: {{"category": "<one of the five words>", "confidence": <number 0 to 1>}}"""


def scene_description(fused: FusedEvidence) -> str:
    """The symbolic scene, in plain sentences. Only evidence-side fields — the same
    information the other arms consume, worded instead of vectorized."""
    g = fused.global_feats
    lines = [f"Blocks placed so far: {g.built_count}."]
    if g.bbox:
        lines.append(f"Build bounding box (x,y,z): {g.bbox[0]}x{g.bbox[1]}x{g.bbox[2]} blocks.")
    if g.has_enclosure:
        lines.append("The structure encloses an interior space.")
    if g.planar_runs:
        lines.append(f"Flat planar runs detected: {g.planar_runs}.")
    held = fused.state_feats.held_item.replace("minecraft:", "")
    if held != "air":
        lines.append(f"The player is holding: {held}.")
    recent = [a for a in fused.state_feats.recent_actions]
    if recent:
        lines.append("Recent actions: " + ", ".join(recent) + ".")
    reads = []
    for goal in GOALS:
        feats = fused.per_goal[goal]
        if feats.fit * feats.comp > 0.05:
            reads.append(f"{goal} ({feats.subtype}): shape fit {feats.fit:.2f},"
                         f" completion {feats.comp:.2f}")
    if reads:
        lines.append("Template comparisons against known category shapes: "
                     + "; ".join(reads) + ".")
    return "\n".join(lines)


def _mapping(category: str | None, confidence: float) -> dict[str, float]:
    """Stated (category, confidence) -> a proper distribution (the pinned mapping)."""
    confidence = max(0.0, min(1.0, confidence))
    residual = (1.0 - confidence) / len(GOALS)
    if category not in GOALS:
        return {goal: 1.0 / len(GOALS) for goal in GOALS}
    return {goal: confidence + residual if goal == category else residual
            for goal in GOALS}


class Arm2:
    """The prompted reader with its disk cache. One instance per comparison run."""

    def __init__(self, cache_path: str = _CACHE_PATH):
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        self.queries = 0
        self.cache_hits = 0
        self.unusable = 0
        if os.path.exists(cache_path):
            live_model = vlm.describe()
            for line in open(cache_path, encoding="utf-8"):
                if line.strip():
                    entry = json.loads(line)
                    # A cache entry is only reusable if the SAME model produced it —
                    # otherwise a model swap would silently serve stale verdicts
                    # under the new model's name (harness review 2026-07-05, F3).
                    if entry.get("model") == live_model:
                        self.cache[entry["key"]] = entry

    def read(self, fused: FusedEvidence) -> dict[str, float]:
        """One step -> the LLM's goal distribution (cached by scene)."""
        prompt = _PROMPT.format(scene=scene_description(fused))
        key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
        entry = self.cache.get(key)
        if entry is None:
            raw = vlm.ask([], prompt, json_format=True)
            self.queries += 1
            category, confidence = self._parse(raw)
            entry = {"key": key, "category": category, "confidence": confidence,
                     "model": vlm.describe(), "raw": (raw or "")[:200]}
            self.cache[key] = entry
            with open(self.cache_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        else:
            self.cache_hits += 1
        if entry["category"] is None:
            self.unusable += 1
        return _mapping(entry["category"], entry["confidence"])

    @staticmethod
    def _parse(raw: str | None) -> tuple[str | None, float]:
        if not raw:
            return None, 0.0
        try:
            reply = json.loads(raw)
            category = str(reply.get("category", "")).strip().lower()
            confidence = float(reply.get("confidence", 0.0))
        except (ValueError, TypeError):
            lower = raw.lower()
            category = next((goal for goal in GOALS if goal in lower), "")
            confidence = 0.5 if category else 0.0
        return (category if category in GOALS else None), confidence
