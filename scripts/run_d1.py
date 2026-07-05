"""Run D1 over a recorded session: write evidence2d.jsonl + a macro-action fidelity report.

    python scripts/run_d1.py [session.jsonl]            symbolic channels only (no models)
    python scripts/run_d1.py [session.jsonl] --pixels   also fill h2d (VPT) + s_goal (MineCLIP)
    flags: --allow-ungated     consume a session even if it fails the B0 gate (debugging only)
           --overwrite         replace an existing pixel-enriched evidence2d.jsonl
           --sgoal-stride N    space the 16-frame MineCLIP clip every N frames counted back
                               from the window end (default 1 = the last 0.8 s; 6 ~ 4.8 s of
                               pre-action context) — the probe's clip-span A/B axis

The session must PASS the B0 gate first — evidence built from an unverified capture would
poison everything downstream, so an ungated session is refused, not warned about.
evidence2d.jsonl is the B1 proof artifact — one scored Evidence2D per macro-action plus
unscored ~1 Hz context records, replayable. The fidelity report checks the rule-based a_hat
against the lossless server events: every scored PLACE/BREAK must line up with real block
events, each consumed exactly once. With --pixels it loads the frozen VPT + MineCLIP heads,
fills the two pixel channels from the saved frames in each record's window, and records the
checkpoints and goal prompts it used (D1 spec §8). Exit 0 if consistent, 1 otherwise.
"""
from __future__ import annotations

import collections
import dataclasses
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.discovery import newest_capture          # noqa: E402
from mica.capture.jsonl_ingest import JsonlSource          # noqa: E402
from mica.contracts.b1 import GOALS, MacroAction           # noqa: E402
from mica.contracts.serialize import evidence2d_to_dict    # noqa: E402
from mica.perception.evidence2d import evidence_stream     # noqa: E402
from mica.perception.pixel_head import write_d1_provenance  # noqa: E402
from mica.validation.b0_gate import gate_checks            # noqa: E402
from mica.validation.evidence2d_check import check_stream  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_BUILD_ACTIONS = (MacroAction.PLACE.value, MacroAction.BREAK.value)


def _enrich_pixels(records, session, stride):
    """Fill h2d (from the record's window) + s_goal (last 16 frames at the given stride,
    reaching further back than the window when stride > 1 — still strictly pre-action)."""
    from PIL import Image
    from mica.perception.mineclip_head import _CLIP_LEN, MineClipHead
    from mica.perception.vpt_trunk import VptTrunk

    vpt, mineclip = VptTrunk(), MineClipHead()
    by_tick = {packet.tick: packet for packet in session.packets}

    def frames_between(t0, t1):
        loaded = []
        for t in range(t0, t1 + 1):
            packet = by_tick.get(t)
            ref = packet.client.pov_frame if packet else None
            if ref is not None and os.path.exists(ref.path):
                with Image.open(ref.path) as image:
                    loaded.append(image.convert("RGB"))   # load into memory; the file handle closes here
        return loaded

    enriched = []
    for ev in records:
        t0, t1 = ev.tick_range
        window = frames_between(t0, t1)
        if not window:
            # No frames in the evidence window: both pixel channels stay None together,
            # even if a wider clip span would have found older frames.
            enriched.append(dataclasses.replace(ev, h2d=None, s_goal=None))
            continue
        clip = window if stride == 1 else frames_between(t1 - _CLIP_LEN * stride + 1, t1)
        enriched.append(dataclasses.replace(ev, h2d=vpt.embed(window), s_goal=mineclip.score(clip, stride)))
    return enriched


def _existing_has_pixels(path: str) -> bool:
    """True when an evidence file on disk already carries GPU-computed pixel channels."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip() and json.loads(line).get("h2d") is not None:
                return True
    return False


def _sgoal_stride() -> int:
    """The --sgoal-stride value (default 1 = the last 0.8 s, today's behavior)."""
    if "--sgoal-stride" in sys.argv:
        return max(1, int(sys.argv[sys.argv.index("--sgoal-stride") + 1]))
    return 1


def main() -> int:
    args = sys.argv[1:]
    value_slots = {i + 1 for i, a in enumerate(args) if a == "--sgoal-stride"}  # its value isn't a path
    positional = [a for i, a in enumerate(args) if not a.startswith("--") and i not in value_slots]
    jsonl = positional[0] if positional else newest_capture(_RAW)
    if not jsonl or not os.path.exists(jsonl):
        print("no capture found")
        return 1
    session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()

    # A capture the B0 gate rejects (truncated, misaligned, missing fields) must never
    # become evidence — refuse it outright so the corpus can't be poisoned quietly.
    capture_dir = os.path.dirname(os.path.abspath(jsonl))
    failing = [(name, detail) for name, passed, detail in gate_checks(session, capture_dir) if not passed]
    if failing and "--allow-ungated" not in sys.argv:
        print(f"D1  {os.path.basename(jsonl)}")
        print("  REFUSED: this session fails the B0 gate (run scripts/b0_gate.py):")
        for name, detail in failing:
            print(f"    [FAIL] {name}  {detail}")
        print("  (--allow-ungated overrides, for debugging only)")
        return 1

    records = list(evidence_stream(session.packets))
    out = jsonl.replace(".jsonl", ".evidence2d.jsonl")
    pixels = "--pixels" in sys.argv
    if not pixels and os.path.exists(out) and _existing_has_pixels(out) and "--overwrite" not in sys.argv:
        print(f"  REFUSED: {os.path.basename(out)} already carries GPU pixel channels; a symbolic-only"
              " run would erase them. Rerun with --pixels, or --overwrite to discard them.")
        return 1

    stride = _sgoal_stride()
    if pixels:
        print(f"  loading frozen VPT + MineCLIP on GPU ... (s_goal clip stride {stride})")
        records = _enrich_pixels(records, session, stride)

    with open(out, "w", encoding="utf-8") as handle:
        for ev in records:
            handle.write(json.dumps(evidence2d_to_dict(ev)) + "\n")

    corrections = [ev for ev in records if ev.scored]
    context = [ev for ev in records if not ev.scored]
    per_class = collections.Counter(ev.a_hat.value for ev in corrections)
    consumed = [eid for ev in records for eid in ev.event_ids]
    # Fidelity is scoped to the HUMAN's events: the agent's own block events are
    # excluded from evidence by contract (A7, contracts/b0.py), so they are expected
    # to be unconsumed — counted separately, never as a failure.
    from mica.contracts.b0 import is_agent_actor
    all_events = [e.event_id for p in session.packets for e in p.server.block_events
                  if not is_agent_actor(e.actor)]
    agent_events = sum(1 for p in session.packets for e in p.server.block_events
                       if is_agent_actor(e.actor))
    duplicates = [eid for eid, n in collections.Counter(consumed).items() if n > 1]
    unconsumed = sorted(set(all_events) - set(consumed))   # every HUMAN event must land in exactly one record
    mislabeled = [ev.a_hat.value for ev in corrections
                  if (ev.a_hat.value in _BUILD_ACTIONS) != bool(ev.event_ids)]

    print(f"D1  {os.path.basename(jsonl)}")
    print(f"  evidence records: {len(corrections)} scored + {len(context)} context  ->  {os.path.basename(out)}")
    print("  macro-actions (scored): " + ", ".join(f"{k}={per_class[k]}" for k in sorted(per_class)))
    print(f"  block events: {len(all_events)} human, {len(set(consumed))} consumed once,"
          f" {len(duplicates)} double-counted, {len(unconsumed)} unconsumed"
          + (f", {agent_events} agent (excluded by A7)" if agent_events else ""))
    ok = not duplicates and not mislabeled and not unconsumed
    print(f"  [{'PASS' if ok else 'FAIL'}] build-action fidelity"
          + ("" if ok else f"  (duplicates={duplicates}, mislabeled={mislabeled}, unconsumed={unconsumed})"))

    if pixels:
        filled = [ev.s_goal for ev in records if ev.s_goal is not None]
        means = [statistics.mean(s[i] for s in filled) for i in range(len(GOALS))] if filled else []
        ranked = sorted(zip(GOALS, means), key=lambda gm: -gm[1])
        print(f"  pixels: {len(filled)}/{len(records)} records carry h2d(1024) + s_goal({len(GOALS)})")
        print("  mean s_goal: " + ", ".join(f"{g}={m:.3f}" for g, m in ranked))
        print(f"  provenance: {os.path.basename(write_d1_provenance(out, stride))}")

    issues = check_stream(records)
    print(f"  [{'PASS' if not issues else 'FAIL'}] B1 contract validation"
          + ("" if not issues else f"  ({len(issues)} issues; first: {issues[0]})"))
    return 0 if (ok and not issues) else 1


if __name__ == "__main__":
    raise SystemExit(main())
