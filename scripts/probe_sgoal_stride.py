"""B1-F3 early-separation probe: does a longer MineCLIP clip span separate goals better?

    python scripts/probe_sgoal_stride.py [--strides 1,6,20]

The pre-registered s_goal risk is flatness: on real play the five goal cosines sit
within ~0.01 of each other. MineCLIP was trained on clips spanning seconds, while our
default clip is the last 0.8 s (16 frames, stride 1) — so the review's F2 question is
whether a multi-second pre-action span (stride 6 ~ 4.8 s, stride 20 ~ 16 s) separates
the builder's true category better. This probe computes s_goal at every stride in one
pass over each session and never writes evidence files, so the banked stride-1
artifacts (guarded by B1-F5) are untouched.

Sessions: the five template captures plus the two clean free builds with full frame
coverage — truth is the builder's own category from capture/raw/labels.json. Records
enter the probe only when their evidence window holds at least one frame (run_d1's own
rule), so every stride scores the identical record set.

Also carries the B1-F3 idle-precision sample: for each session, up to ten idle-scored
records' pre-action frame paths, for an eyeball check that "idle" frames really show
nothing happening.

Report -> capture/raw/sgoal_stride_probe.json. The verdict names the stride with the
best early-window margin (early = progress <= 0.35, the same cutoff Source B calls
early); adopting it as the corpus default is a separate, deliberate step.
"""
from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.capture.jsonl_ingest import JsonlSource            # noqa: E402
from mica.contracts.b1 import GOALS, MacroAction             # noqa: E402
from mica.perception.evidence2d import evidence_stream       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_EARLY_CUTOFF = 0.35     # matches label_finished_builds.py's "early pair" definition
_IDLE_SAMPLES = 10

# The probe corpus, pinned: five template captures + the two clean free builds.
# Contested sessions are excluded not because their truth is doubtful (builder's word
# stands) but to keep this the same seven-session set the template gate certified.
_SESSIONS = (
    "fabric-20260704-232045",   # production / crop field (template)
    "fabric-20260705-002717",   # habitation (template, style caveat; thin frames)
    "fabric-20260705-131308",   # infrastructure / road (template)
    "fabric-20260705-131826",   # defense / square tower (template)
    "fabric-20260705-134615",   # decorative / fountain (template)
    "fabric-20260705-113204",   # decorative / fancy fountain (clean free build)
    "fabric-20260705-112050",   # infrastructure / bridge (clean free build)
)


def _strides() -> tuple[int, ...]:
    if "--strides" in sys.argv:
        raw = sys.argv[sys.argv.index("--strides") + 1]
        return tuple(int(v) for v in raw.split(","))
    return (1, 6, 20)


class _FrameCache:
    """Decoded, MineCLIP-sized frames by tick. Records arrive in tick order and the
    deepest lookback is 16 * max_stride ticks, so anything older is safe to drop —
    the cache stays a few hundred MB below naive full-resolution loading."""

    def __init__(self, session, lookback: int):
        self._refs = {p.tick: p.client.pov_frame for p in session.packets
                      if p.client.pov_frame is not None}
        self._lookback = lookback
        self._decoded: dict[int, "object"] = {}

    def frames_between(self, t0: int, t1: int) -> list:
        import numpy as np
        from PIL import Image

        loaded = []
        for tick in range(t0, t1 + 1):
            ref = self._refs.get(tick)
            if ref is None or not os.path.exists(ref.path):
                continue
            if tick not in self._decoded:
                with Image.open(ref.path) as image:
                    # Pre-size to MineCLIP's native 256x160 here; _to_chw's own resize
                    # then costs nothing and the cache holds ~120 KB per frame.
                    self._decoded[tick] = np.asarray(
                        image.convert("RGB").resize((256, 160), Image.BILINEAR))
            loaded.append(self._decoded[tick])
        for tick in [t for t in self._decoded if t < t1 - self._lookback]:
            del self._decoded[tick]
        return loaded

    def last_path(self, t0: int, t1: int) -> str | None:
        for tick in range(t1, t0 - 1, -1):
            ref = self._refs.get(tick)
            if ref is not None and os.path.exists(ref.path):
                return ref.path
        return None


def _session_metrics(rows: list[dict], truth: str, stride: int) -> dict:
    """Separation numbers for one session at one stride. margin = truth cosine minus
    the mean of the other four; rank 1 = the truth cosine is the largest."""
    margins, ranks, firsts, early_margins = [], [], [], []
    truth_index = GOALS.index(truth)
    for row in rows:
        s = row["s_goal"][stride]
        others = [v for i, v in enumerate(s) if i != truth_index]
        margin = s[truth_index] - statistics.mean(others)
        rank = 1 + sum(1 for v in others if v > s[truth_index])
        margins.append(margin)
        ranks.append(rank)
        firsts.append(rank == 1)
        if row["progress"] <= _EARLY_CUTOFF:
            early_margins.append(margin)
    return {
        "records": len(rows),
        "mean_margin": round(statistics.mean(margins), 4),
        "mean_truth_rank": round(statistics.mean(ranks), 2),
        "truth_first_fraction": round(statistics.mean(firsts), 3),
        "early_records": len(early_margins),
        "early_mean_margin": round(statistics.mean(early_margins), 4) if early_margins else None,
    }


def main() -> int:
    strides = _strides()
    with open(os.path.join(_RAW, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)

    from mica.perception.mineclip_head import _CLIP_LEN, MineClipHead
    print(f"loading frozen MineCLIP on GPU ... (strides {', '.join(map(str, strides))})")
    head = MineClipHead()

    per_session: dict[str, dict] = {}
    idle_samples: dict[str, list] = {}
    for session_id in _SESSIONS:
        jsonl = os.path.join(_RAW, f"{session_id}.jsonl")
        if not os.path.exists(jsonl):
            print(f"  {session_id}: recording missing, skipped")
            continue
        truth = labels[session_id]["goal"]
        session = JsonlSource(jsonl, jsonl.replace(".jsonl", ".manifest.json")).load()
        scored = [ev for ev in evidence_stream(session.packets) if ev.scored]
        total_events = sum(len(ev.event_ids) for ev in scored) or 1
        cache = _FrameCache(session, lookback=_CLIP_LEN * max(strides))

        rows = []
        consumed = 0
        idles = []
        for ev in scored:
            t0, t1 = ev.tick_range
            window = cache.frames_between(t0, t1)
            progress = consumed / total_events
            consumed += len(ev.event_ids)
            if ev.a_hat == MacroAction.IDLE:
                idles.append({"tick": t1, "frame": cache.last_path(t0, t1)})
            if not window:
                continue          # no frames in the window: skipped at EVERY stride
            scores = {}
            for stride in strides:
                clip = window if stride == 1 else cache.frames_between(
                    t1 - _CLIP_LEN * stride + 1, t1)
                scores[stride] = head.score(clip, stride)
            rows.append({"progress": progress, "s_goal": scores})

        step = max(1, len(idles) // _IDLE_SAMPLES)
        idle_samples[session_id] = [s for s in idles[::step][:_IDLE_SAMPLES] if s["frame"]]
        per_session[session_id] = {
            "truth": truth,
            "scored_records": len(scored),
            "records_with_frames": len(rows),
            "per_stride": {str(s): _session_metrics(rows, truth, s) for s in strides} if rows else {},
        }
        line = ", ".join(
            f"stride {s}: margin {per_session[session_id]['per_stride'][str(s)]['mean_margin']:+.4f}"
            f" (early {per_session[session_id]['per_stride'][str(s)]['early_mean_margin']})"
            for s in strides) if rows else "no frame-carrying records"
        print(f"  {session_id} [{truth}] {len(rows)}/{len(scored)} records — {line}")

    # Sessions weigh equally in the aggregate: one long AFK-heavy session must not
    # decide the corpus default by sheer record count.
    aggregate = {}
    for stride in strides:
        key = str(stride)
        entries = [s["per_stride"][key] for s in per_session.values() if s["per_stride"]]
        early = [e["early_mean_margin"] for e in entries if e["early_mean_margin"] is not None]
        aggregate[key] = {
            "sessions": len(entries),
            "mean_margin": round(statistics.mean(e["mean_margin"] for e in entries), 4),
            "mean_truth_rank": round(statistics.mean(e["mean_truth_rank"] for e in entries), 2),
            "truth_first_fraction": round(
                statistics.mean(e["truth_first_fraction"] for e in entries), 3),
            "early_mean_margin": round(statistics.mean(early), 4) if early else None,
        }

    best = max(aggregate, key=lambda k: (aggregate[k]["early_mean_margin"] or -1.0,
                                         aggregate[k]["mean_margin"]))
    flat = all(abs(a["mean_margin"]) < 0.01 and (a["early_mean_margin"] is None
               or abs(a["early_mean_margin"]) < 0.01) for a in aggregate.values())
    report = {
        "question": "does a longer MineCLIP clip span separate the true category better?",
        "strides": list(strides),
        "early_cutoff_progress": _EARLY_CUTOFF,
        "aggregate": aggregate,
        "best_stride_by_early_margin": int(best),
        "channel_flat_at_every_stride": flat,
        "per_session": per_session,
        "idle_precision_sample": idle_samples,
    }
    out = os.path.join(_RAW, "sgoal_stride_probe.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("aggregate (sessions weighted equally; margin = truth minus mean of the rest):")
    for stride in strides:
        a = aggregate[str(stride)]
        print(f"  stride {stride:>2}: margin {a['mean_margin']:+.4f}   early {a['early_mean_margin']}"
              f"   truth-first {a['truth_first_fraction']:.3f}   mean rank {a['mean_truth_rank']}")
    print(f"  best by early margin: stride {best}"
          + ("   — BUT the channel is flat (<0.01) at every stride; the CLIP4MC rung question is live"
             if flat else ""))
    print(f"  -> {os.path.relpath(out, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
