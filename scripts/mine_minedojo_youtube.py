"""Source C — mine the MineDojo YouTube corpus for real house-building footage.

    python scripts/mine_minedojo_youtube.py [--full] [--transcripts N] [--visual N]
                                            [--top N]

SCOPE (D3, Source C, 2026-07-04): this corpus is PIXELS-ONLY. Mined clips carry no
block events — no B0, no structure stream, no fused evidence — so nothing here can
ever become Source B pairs or train the likelihood heads. Licensed roles: s_goal
separation measurement on real footage (diagnostic — MineCLIP also scores the
retrieval, so selection and measurement share a model), h2d domain statistics, and
candidate Phase-E pretraining for the pixel channels. Every artifact this script
writes carries that scope line.

Stages (each later stage refines the previous one's shortlist):
  1. INDEX    — the MineDojo index (Zenodo, CC BY 4.0; youtube_tutorial.json by
                default, --full adds the 174 MB general-gameplay index). Downloaded
                once into capture/youtube/, then offline. Keyword scoring on titles.
  2. TRANSCRIPTS (--transcripts N) — fetch auto-captions for the top N title
                candidates via yt-dlp and score build-vocabulary density per minute.
  3. VISUAL   (--visual N) — sample frames from the top N remaining candidates
                (yt-dlp segment download) and score them with frozen MineCLIP
                against the house prompts. Needs torch + the checkpoint.

Output: capture/youtube/house_subset.json (the kept clips with every score and the
matched phrases) + house_mining_report.json + a printed sample sheet for the manual
spot-check the plan requires.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request

# YouTube titles carry characters the Windows console's default codepage cannot
# print — never let a print crash a mining run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "youtube")
_ZENODO = "https://zenodo.org/api/records/6693792/files/{name}/content"
_SCOPE = ("Source C (D3): pixels-only analysis/pretraining corpus - never fused "
          "evidence, never Source B pairs, never head training data")

# Transparent title scoring — every constant visible here and echoed in the report.
# One vocabulary per goal CATEGORY (contracts/goals.py): the same retrieval pipeline
# runs for each, so Source C can grow into a category-balanced pixel corpus. Each
# category carries its clip-verdict label and the definition the VLM judges against.
_CATEGORIES = {
    "habitation": {
        "label": "HOUSE",
        "definition": ("a dwelling — house, cabin, cottage, hut; not a farm, "
                       "statue, redstone machine, or pure terrain"),
        "strong": ("how to build a house", "house tutorial", "building a house",
                   "build a house", "starter house", "survival house", "modern house",
                   "wooden house", "small house", "easy house", "house build",
                   "cabin tutorial", "cottage tutorial", "base tutorial"),
        "subjects": ("house", "home", "cabin", "cottage", "mansion", "base", "hut", "villa"),
    },
    "production": {
        "label": "PRODUCTION",
        "definition": ("a working farm or animal enclosure — crop fields, wheat or "
                       "vegetable farms, barns, animal pens, stables; not a dwelling "
                       "house and not a decorative garden"),
        "strong": ("farm tutorial", "how to build a farm", "how to make a farm",
                   "wheat farm", "crop farm", "animal farm", "animal pen",
                   "barn tutorial", "starter farm", "survival farm"),
        "subjects": ("farm", "barn", "pen", "crop", "wheat", "stable", "coop", "pasture"),
    },
    "infrastructure": {
        "label": "INFRASTRUCTURE",
        "definition": ("transport or connective structures — bridges, roads, paths, "
                       "railways, tunnels, docks, harbors; not a building"),
        "strong": ("bridge tutorial", "how to build a bridge", "road tutorial",
                   "how to build a road", "path tutorial", "railway tutorial",
                   "rail tutorial", "dock tutorial", "harbor tutorial"),
        "subjects": ("bridge", "road", "path", "rail", "railway", "tunnel", "dock", "harbor"),
    },
    "defense": {
        "label": "DEFENSE",
        "definition": ("fortifications — castles, defensive walls, towers, "
                       "watchtowers, fortresses, moats, gates"),
        "strong": ("castle tutorial", "how to build a castle", "wall tutorial",
                   "tower tutorial", "how to build a tower", "watchtower",
                   "fortress tutorial", "how to build a fort", "castle wall"),
        "subjects": ("castle", "wall", "tower", "fort", "fortress", "moat", "gate", "keep"),
    },
    "decorative": {
        "label": "DECORATIVE",
        "definition": ("ornamental builds — fountains, statues, gardens, gazebos, "
                       "monuments, plazas; built to look at, not to live or work in"),
        "strong": ("fountain tutorial", "how to build a fountain", "statue tutorial",
                   "how to build a statue", "garden tutorial", "how to build a garden",
                   "gazebo tutorial", "monument tutorial"),
        "subjects": ("fountain", "statue", "garden", "gazebo", "monument", "plaza", "flower"),
    },
}
_BUILD_WORDS = ("build", "building", "tutorial", "how to", "construct")
_PENALTY = ("mod review", "modpack", "trailer", "top 10", "top 5", "seed",
            "server", "snapshot", "update", "vs", "battle", "speedrun")
_STRONG_W, _WORD_W, _PENALTY_W = 3, 1, -2
_KEEP_TITLE_SCORE = 4          # a strong phrase + one supporting word, or equivalent
_DURATION_RANGE = (180, 2400)  # a single-build tutorial is minutes, not seconds/hours
_TRANSCRIPT_KEEP = 1.0         # build-vocabulary hits per minute to keep a clip


def _flag(name: str, default: int | None) -> int | None:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _fetch_index(name: str) -> list[dict]:
    path = os.path.join(_OUT, name)
    if not os.path.exists(path):
        print(f"  downloading {name} from Zenodo (once) ...")
        try:
            urllib.request.urlretrieve(_ZENODO.format(name=name), path)
        except OSError as error:
            raise SystemExit(f"cannot reach the MineDojo index ({error}); download "
                             f"{name} by hand into capture/youtube/ and rerun") from error
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _title_score(title: str, vocab: dict) -> tuple[int, list[str]]:
    lower = title.lower()
    matched = [phrase for phrase in vocab["strong"] if phrase in lower]
    score = _STRONG_W * len(matched)
    for word in vocab["subjects"]:
        if re.search(rf"\b{word}\b", lower):
            matched.append(word)
            score += _WORD_W
            break                              # one subject word is enough evidence
    for word in _BUILD_WORDS:
        if word in lower:
            matched.append(word)
            score += _WORD_W
            break
    for phrase in _PENALTY:
        if phrase in lower:
            matched.append(f"-{phrase}")
            score += _PENALTY_W
    return score, matched


def _stage_titles(entries: list[dict], vocab: dict) -> list[dict]:
    candidates = []
    for entry in entries:
        score, matched = _title_score(entry.get("title", ""), vocab)
        duration = entry.get("duration") or 0
        if _DURATION_RANGE[0] <= duration <= _DURATION_RANGE[1]:
            score += _WORD_W
        if score >= _KEEP_TITLE_SCORE:
            candidates.append({**entry, "title_score": score, "matched": matched})
    candidates.sort(key=lambda c: (-c["title_score"], -(c.get("view_count") or 0)))
    return candidates


# ------------------------------------------------------------ transcript stage

def _fetch_transcript(video_id: str, cache_dir: str) -> tuple[str | None, str]:
    """English captions via youtube-transcript-api, cached as plain text.

    Returns (text, status). Coverage is PARTIAL by nature: the index is from 2022
    and many of its videos have captions disabled or are gone — a missing
    transcript therefore never discards a clip, it only means the title score
    stands unrefined. The status lands in the subset so the coverage is visible."""
    cached = os.path.join(cache_dir, f"{video_id}.txt")
    if os.path.exists(cached):
        return open(cached, encoding="utf-8").read(), "ok"
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None, "youtube-transcript-api not installed"
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
        text = " ".join(snippet.text for snippet in fetched)
    except Exception as error:               # disabled / unplayable / removed video
        return None, type(error).__name__
    with open(cached, "w", encoding="utf-8") as handle:
        handle.write(text)
    return text, "ok"


def _stage_transcripts(candidates: list[dict], limit: int, vocab: dict) -> int:
    cache_dir = os.path.join(_OUT, "transcripts")
    os.makedirs(cache_dir, exist_ok=True)
    vocabulary = set(vocab["subjects"]) | {"wall", "roof", "door", "floor", "window",
                                           "stairs", "foundation", "interior", "blocks"}
    fetched = 0
    for candidate in candidates[:limit]:
        text, status = _fetch_transcript(candidate["id"], cache_dir)
        candidate["transcript_status"] = status
        if text is None:
            candidate["transcript_hits_per_min"] = None
            continue
        fetched += 1
        words = text.lower().split()
        hits = sum(1 for word in words if word.strip(".,!?") in vocabulary)
        minutes = max((candidate.get("duration") or 60) / 60.0, 1.0)
        candidate["transcript_hits_per_min"] = round(hits / minutes, 2)
    return fetched


# ----------------------------------------------------------------- visual stage

def _stage_visual(candidates: list[dict], limit: int, category: str) -> None:
    """Frozen MineCLIP over frames sampled mid-video — the optional last filter.
    Selection sharing a model with any later s_goal analysis is why Source C results
    stay diagnostic (the D3 scope note)."""
    if shutil.which("yt-dlp") is None:
        print("  visual: yt-dlp not on PATH - stage skipped")
        return
    try:
        from PIL import Image
        from mica.perception.mineclip_head import MineClipHead
    except ImportError as error:
        print(f"  visual: torch stack unavailable ({error}) - stage skipped")
        return
    from mica.contracts.b1 import GOALS
    head = MineClipHead()
    goal_index = GOALS.index(category)
    frames_dir = os.path.join(_OUT, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for candidate in candidates[:limit]:
        clip_dir = os.path.join(frames_dir, candidate["id"])
        os.makedirs(clip_dir, exist_ok=True)
        if not os.listdir(clip_dir):
            middle = int((candidate.get("duration") or 600) / 2)
            section = f"*{middle}-{middle + 16}"
            result = subprocess.run(
                ["yt-dlp", "-f", "worst[ext=mp4]", "--download-sections", section,
                 "-o", os.path.join(clip_dir, "seg.mp4"), candidate["link"]],
                capture_output=True, text=True, timeout=300)
            seg = os.path.join(clip_dir, "seg.mp4")
            if result.returncode != 0 or not os.path.exists(seg):
                candidate["visual_category_sim"] = None
                continue
            subprocess.run(["ffmpeg", "-i", seg, "-vf", "fps=1", "-frames:v", "16",
                            os.path.join(clip_dir, "%02d.png"), "-loglevel", "quiet"],
                           capture_output=True, timeout=120)
        frames = sorted(p for p in os.listdir(clip_dir) if p.endswith(".png"))
        if not frames:
            candidate["visual_category_sim"] = None
            continue
        images = [Image.open(os.path.join(clip_dir, f)).convert("RGB") for f in frames]
        scores = head.score(images, stride=1)
        candidate["visual_category_sim"] = round(scores[goal_index], 4)


def _mine_category(name: str, entries: list[dict], source: str, top: int) -> dict:
    vocab = _CATEGORIES[name]
    candidates = _stage_titles(entries, vocab)
    print(f"\n{name} ({vocab['label']}): {len(candidates)} title candidates"
          f" >= score {_KEEP_TITLE_SCORE}")

    transcripts = _flag("--transcripts", None)
    if transcripts:
        fetched = _stage_transcripts(candidates, transcripts, vocab)
        print(f"  transcript stage: {fetched}/{min(transcripts, len(candidates))} transcripts"
              f" fetched (missing ones keep their title score — 2022 index, partial coverage)")
    visual = _flag("--visual", None)
    if visual:
        _stage_visual(candidates, visual, name)

    kept = []
    for candidate in candidates[:top]:
        hits = candidate.get("transcript_hits_per_min")
        if hits is not None and hits < _TRANSCRIPT_KEEP:
            continue                            # transcript contradicts the title
        kept.append(candidate)

    subset = {"scope": _SCOPE, "category": name, "label": vocab["label"],
              "definition": vocab["definition"], "source": source,
              "scoring": {"strong_phrases": vocab["strong"],
                          "keep_title_score": _KEEP_TITLE_SCORE,
                          "transcript_keep_hits_per_min": _TRANSCRIPT_KEEP},
              "clips": kept}
    subset_path = os.path.join(_OUT, f"mined_{name}.json")
    with open(subset_path, "w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2)

    distribution: dict[str, int] = {}
    for candidate in candidates:
        key = str(candidate["title_score"])
        distribution[key] = distribution.get(key, 0) + 1

    print(f"  kept {len(kept)} clips -> {os.path.relpath(subset_path, _ROOT)}")
    print("  sample sheet (top 10 - spot-check these by eye):")
    for candidate in kept[:10]:
        extra = ""
        if candidate.get("transcript_hits_per_min") is not None:
            extra += f"  hits/min {candidate['transcript_hits_per_min']}"
        if candidate.get("visual_category_sim") is not None:
            extra += f"  clip-sim {candidate['visual_category_sim']}"
        print(f"    [{candidate['title_score']:>2}] {candidate['title'][:70]}"
              f"  {candidate['link']}{extra}")
    return {"title_candidates": len(candidates), "kept": len(kept),
            "score_distribution": distribution}


def main() -> int:
    os.makedirs(_OUT, exist_ok=True)
    top = _flag("--top", 200)
    chosen = "habitation"
    if "--category" in sys.argv:
        chosen = sys.argv[sys.argv.index("--category") + 1]
    names = list(_CATEGORIES) if chosen == "all" else [c.strip() for c in chosen.split(",")]
    unknown = [n for n in names if n not in _CATEGORIES]
    if unknown:
        print(f"unknown category {unknown}; choose from {list(_CATEGORIES)} or 'all'")
        return 1

    entries = _fetch_index("youtube_tutorial.json")
    source = "youtube_tutorial.json"
    if "--full" in sys.argv:
        entries = entries + _fetch_index("youtube_full.json")
        source += " + youtube_full.json"
    print(f"index: {len(entries)} videos ({source})")
    print(f"  {_SCOPE}")

    report = {"scope": _SCOPE, "index_videos": len(entries), "categories": {}}
    for name in names:
        report["categories"][name] = _mine_category(name, entries, source, top)
    with open(os.path.join(_OUT, "mining_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
