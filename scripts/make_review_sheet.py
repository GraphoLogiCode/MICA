"""Build the Source C manual-review sheet: key frames + evidence for human approval.

    python scripts/make_review_sheet.py [--clips N] [--category <name|all>]
        --clips     per-category clip budget (default 12)
        --category  which mined subset(s) to process (default: all mined_*.json present)

Takes the miner's kept clips (capture/youtube/mined_<category>.json — run
mine_minedojo_youtube.py first), and for each one prepares what a human needs to
JUDGE the label instead of trusting it:

  - 1-5 key frames, extracted at the moments the transcript talks about building
    (falling back to evenly spaced moments when the video has no captions),
  - the transcript windows that matched the house vocabulary (the evidence text),
  - the proposed label (HOUSE), the rule that produced it, and its heuristic score.

Everything lands in capture/youtube/review/ + review_items.json, which the review
server (scripts/review_house_labels.py) serves for the approve/reject pass. Nothing
from Source C is used downstream without that pass: the approved subset
(house_subset_approved.json) is written by the reviewer's decisions, not by this
script.

Videos download once at low resolution into capture/youtube/video/ (cached; reruns
are free). Frame extraction uses imageio-ffmpeg's bundled ffmpeg, so no system
install is needed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

# YouTube titles carry characters the Windows console's default codepage cannot
# print (a fullwidth bar killed the first batch run) — never let a print crash a build.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "youtube")
_REVIEW = os.path.join(_OUT, "review")
_VIDEO = os.path.join(_OUT, "video")
_TRANSCRIPTS = os.path.join(_OUT, "transcripts")
_SCOPE = ("Source C (D3): pixels-only analysis/pretraining corpus - never fused "
          "evidence, never Source B pairs, never head training data. Labels below "
          "are PROPOSED; only the human-approved subset is used downstream.")

_MAX_FRAMES = 4
_MIN_GAP_S = 12            # two key frames closer than this show the same moment
_EVIDENCE_WINDOWS = 3
_VOCAB = ("house", "home", "cabin", "cottage", "mansion", "base", "hut", "villa",
          "wall", "roof", "door", "floor", "window", "stairs", "foundation",
          "interior", "build", "building")
_STRONG_HINTS = ("build a", "building a", "tutorial", "how to build", "start with",
                 "foundation", "first layer")


def _flag(name: str, default: int) -> int:
    if name in sys.argv:
        return int(sys.argv[sys.argv.index(name) + 1])
    return default


def _ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def _segments(video_id: str) -> list[dict] | None:
    """Timed transcript segments [{start, text}], cached; None when unavailable."""
    cached = os.path.join(_TRANSCRIPTS, f"{video_id}.segments.json")
    if os.path.exists(cached):
        return json.load(open(cached, encoding="utf-8"))
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        fetched = YouTubeTranscriptApi().fetch(video_id)
        segments = [{"start": round(s.start, 1), "text": s.text} for s in fetched]
    except Exception:
        return None
    os.makedirs(_TRANSCRIPTS, exist_ok=True)
    with open(cached, "w", encoding="utf-8") as handle:
        json.dump(segments, handle)
    return segments


def _evidence_windows(segments: list[dict]) -> list[dict]:
    """The transcript moments that talk about building — scored, spread out.

    Auto-caption lines are only a few words each, so the score reads the line PLUS
    its neighbors (one caption line rarely holds a whole phrase), and a single
    vocabulary hit qualifies — the ranking still puts the strong phrases first."""
    scored = []
    for index, segment in enumerate(segments):
        text = " ".join(s["text"] for s in segments[max(0, index - 1):index + 2]).strip()
        lower = text.lower()
        score = sum(2 for hint in _STRONG_HINTS if hint in lower)
        score += sum(1 for word in _VOCAB if word in lower)
        if score >= 1:
            scored.append({"t": segment["start"], "text": text[:220], "score": score})
    scored.sort(key=lambda w: -w["score"])
    chosen: list[dict] = []
    for window in scored:
        if all(abs(window["t"] - kept["t"]) >= _MIN_GAP_S for kept in chosen):
            chosen.append(window)
        if len(chosen) >= _EVIDENCE_WINDOWS:
            break
    return sorted(chosen, key=lambda w: w["t"])


def _frame_times(duration: float, windows: list[dict]) -> list[float]:
    """Where to grab frames: at the evidence moments (+2 s so the caption's action is
    on screen), padded with evenly spaced moments; never the very start/end."""
    times = [min(max(w["t"] + 2.0, 5.0), duration - 5.0) for w in windows]
    for fraction in (0.25, 0.5, 0.75):
        candidate = duration * fraction
        if len(times) >= _MAX_FRAMES:
            break
        if all(abs(candidate - t) >= _MIN_GAP_S for t in times):
            times.append(candidate)
    return sorted(times)[:_MAX_FRAMES]


def _download(video_id: str, link: str) -> str | None:
    """One cached low-res mp4 per clip; frames come out of it locally."""
    path = os.path.join(_VIDEO, f"{video_id}.mp4")
    if os.path.exists(path):
        return path
    if shutil.which("yt-dlp") is None:
        return None
    os.makedirs(_VIDEO, exist_ok=True)
    result = subprocess.run(
        ["yt-dlp", "-f", "18/worst[ext=mp4]", "--max-filesize", "150M",
         "-o", path, link],
        capture_output=True, text=True, timeout=600)
    return path if result.returncode == 0 and os.path.exists(path) else None


def _extract_frames(ffmpeg: str, video: str, video_id: str, times: list[float]) -> list[str]:
    clip_dir = os.path.join(_REVIEW, video_id)
    os.makedirs(clip_dir, exist_ok=True)
    files = []
    for index, t in enumerate(times):
        out = os.path.join(clip_dir, f"f{index:02d}.jpg")
        if not os.path.exists(out):
            subprocess.run([ffmpeg, "-ss", str(t), "-i", video, "-frames:v", "1",
                            "-q:v", "3", "-y", out],
                           capture_output=True, timeout=120)
        if os.path.exists(out):
            files.append(f"{video_id}/f{index:02d}.jpg")
    return files


def _build_item(clip: dict, subset: dict, ffmpeg: str) -> dict | None:
    video_id = clip["id"]
    print(f"  {video_id}  {clip['title'][:60]}")
    segments = _segments(video_id)
    windows = _evidence_windows(segments) if segments else []
    duration = clip.get("duration") or 600
    times = _frame_times(duration, windows)
    video = _download(video_id, clip["link"])
    if video is None:
        print("    video unavailable - skipped")
        return None
    frames = _extract_frames(ffmpeg, video, video_id, times)
    if not frames:
        print("    no frames extracted - skipped")
        return None
    hits = clip.get("transcript_hits_per_min")
    rule = f"title matched {clip.get('matched', [])} (score {clip['title_score']})"
    rule += (f"; transcript {hits} build-words/min" if hits is not None
             else "; no transcript (title only)")
    return {
        "id": video_id,
        "title": clip["title"],
        "link": clip["link"],
        "duration": duration,
        "category": subset["category"],
        "proposed_label": subset["label"],
        "definition": subset["definition"],
        "title_score": clip["title_score"],
        "confidence": round(min(1.0, clip["title_score"] / 15.0), 2),
        "rule": rule,
        "windows": windows,
        "frames": [{"file": file, "t": round(t, 1),
                    "window": next((w["text"] for w in windows
                                    if abs(w["t"] + 2.0 - t) < 1.0), None)}
                   for file, t in zip(frames, times)],
    }


def main() -> int:
    ffmpeg = _ffmpeg()
    if ffmpeg is None:
        print("no ffmpeg available - pip install imageio-ffmpeg")
        return 1
    limit = _flag("--clips", 12)
    chosen = None
    if "--category" in sys.argv:
        chosen = sys.argv[sys.argv.index("--category") + 1]
    subset_paths = (sorted(glob.glob(os.path.join(_OUT, "mined_*.json")))
                    if chosen in (None, "all")
                    else [os.path.join(_OUT, f"mined_{chosen}.json")])
    subset_paths = [p for p in subset_paths if os.path.exists(p)]
    if not subset_paths:
        print("no mined subset - run scripts/mine_minedojo_youtube.py first")
        return 1

    # The sheet is MERGED, never wholesale-replaced: items already reviewed keep
    # their place (a video retrieved by two categories keeps its higher-scored one,
    # unless a verdict already exists for it — decided items are never re-labeled).
    out = os.path.join(_OUT, "review_items.json")
    existing = (json.load(open(out, encoding="utf-8"))["items"]
                if os.path.exists(out) else [])
    items = {item["id"]: item for item in existing}

    def _save() -> None:
        # Written after every new clip so an interruption keeps its progress
        # (frame extraction over a big batch is long; losing it all is not ok).
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"scope": _SCOPE, "built_ms": int(time.time() * 1000),
                       "items": list(items.values())}, handle, indent=2)
        os.replace(tmp, out)

    for subset_path in subset_paths:
        subset = json.load(open(subset_path, encoding="utf-8"))
        print(f"{subset['category']} ({subset['label']}):")
        built = 0
        for clip in subset["clips"]:
            if built >= limit:
                break
            current = items.get(clip["id"])
            if current is not None:
                built += current.get("category", "habitation") == subset["category"]
                continue                        # already on the sheet (maybe decided)
            item = _build_item(clip, subset, ffmpeg)
            if item is not None:
                items[item["id"]] = item
                built += 1
                _save()

    _save()
    print(f"review sheet: {len(items)} clips -> {os.path.relpath(out, _ROOT)}")
    print("now run:  python scripts/review_house_labels.py   and approve/reject in the browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
