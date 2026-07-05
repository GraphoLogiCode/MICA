"""Tiered auto-labeling for Source C clips — automation with a standing human audit.

    python scripts/auto_label_clips.py [--audit-fraction 0.1]

Why tiered (and not fully automatic): the human's 5/5 agreement with the heuristic
came entirely from score-15 titles — the unambiguous top. The bulk of candidates
sit at scores 6-9 with no human data at all, and n=5 agreement bounds precision no
tighter than ~50% at 95% confidence. So no single signal is trusted alone:

  AUTO HOUSE      title heuristic says house (score >= TITLE_TRUST) AND the VLM,
                  shown the actual frames, independently says HOUSE
  AUTO NOT_HOUSE  the VLM says NOT_HOUSE and the title score is weak (< TITLE_STRONG)
  HUMAN QUEUE     everything else — the two signals disagree, the VLM is unsure,
                  or the clip was drawn as an AUDIT sample (a random slice of
                  would-be-auto clips always goes to the human, so auto precision
                  is MEASURED continuously, never assumed)

Auto decisions land in review_decisions.json with decided_by="auto" plus the VLM's
reason, and the review UI shows them with an AUTO badge — one click overrides them
(the override is recorded, so auto-vs-human agreement accumulates). The VLM also
proposes per-frame stage tags (EXTERIOR/WALL_BUILD/ROOF_BUILD/INTERIOR/OTHER),
saved as auto annotations for the reviewer to refine.

The model comes from mica/data/vlm.py: a LOCAL open model served by Ollama on this
machine's GPU when available (default qwen2.5vl:7b — no key, no cost, nothing
leaves the machine), else the Anthropic API when a key is set. With neither,
nothing is decided — every clip stays in the human queue and this script says so.

    --validate    decide NOTHING; instead run the VLM over every clip the human
                  already decided and grade it — the earn-its-place gate a model
                  must pass before its auto verdicts are trusted at all.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from mica.data import vlm  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "youtube")
_ITEMS = os.path.join(_OUT, "review_items.json")
_DECISIONS = os.path.join(_OUT, "review_decisions.json")
_REPORT = os.path.join(_OUT, "auto_label_report.json")
_REVIEW = os.path.join(_OUT, "review")

# The tier rule, pinned (echoed into the report):
TITLE_TRUST = 6     # heuristic floor for an auto-approve (below it, always human)
TITLE_STRONG = 10   # a title this strong is never auto-REJECTED on the VLM's word alone
# Audit slices are PER CATEGORY (2026-07-04 guardrail): trust is earned by graded
# agreement, separately per category. Current grades (n=46 overall at 93%):
# habitation 1/1 + 97% validation, production 10/10 -> 10%; infrastructure 11/12
# (92%) and defense 11/11 -> 25% (recovering/clean but single review burst, n
# small); decorative 10/12 (83%) -> stays 40% until it clears 90%.
# --audit-fraction N overrides all categories (for experiments).
AUDIT_FRACTIONS = {"habitation": 0.10, "production": 0.10,
                   "infrastructure": 0.25, "defense": 0.25, "decorative": 0.40}
AUDIT_DEFAULT = 0.25   # a category not listed above
AUDIT_MIN = 1          # per category with any would-be-auto clips
# INVENTORY_UI added 2026-07-04 (user finding): tutorial POV regularly shows the
# inventory/crafting screen, and a model without that option returned NOTHING for
# such frames — a missing tag instead of a tag. Every frame must get exactly one.
# Definitions made CONCRETE 2026-07-04 (user decision, after reviewing the audit
# cards): each stage is a statement about where the CAMERA is and which part of
# the build is being worked — visually decidable from one frame, no judgment calls.
_STAGES = ("EXTERIOR", "WALL_BUILD", "ROOF_BUILD", "INTERIOR", "INVENTORY_UI", "OTHER")

_STAGE_GUIDE = """Tag every numbered frame with exactly ONE stage. Decide in this order:
  INVENTORY_UI  a menu covers the view — inventory, crafting table, chest, settings.
                Check this FIRST: if a UI overlay dominates the frame, it wins.
  INTERIOR      the camera is INSIDE the build — walls, floor, or ceiling of the
                build surround the view.
  ROOF_BUILD    building on TOP of the build — roof or top-layer work, the camera
                at or above the build's upper edge.
  WALL_BUILD    working on the SIDE of the build — a wall face seen up close,
                blocks being placed against it.
  EXTERIOR      the camera is OUTSIDE the build looking at it — the structure (or
                most of it) visible from outside, not close up against one face.
  OTHER         ONLY when none of the above clearly fits — bare terrain, sky,
                title cards, scenes without the build.
Every frame MUST appear in "stages" — no omissions."""

_DEFAULT_DEFINITION = ("a dwelling — house, cabin, cottage, hut; not a farm, "
                       "statue, redstone machine, or pure terrain")

_PROMPT = """These are frames from a Minecraft video titled: "{title}"
{evidence}
Answer in JSON only, no other text, exactly this shape:
{{"verdict": "{label}" or "NOT_{label}" or "UNSURE",
  "stages": {{"1": "<stage>", "2": "<stage>", ...}},
  "reason": "<one short sentence>"}}
"verdict" says whether this video shows someone BUILDING {definition}.
{stage_guide}"""

_RETAG_PROMPT = """These are frames from a Minecraft video titled: "{title}"
Answer in JSON only, no other text: {{"1": "<stage>", "2": "<stage>", ...}}
{stage_guide}"""


def _vlm(item: dict) -> dict | None:
    paths = [os.path.join(_REVIEW, frame["file"]) for frame in item["frames"]]
    paths = [path for path in paths if os.path.exists(path)]
    if not paths:
        return None
    evidence = ""
    if item["windows"]:
        quotes = " / ".join(f'"{w["text"][:100]}"' for w in item["windows"][:2])
        evidence = f"The video's own transcript says: {quotes}"
    label = item.get("proposed_label", "HOUSE")
    prompt = _PROMPT.format(title=item["title"], evidence=evidence, label=label,
                            definition=item.get("definition", _DEFAULT_DEFINITION),
                            stage_guide=_STAGE_GUIDE)
    text = vlm.ask(paths, prompt, json_format=True)
    if text is None:
        return None
    try:
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, KeyError) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if "error" in parsed:
        return {"error": parsed["error"]}
    verdict = parsed.get("verdict")
    if verdict not in (label, f"NOT_{label}", "UNSURE"):
        verdict = "UNSURE"
    stages = {}
    for index, frame in enumerate(item["frames"], start=1):
        stage = parsed.get("stages", {}).get(str(index))
        if stage in _STAGES:
            stages[frame["file"]] = stage
    return {"verdict": verdict, "stages": stages,
            "reason": str(parsed.get("reason", ""))[:200], "model": vlm.describe()}


def _tag_single_frame(path: str) -> str | None:
    """One frame, one word — the fallback for frames the batch call skips (the
    local model sometimes enumerates fewer JSON entries than it was shown images)."""
    text = vlm.ask([path], "One frame from a Minecraft building video. Answer with"
                           f" EXACTLY one word from this list: {', '.join(_STAGES)}.\n"
                           + _STAGE_GUIDE)
    if text is None:
        return None
    upper = text.upper()
    return next((stage for stage in _STAGES if stage in upper), None)


def _retag(items: list[dict], decisions: dict) -> int:
    """Fill MISSING frame stage tags across the whole sheet — human tags and any
    existing auto tags stay untouched; only gaps get the model's answer. Run after
    a stage-scheme change (like adding INVENTORY_UI) so old clips catch up."""
    filled = asked = 0
    for item in items:
        entry = decisions.setdefault(item["id"], {})
        frames = entry.setdefault("frames", {})
        missing = [f for f in item["frames"] if f["file"] not in frames]
        if not missing:
            continue
        paths = [os.path.join(_REVIEW, f["file"]) for f in item["frames"]]
        if not all(os.path.exists(p) for p in paths):
            continue
        asked += 1
        text = vlm.ask(paths, _RETAG_PROMPT.format(title=item["title"],
                                                   stage_guide=_STAGE_GUIDE),
                       json_format=True)
        if text is None:
            continue
        try:
            parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except ValueError:
            parsed = {}
        for index, frame in enumerate(item["frames"], start=1):
            stage = parsed.get(str(index))
            if frame["file"] not in frames and stage in _STAGES:
                frames[frame["file"]] = stage
                filled += 1
        # whatever the batch call skipped gets asked one frame at a time
        for frame in item["frames"]:
            if frame["file"] in frames:
                continue
            stage = _tag_single_frame(os.path.join(_REVIEW, frame["file"]))
            if stage is not None:
                frames[frame["file"]] = stage
                filled += 1
    with open(_DECISIONS, "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import review_house_labels as review
    review._write_approved(review._load_items(), decisions)
    print(f"retag ({vlm.describe()}): {filled} missing frame tags filled"
          f" across {asked} clips; human tags untouched")
    return 0


def _restage(items: list[dict], decisions: dict) -> int:
    """Re-tag stages under the CURRENT definitions — after a scheme change.

    Which tags are redone: only the model's own. A tag is provably human when its
    clip has no recorded VLM proposal at all, or when it DIFFERS from what the
    model proposed (someone overrode it) — those are never touched. Tags equal to
    the model's old proposal, and missing tags, get fresh answers under the new
    guide. The clip's own VLM record is updated so this stays re-runnable."""
    redone = protected = 0
    for item in items:
        entry = decisions.setdefault(item["id"], {})
        frames = entry.setdefault("frames", {})
        vlm_stages = entry.get("vlm", {}).get("stages", {})
        redo_files = []
        for frame in item["frames"]:
            file = frame["file"]
            current = frames.get(file)
            if current is not None and ("vlm" not in entry or current != vlm_stages.get(file)):
                protected += 1                    # human-set: never touched
            else:
                redo_files.append(file)
        if not redo_files:
            continue
        paths = [os.path.join(_REVIEW, f["file"]) for f in item["frames"]]
        if not all(os.path.exists(p) for p in paths):
            continue
        text = vlm.ask(paths, _RETAG_PROMPT.format(title=item["title"],
                                                   stage_guide=_STAGE_GUIDE),
                       json_format=True)
        parsed = {}
        if text is not None:
            try:
                parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
            except ValueError:
                pass
        new_stages = {}
        for index, frame in enumerate(item["frames"], start=1):
            stage = parsed.get(str(index))
            if stage not in _STAGES and frame["file"] in redo_files:
                stage = _tag_single_frame(os.path.join(_REVIEW, frame["file"]))
            if stage in _STAGES:
                new_stages[frame["file"]] = stage
        for file in redo_files:
            if file in new_stages:
                frames[file] = new_stages[file]
                redone += 1
        if "vlm" in entry:                        # keep the provenance heuristic honest
            entry["vlm"]["stages"] = {f: s for f, s in new_stages.items()}
    with open(_DECISIONS, "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import review_house_labels as review
    review._write_approved(review._load_items(), decisions)
    print(f"restage ({vlm.describe()}): {redone} auto tags redone under the new"
          f" definitions; {protected} human-set tags untouched")
    return 0


def _validate(items: list[dict], decisions: dict) -> int:
    """Grade the current VLM against every clip a HUMAN decided — across all five
    categories, human labels only (grading the model against its own auto labels
    would be circular). No decision is touched: this is the gate a model passes
    before its verdicts are trusted."""
    decided = [item for item in items
               if decisions.get(item["id"], {}).get("label")
               and decisions[item["id"]].get("decided_by", "human") == "human"]
    if not decided:
        print("nothing to validate against - no human-decided clips yet")
        return 1
    print(f"validating {vlm.describe()} against {len(decided)} human-decided clips"
          " (all categories)")
    rows, verdict_hits = [], 0
    stage_hits = stage_total = 0
    per_cat: dict[str, list[int]] = {}
    for item in decided:
        human = decisions[item["id"]]
        cat = item.get("category", "habitation")
        result = _vlm(item)
        if result is None or "error" in result:
            rows.append({"id": item["id"], "category": cat, "human": human["label"],
                         "vlm": f"ERROR: {(result or {}).get('error', 'no backend')}"})
            continue
        agree = result["verdict"] == human["label"]
        verdict_hits += agree
        per_cat.setdefault(cat, [0, 0])
        per_cat[cat][0] += agree
        per_cat[cat][1] += 1
        human_tags = human.get("frames", {})
        for file, stage in result["stages"].items():
            if file in human_tags:
                stage_total += 1
                stage_hits += stage == human_tags[file]
        rows.append({"id": item["id"], "category": cat, "human": human["label"],
                     "vlm": result["verdict"], "agree": agree,
                     "reason": result["reason"]})
        mark = "ok " if agree else ">>>"
        print(f"  {mark} {cat[:5]:<5} {item['id']}  human {human['label']:<15}"
              f" vlm {result['verdict']:<15} {result['reason'][:55]}")
    graded = [row for row in rows if "agree" in row]
    agreement = sum(row["agree"] for row in graded) / len(graded) if graded else 0.0
    print("  per-category verdict agreement:")
    for cat in sorted(per_cat):
        ok, n = per_cat[cat]
        print(f"    {cat:<15} {ok}/{n}  ({ok/n:.0%})")
    stage_agreement = stage_hits / stage_total if stage_total else None
    validation = {"model": vlm.describe(), "clips": len(decided),
                  "verdict_agreement": round(agreement, 3),
                  "per_category": {c: {"ok": v[0], "n": v[1]} for c, v in per_cat.items()},
                  "stage_agreement": round(stage_agreement, 3) if stage_agreement is not None else None,
                  "stage_pairs_graded": stage_total, "rows": rows}
    report = json.load(open(_REPORT, encoding="utf-8")) if os.path.exists(_REPORT) else {}
    report["validation"] = validation
    with open(_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"  verdict agreement: {agreement:.0%} over {len(graded)} clips"
          + (f"; frame-stage agreement {stage_agreement:.0%} over {stage_total} tagged frames"
             if stage_agreement is not None else ""))
    verdict_ok = agreement >= 0.9
    print(f"  [{'PASS' if verdict_ok else 'FAIL'}] earn-its-place bar (>= 90% verdict agreement)"
          f" -> report['validation'] in {os.path.basename(_REPORT)}")
    return 0 if verdict_ok else 1


def main() -> int:
    if "--audit-fraction" in sys.argv:
        override = float(sys.argv[sys.argv.index("--audit-fraction") + 1])
        for category in list(AUDIT_FRACTIONS):
            AUDIT_FRACTIONS[category] = override
        globals()["AUDIT_DEFAULT"] = override
    if not os.path.exists(_ITEMS):
        print("no review sheet - run scripts/make_review_sheet.py first")
        return 1
    if vlm.backend() == "none":
        print("no VLM available - the tier cannot run, so NOTHING was auto-decided.\n"
              "Either start the local server (ollama serve; model qwen2.5vl:7b) or set"
              " ANTHROPIC_API_KEY, then rerun.\nEvery clip stays in the human review"
              " queue (scripts/review_house_labels.py).")
        return 1
    items = json.load(open(_ITEMS, encoding="utf-8"))["items"]
    decisions = json.load(open(_DECISIONS, encoding="utf-8")) if os.path.exists(_DECISIONS) else {}
    if "--validate" in sys.argv:
        return _validate(items, decisions)
    if "--retag" in sys.argv:
        return _retag(items, decisions)
    if "--restage" in sys.argv:
        return _restage(items, decisions)
    print(f"model: {vlm.describe()}")

    # Pending = no verdict AND not reserved as an audit sample — audit clips are the
    # human's to decide (their verdict grades the auto tier; re-deciding them here
    # would grade the machine against itself).
    pending = [item for item in items
               if not decisions.get(item["id"], {}).get("label")
               and not decisions.get(item["id"], {}).get("audit_of_auto")]
    print(f"auto-labeling {len(pending)} undecided clips (of {len(items)} on the sheet)")
    rng = random.Random(7)
    counts = {"auto_approved": 0, "auto_rejected": 0, "queued": 0, "audit": 0, "vlm_error": 0}
    would_auto: list[tuple[dict, dict, str]] = []

    for item in pending:
        verdict = _vlm(item)
        if verdict is None or "error" in verdict:
            counts["vlm_error"] += verdict is not None
            counts["queued"] += 1
            continue
        entry = decisions.setdefault(item["id"], {})
        entry["vlm"] = verdict                       # always shown in the UI as the hint
        label = item.get("proposed_label", "HOUSE")
        title_score = item.get("title_score", 0)
        if verdict["verdict"] == label and title_score >= TITLE_TRUST:
            would_auto.append((item, entry, label))
        elif verdict["verdict"] == f"NOT_{label}" and title_score < TITLE_STRONG:
            would_auto.append((item, entry, f"NOT_{label}"))
        else:
            counts["queued"] += 1                    # disagreement or UNSURE: human decides

    # The audit slice: a random sample of would-be-auto clips goes to the human
    # UNDECIDED, with the auto verdict recorded as the prediction to grade.
    # Drawn PER CATEGORY at that category's fraction — trust is earned separately.
    by_category: dict[str, list] = {}
    for pair in would_auto:
        by_category.setdefault(pair[0].get("category", "habitation"), []).append(pair)
    audit_picks: set[int] = set()
    audit_by_category: dict[str, int] = {}
    for category, pairs in by_category.items():
        fraction = AUDIT_FRACTIONS.get(category, AUDIT_DEFAULT)
        count = min(max(AUDIT_MIN, round(len(pairs) * fraction)), len(pairs))
        audit_by_category[category] = count
        audit_picks.update(id(pair) for pair in rng.sample(pairs, count))
    for pair in would_auto:
        item, entry, label = pair
        if id(pair) in audit_picks:
            entry["audit_of_auto"] = label           # human will decide; agreement measured
            counts["audit"] += 1
            counts["queued"] += 1
            continue
        entry["label"] = label
        entry["decided_by"] = "auto"
        entry["ts"] = int(time.time() * 1000)
        frames = entry.setdefault("frames", {})
        for file, stage in entry["vlm"]["stages"].items():
            frames.setdefault(file, stage)           # auto tags never overwrite human ones
        counts["auto_rejected" if label.startswith("NOT_") else "auto_approved"] += 1

    with open(_DECISIONS, "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    # Rebuild the approved subset through the review server's own writer.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import review_house_labels as review
    review._write_approved(review._load_items(), decisions)

    # Agreement so far: audited/overridden auto verdicts graded against the human.
    graded = [(d.get("audit_of_auto") or d.get("overridden_auto"), d["label"])
              for d in decisions.values()
              if d.get("label") and (d.get("audit_of_auto") or d.get("overridden_auto"))]
    agreement = (sum(1 for auto, human in graded if auto == human) / len(graded)
                 if graded else None)
    report = {"rule": {"title_trust": TITLE_TRUST, "title_strong": TITLE_STRONG,
                       "audit_fractions": AUDIT_FRACTIONS, "audit_default": AUDIT_DEFAULT,
                       "model": vlm.describe()},
              "counts": counts, "audits_drawn_by_category": audit_by_category,
              "graded_audits": len(graded),
              "auto_human_agreement": agreement}
    with open(_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"  auto-approved {counts['auto_approved']}, auto-rejected {counts['auto_rejected']},"
          f" human queue {counts['queued']} (of which {counts['audit']} audit samples)"
          + (f", vlm errors {counts['vlm_error']}" if counts["vlm_error"] else ""))
    if agreement is not None:
        print(f"  auto-vs-human agreement so far: {agreement:.0%} over {len(graded)} graded")
        if agreement < 0.9:
            print("  !! agreement below 90% - tighten the rule (raise TITLE_TRUST) before"
                  " trusting further auto batches !!")
    print(f"  report -> {os.path.relpath(_REPORT, _ROOT)};"
          " review the queue at scripts/review_house_labels.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
