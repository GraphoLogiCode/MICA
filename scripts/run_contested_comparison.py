"""The pre-registered contested-pairs comparison — one command, regenerable (D0 style).

    python scripts/run_contested_comparison.py [--skip-pairs] [--skip-train] [--no-vlm]

Question (vault note "2026-07-12 - Contested-Pairs Comparison"): does training the
likelihood heads on matcher-CONTESTED sessions' pairs (builder's label as truth) help
or hurt, measured on a frozen shared eval set neither arm trained on?

Chain:
  1. label_finished_builds.py --contested-pairs  (writes *.source_b_contested.jsonl)
  2. train the CONTROL arm   -> models/heads_cmp_control.*   (agreed pairs only)
     train the CONTESTED arm -> models/heads_cmp_contested.* (+ all contested pairs)
     Both reserve the same eval sessions OUT of train AND validation; production
     heads_v1.* is untouched (checksum-asserted).
  3. replay the filter (fused + 3D-only + 2D-only) over the shared eval sessions with
     each arm's heads + its jointly-fitted knobs; print the table + the
     pre-registered verdict; write capture/raw/contested_comparison.json.

The decision rule is pre-registered in the vault note and printed with the result.
RULE v2 (re-registered 2026-07-19, before any v2 run — the 07-12 verdict's inputs
were all repaired since: idle leak, corpus regeneration, both arms at ECE ~0.45):
the contested arm "wins" only if ALL FOUR hold —
  1. all-eval final accuracy STRICTLY above control's (the off-template gain the
     selection-bias hypothesis predicts; a tie is not a win),
  2. clean-eval final accuracy >= control's,
  3. pooled ECE <= control's + 0.02,
  4. mean sustained-from <= control's + 0.02 (earliness not materially worse).
A win does not change the pipeline — that would be a separate dated D3 amendment
for the user. No contested-session selection of any kind (VLM-based selection is
prohibited: the VLM is pinned diagnostic-only, 2026-07-19).
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from calibration_report import _reliability                        # noqa: E402
from run_tracker import _sustained_from                            # noqa: E402
from mica.capture import session_store                             # noqa: E402
from mica.contracts.b1 import GOALS, MacroAction                   # noqa: E402
from mica.contracts.b3 import fuse_streams                         # noqa: E402
from mica.intent.heads_v0 import strip_behavior, strip_structure   # noqa: E402
from mica.intent.heads_v1 import TrainedHeads                      # noqa: E402
from mica.intent.tracker import (                                  # noqa: E402
    category_marginal, correct, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
_RAW = session_store.RAW_ROOT
_REPORT = os.path.join(_ROOT, "capture", "scripted", "source_b_report.json")
_OUT = os.path.join(_RAW, "contested_comparison.json")

# The frozen eval design (vault note, registered 2026-07-12 BEFORE the run).
RESERVED = ("fabric-20260712-030849",   # infrastructure (clean)
            "fabric-20260712-025354",   # production (clean)
            "fabric-20260705-002717")   # habitation (clean)
PINNED_HOLDOUT_REAL = "fabric-20260705-134615"   # decorative (clean)
ECE_TOLERANCE = 0.02
SUSTAINED_TOLERANCE = 0.02          # rule v2: earliness may not materially regress

ARMS = {"control": [], "contested": ["--include-contested-pairs"]}


def _sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _run(script: str, *args: str) -> None:
    code = subprocess.run([sys.executable, os.path.join(_SCRIPTS, script), *args]).returncode
    if code != 0:
        raise SystemExit(f"{script} exited {code} — comparison halted")


def _quarantined(sid: str) -> bool:
    """The quarantine rule, applied here like everywhere else: a structure-
    quarantined session's banked evidence is untrustworthy (210003's truncated
    bank was invisible under the old silent zip-truncation and crashes honestly
    under the hardened fuse — either way it must not be evaluated)."""
    path = os.path.join(session_store.session_dir(sid), "session_report.json")
    try:
        with open(path, encoding="utf-8") as handle:
            return bool(json.load(handle).get("structure_quarantined"))
    except (OSError, ValueError):
        return False


def _eval_sessions() -> dict[str, dict]:
    """The frozen shared eval set: the 4 clean sessions + every discarded one.
    Truth is always the BUILDER's category. Neither arm has pairs for any of these
    (clean ones are reserved/pinned out; discarded never produce pairs)."""
    with open(os.path.join(_RAW, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    with open(_REPORT, encoding="utf-8") as handle:
        rows = {r["session"]: r for r in json.load(handle)["real"]["labeled"]}
    chosen = {}
    for sid in (PINNED_HOLDOUT_REAL, *RESERVED):
        chosen[sid] = {"truth": labels[sid]["goal"], "kind": "clean"}
    for sid, row in sorted(rows.items()):
        if row.get("label") and not row["label"].get("kept"):
            if _quarantined(sid):
                print(f"  eval set: {sid} skipped (structure-quarantined bank)")
                continue
            chosen[sid] = {"truth": labels[sid]["goal"], "kind": "discarded"}
    return chosen


def _records(sid: str) -> list:
    b1_path = session_store.session_file(sid, ".evidence2d.jsonl")
    b2_path = session_store.session_file(sid, ".evidence3d.jsonl")
    b1 = [json.loads(line) for line in open(b1_path, encoding="utf-8") if line.strip()]
    b2 = [json.loads(line) for line in open(b2_path, encoding="utf-8") if line.strip()]
    return fuse_streams(b1, b2, sid)


def _eval_arm(arm: TrainedHeads, sessions: dict[str, dict]) -> dict:
    """Every pre-registered metric for one arm over the shared eval set."""
    params = arm.tracker_params()
    pooled, finals_clean, finals_all = [], [], []
    margins, sustained = [], []
    overpowered = corrections = 0
    per_category: dict[str, list[bool]] = {}
    for sid, meta in sessions.items():
        records = _records(sid)
        streams = {"fused": records,
                   "no_behavior": [strip_behavior(f) for f in records],
                   "no_structure": [strip_structure(f) for f in records]}
        beliefs = {name: uniform_belief() for name in streams}
        previous_tick, calls = 0, []
        for index in range(len(records)):
            dt = max(records[index].tick - previous_tick, 1) / 20.0
            previous_tick = records[index].tick
            tops = {}
            for name, stream in streams.items():
                fused = stream[index]
                beliefs[name] = predict(beliefs[name], dt, params)
                table = arm.likelihood(fused, fused.a_hat)
                if name == "fused":
                    # 12-F8 mode margin at choice points, belief-weighted (the
                    # calibration report's definition, computed pre-correction).
                    if fused.a_hat == MacroAction.PLACE:
                        delib_mass = sum(beliefs[name][(g, 0)] for g in GOALS)
                        if delib_mass > 0:
                            weighted = sum(beliefs[name][(g, 0)] * table[(g, 0)]
                                           for g in GOALS) / delib_mass
                            margins.append(weighted - table[(GOALS[0], 1)])
                beliefs[name], _ = correct(beliefs[name], table, params)
                marginal = category_marginal(beliefs[name])
                tops[name] = max(marginal, key=marginal.get)
            corrections += 1
            if (tops["fused"] == "defense"
                    and tops["no_behavior"] != "defense" and tops["no_structure"] != "defense"):
                overpowered += 1
            marginal = category_marginal(beliefs["fused"])
            calls.append(tops["fused"])
            pooled.append((marginal[tops["fused"]], tops["fused"] == meta["truth"]))
        final_ok = calls[-1] == meta["truth"]
        finals_all.append(final_ok)
        if meta["kind"] == "clean":
            finals_clean.append(final_ok)
        sustained.append(_sustained_from(calls, meta["truth"]))
        per_category.setdefault(meta["truth"], []).append(final_ok)
    reliability = _reliability(pooled)
    return {
        "clean_final_accuracy": round(sum(finals_clean) / len(finals_clean), 3),
        "all_final_accuracy": round(sum(finals_all) / len(finals_all), 3),
        "mean_sustained_from": round(statistics.mean(sustained), 3),
        "pooled_ece": reliability["ece"],
        "points": reliability["points"],
        "mode_margin": round(statistics.mean(margins), 4) if margins else None,
        "defense_overpowered_rate": round(overpowered / corrections, 4),
        "per_category_final": {goal: f"{sum(oks)}/{len(oks)}"
                               for goal, oks in sorted(per_category.items())},
    }


def main() -> int:
    no_vlm = ["--no-vlm"] if "--no-vlm" in sys.argv else []
    guarded = {name: _sha(os.path.join(_ROOT, "models", name))
               for name in ("heads_v1.npz", "heads_v1.json")}

    if "--skip-pairs" not in sys.argv:
        print("=== step 1: matcher run with --contested-pairs ===")
        _run("label_finished_builds.py", "--contested-pairs", *no_vlm)
    if "--skip-train" not in sys.argv:
        for arm_name, extra in ARMS.items():
            print(f"\n=== step 2: training the {arm_name.upper()} arm ===")
            _run("train_heads.py", "--out-name", f"heads_cmp_{arm_name}",
                 "--extra-holdout", ",".join(RESERVED), *extra)

    for name, digest in guarded.items():
        assert _sha(os.path.join(_ROOT, "models", name)) == digest, \
            f"PRODUCTION MODEL {name} CHANGED — comparison is invalid, investigate"
    print("\nproduction heads_v1.* checksums unchanged — arms trained to their own files")

    sessions = _eval_sessions()
    print(f"\n=== step 3: shared eval over {len(sessions)} sessions "
          f"({sum(1 for m in sessions.values() if m['kind'] == 'clean')} clean + "
          f"{sum(1 for m in sessions.values() if m['kind'] == 'discarded')} discarded) ===")
    results = {}
    for arm_name in ARMS:
        results[arm_name] = _eval_arm(TrainedHeads(f"heads_cmp_{arm_name}"), sessions)

    control, contested = results["control"], results["contested"]
    verdict = (contested["all_final_accuracy"] > control["all_final_accuracy"]
               and contested["clean_final_accuracy"] >= control["clean_final_accuracy"]
               and contested["pooled_ece"] <= control["pooled_ece"] + ECE_TOLERANCE
               and contested["mean_sustained_from"]
                   <= control["mean_sustained_from"] + SUSTAINED_TOLERANCE)
    report = {"eval_sessions": sessions, "arms": results,
              "decision_rule": {"version": 2, "registered": "2026-07-19",
                                "all_accuracy_strictly_above_control": True,
                                "clean_accuracy_at_least_control": True,
                                "ece_tolerance": ECE_TOLERANCE,
                                "sustained_tolerance": SUSTAINED_TOLERANCE},
              "verdict": ("CONTESTED-INCLUSION WINS (rule v2) — worth a D3 "
                          "amendment decision"
                          if verdict else "withholding rule STANDS")}
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    width = max(len(k) for k in control)
    print(f"\n{'metric':<{width}}  {'control':>10}  {'contested':>10}")
    for key in control:
        if key == "per_category_final":
            continue
        print(f"{key:<{width}}  {str(control[key]):>10}  {str(contested[key]):>10}")
    print(f"per-category final: control {control['per_category_final']}")
    print(f"                  contested {contested['per_category_final']}")
    print(f"\nPRE-REGISTERED VERDICT: {report['verdict']}")
    print(f"report -> {os.path.relpath(_OUT, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
