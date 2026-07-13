"""The per-session intent + assist report, and the defense-bias diagnostic (D7 §5/§7).

    python scripts/run_intent_assist_report.py            # trained v1 heads (default)
    python scripts/run_intent_assist_report.py --heads v0

For every labeled real session this prints and banks, in one row: what the belief
predicted and how confidently, the final v3 label, the declared target (when the
builder recorded one), the inventory evidence available, whether DEFENSE overpowered
the single streams, and the material story — can MICA help, what is missing, what it
would gather. Everything comes from banked logs; one command regenerates it all.

The defense-overpower flag (D7 §5, both reads per review F3):
  posterior level    fused belief's top is defense while BOTH single-stream filters
                     (3D-only, 2D-only) put their top elsewhere at that correction
  likelihood level   same test on the instantaneous likelihood argmax per stream —
                     separates "fusion dynamics created defense" from "one stream's
                     likelihood shape keeps voting defense"

Honesty notes printed with the numbers: pre-2026-07-10 sessions carry no inventory
(the mod did not record it); offline there is no agent inventory, so can_help reads
false-with-reason, not a guess; a declared target never touches the filter — it is
compared AGAINST the belief, which is the point.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from run_tracker import real_labeled_sessions                                  # noqa: E402
from mica.assist import sufficiency as assist                                  # noqa: E402
from mica.contracts.b1 import GOALS                                            # noqa: E402
from mica.contracts.b3 import fuse_streams                                     # noqa: E402
from mica.intent import heads_v0                                               # noqa: E402
from mica.intent.heads_v0 import strip_behavior, strip_structure               # noqa: E402
from mica.intent.tracker import (                                              # noqa: E402
    TrackerParams, category_marginal, correct, predict, uniform_belief,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RAW = os.path.join(_ROOT, "capture", "raw")
_OUT = os.path.join(_RAW, "intent_assist_report.json")


def _load_lines(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fused_records(meta: dict, session_id: str = "?") -> list:
    return fuse_streams(_load_lines(meta["b1"]), _load_lines(meta["b2"]), session_id)


def _likelihood_top(likelihood, fused) -> str:
    """The instantaneous read: which category this one observation's likelihood
    votes for, both modes pooled — no history, no dynamics."""
    table = likelihood(fused, fused.a_hat)
    return max(GOALS, key=lambda g: table[(g, 0)] + table[(g, 1)])


def _session_row(sid: str, meta: dict, declared: dict | None, label: dict,
                 likelihood, params: TrackerParams) -> dict:
    records = _fused_records(meta, sid)
    variants = {"fused": records,
                "no_behavior": [strip_behavior(f) for f in records],
                "no_structure": [strip_structure(f) for f in records]}
    beliefs = {name: uniform_belief() for name in variants}
    previous_tick = 0
    overpowered_posterior = overpowered_likelihood = 0
    tops = {}
    for index in range(len(records)):
        dt = max(records[index].tick - previous_tick, 1) / 20.0
        previous_tick = records[index].tick
        for name, stream in variants.items():
            fused = stream[index]
            beliefs[name] = predict(beliefs[name], dt, params)
            beliefs[name], _ = correct(beliefs[name], likelihood(fused, fused.a_hat), params)
            marginal = category_marginal(beliefs[name])
            tops[name] = max(marginal, key=marginal.get)
        if (tops["fused"] == "defense"
                and tops["no_behavior"] != "defense" and tops["no_structure"] != "defense"):
            overpowered_posterior += 1
        if (tops["fused"] == "defense"
                and _likelihood_top(likelihood, variants["no_behavior"][index]) != "defense"
                and _likelihood_top(likelihood, variants["no_structure"][index]) != "defense"):
            overpowered_likelihood += 1

    marginal = category_marginal(beliefs["fused"])
    predicted = max(marginal, key=marginal.get)
    truth = meta["goal"]

    # The material story: declared target first, else the belief winner's style read.
    if declared:
        target, target_source = declared.get("subtype"), "declared"
    else:
        target, target_source = records[-1].per_goal[predicted].subtype, "inferred"
    requirements = assist.template_requirements(target)
    if requirements is None:
        # Banked evidence from before taxonomy v3 carries INSTANCE names as style
        # reads ("flat span"); resolve those to their v3 subtype so the material
        # math still applies. A true definitions-only subtype stays None — honest.
        from mica.perception.templates import instance
        try:
            target = instance(target).subtype
            requirements = assist.template_requirements(target)
        except KeyError:
            pass
    materials = assist.sufficiency(requirements,
                                   records[-1].state_feats.inventory, None)

    inventory = records[-1].state_feats.inventory
    return {
        "session": sid,
        "predicted_intent": predicted,
        "confidence": round(marginal[predicted], 4),
        "truth": truth,
        "final_label": {"goal": label.get("goal"), "subtype": label.get("subtype")},
        "declared_target": declared,
        "belief_vs_declaration": (None if not declared else
                                  "agrees" if declared.get("goal") == predicted else
                                  "DISAGREES"),
        "inventory_evidence": ({"items": len(inventory),
                                "total": sum(c for _, c in inventory)}
                               if inventory is not None else
                               "not captured (pre-2026-07-10 session)"),
        "defense_overpowered": {
            "corrections": len(records),
            "posterior_level": overpowered_posterior,
            "likelihood_level": overpowered_likelihood,
            "rate": round(overpowered_posterior / len(records), 4) if records else 0.0,
            "final_was_defense_error": predicted == "defense" and truth != "defense",
        },
        "assist": (None if materials is None else {
            "target": target, "target_source": target_source,
            "can_help": materials["can_help"],
            "agent_note": "no agent inventory offline — can_help is agent-side only",
            "missing": materials["missing"],
            "gather_next": materials["gather_next"],
        }) or {"target": target, "target_source": target_source,
               "note": "definitions-only subtype — no template to measure against"},
    }


def main() -> int:
    if "--heads" in sys.argv and sys.argv[sys.argv.index("--heads") + 1] == "v0":
        likelihood, params, heads = heads_v0.likelihood, TrackerParams(), "hand-coded v0"
    else:
        from mica.intent import heads_v1
        likelihood, params, heads = heads_v1.likelihood, heads_v1.tracker_params(), "trained v1"

    labels_meta = real_labeled_sessions()
    if not labels_meta:
        print("no labeled real captures with banked evidence found")
        return 1
    with open(os.path.join(_RAW, "labels.json"), encoding="utf-8") as handle:
        labels = json.load(handle)
    try:
        with open(os.path.join(_RAW, "declared_targets.json"), encoding="utf-8") as handle:
            declared_all = json.load(handle)
    except (OSError, ValueError):
        declared_all = {}

    rows = [_session_row(sid, meta, declared_all.get(sid), labels.get(sid, {}),
                         likelihood, params)
            for sid, meta in sorted(labels_meta.items())]

    total = sum(r["defense_overpowered"]["corrections"] for r in rows)
    posterior = sum(r["defense_overpowered"]["posterior_level"] for r in rows)
    lik = sum(r["defense_overpowered"]["likelihood_level"] for r in rows)
    by_truth: dict[str, list[float]] = {}
    for row in rows:
        by_truth.setdefault(row["truth"], []).append(row["defense_overpowered"]["rate"])
    report = {
        "heads": heads,
        "sessions": rows,
        "defense_bias_aggregate": {
            "corrections_total": total,
            "overpowered_posterior": posterior,
            "overpowered_posterior_rate": round(posterior / total, 4) if total else 0.0,
            "overpowered_likelihood": lik,
            "sessions_affected": sum(1 for r in rows
                                     if r["defense_overpowered"]["posterior_level"]),
            "final_defense_errors": sum(1 for r in rows
                                        if r["defense_overpowered"]["final_was_defense_error"]),
            "rate_by_true_category": {g: round(sum(v) / len(v), 4)
                                      for g, v in sorted(by_truth.items())},
        },
    }
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print(f"intent + assist report ({heads}) over {len(rows)} real sessions "
          f"-> {os.path.relpath(_OUT, _ROOT)}")
    for row in rows:
        d = row["defense_overpowered"]
        assist_note = row["assist"].get("missing") if row["assist"] else None
        print(f"  {row['session']}: predicted {row['predicted_intent']} "
              f"({row['confidence']:.2f}) vs truth {row['truth']}"
              + (f" · declared {row['declared_target']['goal']}/"
                 f"{row['declared_target']['subtype']} ({row['belief_vs_declaration']})"
                 if row["declared_target"] else "")
              + f" · defense-overpowered {d['posterior_level']}/{d['corrections']}"
              + (f" · missing {assist_note}" if assist_note else ""))
    agg = report["defense_bias_aggregate"]
    print(f"  DEFENSE BIAS: {agg['overpowered_posterior']}/{agg['corrections_total']} "
          f"corrections ({agg['overpowered_posterior_rate']:.1%}) posterior-level, "
          f"{agg['overpowered_likelihood']} likelihood-level, "
          f"{agg['sessions_affected']} sessions affected, "
          f"{agg['final_defense_errors']} final defense errors")
    print(f"  by true category: {agg['rate_by_true_category']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
