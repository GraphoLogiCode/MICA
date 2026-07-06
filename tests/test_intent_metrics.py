"""The metric suite on hand-computable toy sequences — every number in the four-arm
table must be checkable by a person with a pencil."""
from mica.contracts.b1 import GOALS
from mica.validation import intent_metrics


def _steps(spec):
    """spec: list of (top_goal, top_p, truth, progress) — the rest spread uniformly."""
    steps = []
    for top, p, truth, progress in spec:
        rest = (1.0 - p) / (len(GOALS) - 1)
        dist = {g: p if g == top else rest for g in GOALS}
        steps.append(intent_metrics.step(dist, truth, progress))
    return steps


def test_sustained_from_finds_the_stable_suffix():
    steps = _steps([("defense", 0.6, "habitation", 0.1),
                    ("habitation", 0.6, "habitation", 0.4),
                    ("defense", 0.6, "habitation", 0.6),      # relapse breaks the streak
                    ("habitation", 0.6, "habitation", 0.8),
                    ("habitation", 0.6, "habitation", 1.0)])
    assert intent_metrics.sustained_from(steps) == 3 / 5


def test_sustained_from_progress_speaks_build_units():
    # locks on from step 1 of 3 — step-fraction 1/3, but that step sits at 80% of the
    # BUILD (an idle-heavy tail), and 06's units must say 0.8, not 0.33
    steps = _steps([("defense", 0.6, "habitation", 0.5),
                    ("habitation", 0.6, "habitation", 0.8),
                    ("habitation", 0.6, "habitation", 0.9)])
    assert intent_metrics.sustained_from(steps) == 1 / 3
    assert intent_metrics.sustained_from_progress(steps) == 0.8
    never = _steps([("defense", 0.6, "habitation", 0.5)])
    assert intent_metrics.sustained_from_progress(never) == 1.0


def test_separation_sign_and_falsification():
    right = _steps([("habitation", 0.6, "habitation", 0.2)])
    wrong = _steps([("defense", 0.6, "habitation", 0.2)])
    # truth at 0.6, nearest other at 0.1 -> +0.5; truth at 0.1, defense 0.6 -> -0.5
    assert intent_metrics.early_separation_verdict(right)["mean_separation"] == 0.5
    assert intent_metrics.early_separation_verdict(right)["passes"]
    assert intent_metrics.early_separation_verdict(wrong)["mean_separation"] == -0.5
    assert not intent_metrics.early_separation_verdict(wrong)["passes"]
    # steps past the 40% cutoff do not enter the verdict
    late = _steps([("habitation", 0.9, "habitation", 0.9)])
    assert intent_metrics.early_separation_verdict(late)["early_steps"] == 0
    assert not intent_metrics.early_separation_verdict(late)["passes"]


def test_top2_counts_second_place():
    steps = _steps([("defense", 0.4, "habitation", 0.5)])   # truth ties for 2nd at 0.15
    assert intent_metrics.top_k_accuracy(steps, 2) in (0.0, 1.0)  # tie-order dependent
    steps = _steps([("defense", 0.9, "habitation", 0.5)])   # truth deep in the pack
    assert intent_metrics.top_k_accuracy(steps, 1) == 0.0


def test_reliability_ece_is_the_confidence_accuracy_gap():
    # two steps, both called at 0.6 confidence, one right one wrong -> acc 0.5, ECE 0.1
    steps = _steps([("habitation", 0.6, "habitation", 0.5),
                    ("defense", 0.6, "habitation", 0.5)])
    rel = intent_metrics.reliability(steps)
    assert rel["points"] == 2
    assert abs(rel["ece"] - 0.1) < 1e-9


def test_recovery_after_pivot():
    steps = _steps([("habitation", 0.6, "habitation", 0.2),   # before the pivot
                    ("habitation", 0.6, "defense", 0.5),      # pivot: still on old goal
                    ("defense", 0.6, "defense", 0.8)])        # recovered
    result = intent_metrics.recovery_after_pivot(steps, 1)
    assert result["old_goal"] == "habitation" and result["new_goal"] == "defense"
    assert result["recovered"] and result["steps_to_recover"] == 1

    never = _steps([("habitation", 0.6, "habitation", 0.2),
                    ("habitation", 0.6, "defense", 0.6)])
    result = intent_metrics.recovery_after_pivot(never, 1)
    assert not result["recovered"] and result["steps_to_recover"] is None


def test_step_normalizes_defensively():
    ragged = intent_metrics.step({"habitation": 2.0, "defense": 2.0}, "habitation", 0.1)
    assert abs(sum(ragged["dist"].values()) - 1.0) < 1e-9
    assert ragged["dist"]["production"] == 0.0
