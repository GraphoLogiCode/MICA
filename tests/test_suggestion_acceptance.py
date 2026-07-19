"""The acceptance signal's pre-registered semantics (D9 §3), on hand-checkable
fixtures: every outcome tier, the voiced/control split, and the lift arithmetic.
The constants are part of the definitions — a test breaking on a constant change
is the version-bump reminder working."""
from mica.validation import suggestion_acceptance as accept


def _place(tick, cell, block="minecraft:oak_planks"):
    return (tick, cell, block)


def test_outcome_tiers_strongest_first():
    cell = (10, 64, 10)
    # exact: same block, same cell, inside the window
    assert accept.classify(100, "minecraft:oak_planks", cell,
                           [_place(150, (10, 64, 10))]) == "followed_exact"
    # contradicted outranks near: a DIFFERENT block at the exact cell is an answer
    assert accept.classify(100, "minecraft:oak_planks", cell,
                           [_place(150, (10, 64, 10), "minecraft:stone"),
                            _place(160, (11, 64, 10))]) == "contradicted"
    # near: same block within Chebyshev NEAR_RADIUS
    assert accept.classify(100, "minecraft:oak_planks", cell,
                           [_place(150, (12, 64, 11))]) == "followed_near"
    # type: same block within TYPE_RADIUS but outside NEAR_RADIUS
    assert accept.classify(100, "minecraft:oak_planks", cell,
                           [_place(150, (17, 64, 10))]) == "followed_type"
    # ignored: nothing relevant in the window
    assert accept.classify(100, "minecraft:oak_planks", cell,
                           [_place(150, (40, 64, 40))]) == "ignored"


def test_block_names_match_across_the_namespace_gap():
    # the capture writes "minecraft:spruce_planks", proposals say "spruce_planks";
    # v1 compared them raw, so no follow tier could ever fire on real data
    cell = (10, 64, 10)
    assert accept.classify(100, "spruce_planks", cell,
                           [_place(150, cell, "minecraft:spruce_planks")]) == "followed_exact"
    # and a genuinely different namespaced block at the cell is still an answer
    assert accept.classify(100, "spruce_planks", cell,
                           [_place(150, cell, "minecraft:stone")]) == "contradicted"


def test_window_edges_are_pre_registered():
    cell = (0, 64, 0)
    # a placement exactly at tick + window counts; one tick later does not;
    # the suggestion's own tick does not (strictly after)
    w = accept.FOLLOW_WINDOW_TICKS
    assert accept.classify(100, "minecraft:stone", cell,
                           [_place(100 + w, cell, "minecraft:stone")]) == "followed_exact"
    assert accept.classify(100, "minecraft:stone", cell,
                           [_place(101 + w, cell, "minecraft:stone")]) == "ignored"
    assert accept.classify(100, "minecraft:stone", cell,
                           [_place(100, cell, "minecraft:stone")]) == "ignored"


def _row(tick, state, block="minecraft:oak_planks", cell=(10, 64, 10)):
    return {"tick": tick, "chosen_state": state,
            "proposal_first": {"block": block, "cell": list(cell)}}


def test_report_splits_voiced_from_control_and_computes_lift():
    # two voiced suggestions (one followed, one ignored) and two unvoiced
    # control reads (neither followed): lift = 0.5 - 0.0
    voiced = [_row(100, "suggest"), _row(800, "suggest", cell=(30, 64, 30))]
    control = [_row(1600, "observe", cell=(50, 64, 50)),
               _row(2400, "suggest", cell=(60, 64, 60))]   # throttled: never voiced
    places = [_place(150, (10, 64, 10))]                    # follows the first only
    result = accept.report(voiced + control, voiced, places)
    assert result["voiced"]["n"] == 2 and result["control_unvoiced"]["n"] == 2
    assert result["voiced"]["followed_rate"] == 0.5
    assert result["control_unvoiced"]["followed_rate"] == 0.0
    assert result["lift"] == 0.5
    assert result["voiced"]["outcomes"] == {"followed_exact": 1, "ignored": 1}


def test_control_can_beat_voiced_and_lift_goes_negative():
    # the honesty case: the human does the proposed thing WITHOUT being told —
    # the control group exists precisely so this reads as zero-or-negative lift,
    # not as acceptance
    voiced = [_row(100, "suggest", cell=(10, 64, 10))]
    control = [_row(800, "observe", cell=(30, 64, 30))]
    places = [_place(850, (30, 64, 30))]                    # follows the UNVOICED one
    result = accept.report(voiced + control, voiced, places)
    assert result["lift"] == -1.0


def test_pair_voicings_matches_nearest_read_and_counts_unmatched():
    rows = [_row(100, "suggest"), _row(300, "preview", cell=(20, 64, 20)),
            _row(500, "observe")]                            # observe never pairs
    paired, unmatched = accept.pair_voicings(
        [{"ts": 5_150}, {"ts": 15_400}, {"ts": 99_000}],     # ms; ticks 103, 308, 1980
        rows, lambda ms: int(ms / 50))
    assert [r["tick"] for r in paired] == [100, 300]
    assert unmatched == 1                                    # tick 1980: nothing near


def test_pair_voicings_never_attaches_to_a_read_from_the_future():
    # the body speaks ~2 s after the read that produced the words, so at 1 Hz the
    # NEXT read is often nearer to the voiced tick — but a voicing cannot come
    # from a read that hadn't happened yet (the 2026-07-18 session paired wrong)
    rows = [_row(100, "suggest"), _row(120, "suggest", cell=(30, 64, 30))]
    paired, unmatched = accept.pair_voicings(
        [{"ts": 5_900}],                                     # tick 118: nearest is 120
        rows, lambda ms: int(ms / 50))
    assert [r["tick"] for r in paired] == [100]
    assert unmatched == 0


def test_pair_voicings_prefers_the_read_proposing_the_logged_cell():
    # when the voiced row logs the cell it spoke about, only reads proposing that
    # cell are candidates — even if another read sits closer in time
    rows = [_row(100, "suggest", cell=(10, 64, 10)),
            _row(110, "suggest", cell=(30, 64, 30))]
    paired, _ = accept.pair_voicings(
        [{"ts": 5_600, "cell": [10, 64, 10]}],               # tick 112: nearer to 110
        rows, lambda ms: int(ms / 50))
    assert [tuple(r["proposal_first"]["cell"]) for r in paired] == [(10, 64, 10)]
