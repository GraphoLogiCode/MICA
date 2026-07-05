"""The B1 contract validator accepts a clean Evidence2D stream and flags each kind of
violation — so a PASS on real D1 output actually means the records are safe for D3."""
import dataclasses

from mica.capture.sample_builds import mixed_build, sparse_build
from mica.capture.synthetic import generate_session
from mica.contracts.b1 import GOALS, MacroAction
from mica.perception.evidence2d import evidence_stream
from mica.validation.evidence2d_check import check_record, check_stream


def _stream():
    return list(evidence_stream(generate_session(mixed_build()).packets))


def test_clean_symbolic_stream_passes():
    assert check_stream(_stream()) == []


def test_bad_confidence_flagged():
    records = _stream()
    records[0] = dataclasses.replace(records[0], a_hat_conf=1.5)
    assert any("a_hat_conf" in m for m in check_stream(records))


def test_idle_flag_inconsistency_flagged():
    place = next(r for r in _stream() if r.a_hat is MacroAction.PLACE)
    assert any("idle" in m for m in check_record(dataclasses.replace(place, idle=True)))


def test_s_goal_length_and_range_flagged():
    r = _stream()[0]
    short = dataclasses.replace(r, h2d=(0.0,) * 4, s_goal=(0.1, 0.2))
    assert any("s_goal length" in m for m in check_record(short))
    out_of_range = dataclasses.replace(r, h2d=(0.0,) * 4, s_goal=tuple([0.1] * (len(GOALS) - 1) + [9.0]))
    assert any("[-1, 1]" in m for m in check_record(out_of_range))


def test_pixel_presence_must_match():
    r = _stream()[0]
    assert any("together" in m for m in check_record(dataclasses.replace(r, h2d=(0.0,) * 4, s_goal=None)))


def test_double_consumption_flagged():
    records = _stream()
    records.append(next(r for r in records if r.event_ids))   # same event ids appear twice
    assert any("consum" in m for m in check_stream(records))


def test_stream_with_context_records_passes():
    # sparse_build produces both scored corrections and unscored ~1 Hz context records;
    # a conforming mixed stream must validate clean.
    records = list(evidence_stream(generate_session(sparse_build()).packets))
    assert any(not r.scored for r in records)
    assert check_stream(records) == []


def test_context_record_consuming_events_flagged():
    place = next(r for r in _stream() if r.event_ids)
    bad = dataclasses.replace(place, scored=False)   # keeps its event ids -> double counting
    assert any("context" in m for m in check_record(bad))
