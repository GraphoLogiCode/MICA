"""The live pixel head (D6 §8 stage 2), tested torch-free with fake encoders.

What must hold: enrich uses exactly the record's own frame window (strictly
pre-action), an empty window or an unready/slow worker leaves both pixel
channels None together (the batch no-frames behavior), and the display readout
advances on its own about once a second without ever touching a record.
"""
import dataclasses
import threading
import time

from mica.contracts.b0 import FrameRef
from mica.contracts.b1 import GOALS, Evidence2D, Focus, MacroAction, StateFeats
from mica.contracts.serialize import evidence2d_to_dict
from mica.perception.pixel_head import LivePixelHead


class FakeEncoders:
    """Counts calls and remembers which frame ticks each computation saw."""

    def __init__(self, load_delay_s=0.0, compute_delay_s=0.0, fail_load=False):
        self.load_delay_s = load_delay_s
        self.compute_delay_s = compute_delay_s
        self.fail_load = fail_load
        self.h2d_windows: list[tuple[int, ...]] = []
        self.s_goal_clips: list[tuple[int, ...]] = []
        self.loaded = threading.Event()

    def load(self):
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
        if self.fail_load:
            raise RuntimeError("no torch here")
        self.loaded.set()

    def h2d(self, frames):
        if self.compute_delay_s:
            time.sleep(self.compute_delay_s)
        self.h2d_windows.append(tuple(f[0] for f in frames))
        return (0.5,) * 64

    def s_goal(self, frames, stride):
        self.s_goal_clips.append(tuple(f[0] for f in frames))
        return tuple(0.1 * (i + 1) for i in range(len(GOALS)))


def _record(t0, t1, scored=True):
    return Evidence2D(
        tick_range=(t0, t1), a_hat=MacroAction.PLACE, a_hat_conf=1.0, idle=False,
        state_feats=StateFeats(held_item="minecraft:dirt", hotbar=("minecraft:dirt",),
                               pos_delta=(0.0, 0.0, 0.0), yaw_delta=0.0, pitch_delta=0.0,
                               recent_actions=("place",)),
        focus=Focus(block=None, dwell_ticks=0),
        scored=scored, event_ids=(1,) if scored else ())


def _head(encoders, **kwargs):
    head = LivePixelHead(encoders, **kwargs)
    assert head_ready(head, expect=not encoders.fail_load)
    return head


def head_ready(head, expect=True, timeout_s=2.0):
    """Wait for the worker to finish (or fail) loading."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if head.status()["loaded"] == expect and (expect or head.load_error):
            return True
        time.sleep(0.01)
    return False


def _feed(head, ticks):
    ref = FrameRef(path="", width=4, height=2)
    for tick in ticks:
        head.observe(tick, ref, bytes(4 * 2 * 4))


def test_enrich_uses_exactly_the_window_frames():
    encoders = FakeEncoders()
    head = _head(encoders)
    _feed(head, range(1, 51))                      # ring holds ticks 1..50
    enriched = head.enrich(_record(10, 29), ())
    assert enriched.h2d == (0.5,) * 64
    assert enriched.s_goal == tuple(0.1 * (i + 1) for i in range(len(GOALS)))
    assert encoders.h2d_windows[-1] == tuple(range(10, 30))
    # strictly pre-action: nothing after the window's end ever reaches a model
    assert max(encoders.h2d_windows[-1]) == 29
    assert max(encoders.s_goal_clips[-1]) == 29


def test_stride_widens_the_clip_but_not_the_window():
    encoders = FakeEncoders()
    head = _head(encoders, stride=2)
    _feed(head, range(1, 101))
    head.enrich(_record(80, 99), ())
    assert encoders.h2d_windows[-1] == tuple(range(80, 100))     # window: the record's own
    clip = encoders.s_goal_clips[-1]
    assert min(clip) == 99 - 16 * 2 + 1                          # clip reaches further back
    assert max(clip) == 99                                       # ... but never forward


def test_no_frames_in_window_leaves_both_channels_none():
    encoders = FakeEncoders()
    head = _head(encoders)
    _feed(head, range(1, 20))
    enriched = head.enrich(_record(100, 119), ())
    assert enriched.h2d is None and enriched.s_goal is None
    assert encoders.h2d_windows == []              # the models were never asked


def test_not_loaded_passes_the_record_unchanged():
    head = LivePixelHead(FakeEncoders(load_delay_s=5.0))
    record = _record(10, 29)
    assert head.enrich(record, ()) is record
    assert head.status()["skipped_not_ready"] == 1


def test_load_failure_degrades_to_symbolic():
    head = LivePixelHead(FakeEncoders(fail_load=True))
    assert head_ready(head, expect=False)
    assert "no torch here" in head.load_error
    record = _record(10, 29)
    assert head.enrich(record, ()) is record


def test_deadline_miss_passes_the_record_unchanged():
    encoders = FakeEncoders(compute_delay_s=0.5)
    head = _head(encoders, deadline_s=0.05)
    _feed(head, range(1, 51))
    record = _record(10, 29)
    assert head.enrich(record, ()) is record
    assert head.status()["deadline_missed"] == 1


def test_display_readout_advances_without_touching_records():
    encoders = FakeEncoders()
    head = _head(encoders)
    _feed(head, range(1, 61))                      # crosses the ~1 Hz display cadence twice
    deadline = time.monotonic() + 2.0
    while head.latest_display is None and time.monotonic() < deadline:
        time.sleep(0.01)
    status = head.status()
    assert status["display_tick"] is not None
    assert set(status["s_goal_live"]) == set(GOALS)
    assert status["h2d_norm_live"] is not None     # both models feed the display now
    assert len(status["h2d_profile_live"]) == 32   # the 1024-d strip, bucketed
    assert head.enriched == 0                      # ... but no RECORD was touched


def test_enriched_record_serializes_like_batch():
    head = _head(FakeEncoders())
    _feed(head, range(1, 51))
    payload = evidence2d_to_dict(head.enrich(_record(10, 29), ()))
    assert len(payload["h2d"]) == 64
    assert len(payload["s_goal"]) == len(GOALS)
