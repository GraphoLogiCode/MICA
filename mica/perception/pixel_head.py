"""The live pixel head: VPT + MineCLIP running WHILE the game runs (D6 §8, stage 2).

Offline, `run_d1 --pixels` fills two channels on every behavior record: h2d (what
the play looks like, from VPT) and s_goal (how much the recent view resembles
building each goal category, from MineCLIP against text prompts). This module is
the live version of that same step. It plugs into the one seam the live pipeline
reserved for it — `PipelineConfig.pixel_head`, whose only method is
`enrich(record, frames)` — so nothing in the pipeline cores changes.

How it stays out of the game's way: all the heavy work (loading the models,
running them) happens on one background worker thread that owns the GPU. The
runner feeds every arriving frame to `observe()` (cheap: append to a ring).
When the pipeline releases a scored record, `enrich()` hands the worker that
record's exact frame window — the same window rule the batch path uses — and
waits a short deadline for the answer. If the models are still loading, or the
worker cannot answer in time, the record passes through unchanged with both
pixel channels empty together, exactly the batch no-frames behavior. So the
symbolic pipeline never blocks on the GPU for more than the deadline.

Between records the worker also scores the newest clip about once a second and
keeps the result as a display-only readout (`status()`): what MineCLIP thinks
the player is doing RIGHT NOW. That number is for the status line, the
live_status.json bridge, and the in-game agent — it is never written into a
record, so the evidence log stays strictly the enriched-record path.

The encoders arrive through a small object with load/h2d/s_goal so tests can
inject fakes; the real one (TorchEncoders) imports torch only when it loads.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import queue
import threading
from collections import deque
from typing import Protocol

from ..contracts.b1 import GOALS, Evidence2D

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CLIP_LEN = 16                 # MineCLIP's clip length (mirrors mineclip_head, which
                               # cannot be imported here: it pulls in torch at import)
_FRAME_RING = 260              # raw frames kept: covers a stride-16 clip (16*16) plus slack
_ENRICH_DEADLINE_S = 0.4       # how long a scored record may wait on the GPU
_DISPLAY_EVERY_TICKS = 20      # the ~1 Hz display readout cadence
_JOB_QUEUE_CAP = 4             # a worker this far behind should shed work, not queue it


class Encoders(Protocol):
    """What the worker needs from the models. Frames arrive raw off the wire:
    (tick, width, height, rgba_bytes) tuples — the implementation decodes them."""

    def load(self) -> None: ...
    def h2d(self, frames: list) -> tuple | None: ...
    def s_goal(self, frames: list, stride: int) -> tuple | None: ...


class TorchEncoders:
    """The real models: frozen VPT trunk + frozen MineCLIP, loaded on the worker
    thread so the game and pipeline never wait on torch."""

    def __init__(self, device: str = "cuda"):
        self._device = device
        self._vpt = None
        self._clip = None

    def load(self) -> None:
        import numpy as np
        from .mineclip_head import MineClipHead
        from .vpt_trunk import VptTrunk
        self._vpt = VptTrunk(self._device)
        self._clip = MineClipHead(self._device)
        # Warm up: the first CUDA pass compiles kernels and can take seconds —
        # done here, so the head never reports "loaded" while still that slow.
        dummy = [np.zeros((64, 64, 3), dtype=np.uint8)]
        self._vpt.embed(dummy)
        self._clip.score(dummy, 1)

    @staticmethod
    def _decode(frame) -> "object":
        """One wire frame (tick, w, h, raw RGBA bytes) -> an HxWx3 uint8 array,
        the shape both model wrappers accept."""
        import numpy as np
        _, width, height, data = frame
        rgba = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(rgba[:, :, :3])

    def h2d(self, frames: list) -> tuple | None:
        return self._vpt.embed([self._decode(f) for f in frames])

    def s_goal(self, frames: list, stride: int) -> tuple | None:
        return self._clip.score([self._decode(f) for f in frames], stride)


class LivePixelHead:
    """Implements the pipeline's PixelHead seam, backed by one GPU worker thread."""

    def __init__(self, encoders: Encoders, stride: int = 1,
                 deadline_s: float = _ENRICH_DEADLINE_S):
        self._encoders = encoders
        self.stride = stride
        self._deadline_s = deadline_s
        self._ring: deque = deque(maxlen=_FRAME_RING)   # (tick, w, h, rgba_bytes)
        self._jobs: queue.Queue = queue.Queue(maxsize=_JOB_QUEUE_CAP)
        self._loaded = threading.Event()
        self.load_error: str | None = None
        self.latest_display: dict | None = None          # {"tick": t, "s_goal": {goal: v}}
        self._last_display_tick = -10**9
        self.enriched = 0
        self.skipped_not_ready = 0
        self.deadline_missed = 0
        self._worker = threading.Thread(target=self._run, name="pixel-head", daemon=True)
        self._worker.start()

    # ---------------------------------------------------------------- runner side

    def observe(self, tick: int, frame_ref, rgba: bytes) -> None:
        """Every live frame lands here (cheap). Also nudges the ~1 Hz display job."""
        self._ring.append((tick, frame_ref.width, frame_ref.height, rgba))
        if (self._loaded.is_set()
                and tick - self._last_display_tick >= _DISPLAY_EVERY_TICKS
                and self._jobs.qsize() < 2):  # never let stale readouts pile up
            clip = self._clip_frames(tick)
            if clip:
                try:
                    self._jobs.put_nowait(("display", tick, clip, None))
                    self._last_display_tick = tick
                except queue.Full:
                    pass                     # worker busy: skip this readout, not a record

    def enrich(self, record: Evidence2D, frames) -> Evidence2D:
        """The pipeline seam. Exact window, bounded wait, unchanged on any miss."""
        if not self._loaded.is_set():
            self.skipped_not_ready += 1
            return record
        t0, t1 = record.tick_range
        window = [f for f in self._ring if t0 <= f[0] <= t1]
        if not window:
            # No frames inside the evidence window: both channels stay None together,
            # even if a wider clip span could reach older frames (batch rule).
            return dataclasses.replace(record, h2d=None, s_goal=None)
        clip = window if self.stride == 1 else self._clip_frames(t1)
        reply: queue.Queue = queue.Queue(maxsize=1)
        job = ("record", None, (window, clip), reply)
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            # A record outranks whatever queued up (display readouts, or an older
            # record whose caller already gave up): displace one and retry once.
            try:
                self._jobs.get_nowait()
                self._jobs.put_nowait(job)
            except (queue.Empty, queue.Full):
                self.deadline_missed += 1
                return record
        try:
            h2d, s_goal = reply.get(timeout=self._deadline_s)
        except queue.Empty:
            self.deadline_missed += 1        # the answer, if it ever comes, goes nowhere
            return record
        self.enriched += 1
        return dataclasses.replace(record, h2d=h2d, s_goal=s_goal)

    def status(self) -> dict:
        display = self.latest_display
        return {
            "loaded": self._loaded.is_set(),
            "load_error": self.load_error,
            "enriched": self.enriched,
            "skipped_not_ready": self.skipped_not_ready,
            "deadline_missed": self.deadline_missed,
            "display_tick": display["tick"] if display else None,
            "s_goal_live": display.get("s_goal") if display else None,
            "h2d_norm_live": display.get("h2d_norm") if display else None,
            "h2d_profile_live": display.get("h2d_profile") if display else None,
        }

    def close(self) -> None:
        try:
            self._jobs.put_nowait(("stop", None, None, None))
        except queue.Full:
            pass                              # daemon thread; process exit reaps it

    # ---------------------------------------------------------------- worker side

    def _clip_frames(self, end_tick: int) -> list:
        """The s_goal clip: frames at/before end_tick reaching back CLIP_LEN * stride
        ticks (stride 1 = the last ~0.8 s; the head's stride mirrors run_d1's flag)."""
        first = end_tick - _CLIP_LEN * self.stride + 1
        return [f for f in self._ring if first <= f[0] <= end_tick]

    def _run(self) -> None:
        try:
            self._encoders.load()
        except Exception as error:            # missing torch/checkpoints: degrade, loudly
            self.load_error = f"{type(error).__name__}: {error}"
            return
        self._loaded.set()
        while True:
            kind, tick, payload, reply = self._jobs.get()
            if kind == "stop":
                return
            try:
                if kind == "record":
                    window, clip = payload
                    result = (self._encoders.h2d(window),
                              self._encoders.s_goal(clip, self.stride))
                    if reply is not None:
                        try:
                            reply.put_nowait(result)
                        except queue.Full:
                            pass              # enrich gave up already
                else:                         # display: both models on the newest clip —
                    # the ~1 Hz readout a human watches to see the models working.
                    scores = self._encoders.s_goal(payload, self.stride)
                    h2d = self._encoders.h2d(payload)
                    display: dict = {"tick": tick}
                    if scores is not None:
                        display["s_goal"] = dict(zip(GOALS, (round(v, 4) for v in scores)))
                    if h2d is not None:
                        display["h2d_norm"] = round(math.sqrt(sum(v * v for v in h2d)), 2)
                        # 1024 dims -> 32 bucket means: a strip a human can watch move
                        step = max(1, len(h2d) // 32)
                        display["h2d_profile"] = [
                            round(sum(h2d[i:i + step]) / step, 3)
                            for i in range(0, len(h2d) - step + 1, step)][:32]
                    if len(display) > 1:
                        self.latest_display = display
            except Exception as error:        # one bad frame must not kill the worker
                self.load_error = f"{type(error).__name__}: {error}"


# ------------------------------------------------------------------- provenance

def _sha256(path: str) -> str:
    if not os.path.exists(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_d1_provenance(evidence_path: str, stride: int) -> str:
    """Record the exact checkpoints, the goal prompts, and the s_goal path a pixel
    run used (D1 spec §8) — shared by run_d1 (batch) and run_live (live head).
    The prompts, similarity path, and clip stride all define s_goal the way D2's
    templates define its features — changing any of them silently changes every
    future score, so the sidecar must pin them.
    """
    from ..contracts.b1 import GOALS as goal_names
    from ..contracts.goals import TAXONOMY, TAXONOMY_VERSION
    from .mineclip_head import PROMPT_TEMPLATES

    provenance = {
        "vpt_checkpoint": "vpt-1x.weights",
        "vpt_checkpoint_sha256": _sha256(os.path.join(_ROOT, "models", "vpt-1x.weights")),
        "mineclip_checkpoint": "mineclip_attn.pth",
        "mineclip_checkpoint_sha256": _sha256(os.path.join(_ROOT, "models", "mineclip_attn.pth")),
        "goals": list(goal_names),
        "goal_taxonomy_version": TAXONOMY_VERSION,
        "goal_taxonomy": {goal: list(subs) for goal, subs in TAXONOMY.items()},
        "prompt_templates": list(PROMPT_TEMPLATES),
        "s_goal_path": "trained-video-adapter",   # through the reward head's adapter + gate
        "s_goal_clip_stride": stride,
    }
    out = evidence_path.replace(".evidence2d.jsonl", ".d1_provenance.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    return out
