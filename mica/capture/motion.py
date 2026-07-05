"""Choreographed body and gaze for a scripted build — the naturalness layer.

Turns a frozen placement plan into a per-tick pose track: the builder walks within
reach before acting, settles its aim on a block before clicking, lingers on what it
just placed, stares at the spot a mistake was broken from, sweeps the growing
structure during thinking pauses, and occasionally glances at a virtual companion
watching from outside the build. The plan itself is never altered — every block
change keeps its tick and position; only the pose around the actions is invented.
Everything draws from one generator seeded by the session id, so the same build
always yields the same track.

The crosshair here is authored attention, not a ray trace: it names a block only
while the gaze deliberately rests on one, and stays blank while turning or idling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

from ..contracts.b0 import BlockOp, BlockPos, CrosshairTarget

if TYPE_CHECKING:
    from .synthetic import ScriptedBuild, ScriptedPlacement

_WALK_SPEED = 0.216        # blocks per tick, vanilla walking
_SPRINT_SPEED = 0.28       # blocks per tick, vanilla sprinting
_FLICK_CAP_DEG = 180.0     # the hard bound on one tick of head turning (a mouse yank)
_EYE_HEIGHT = 1.62
_GROUND_Y = 64.0
_STAND_RADIUS = 4.0        # stand at most this far from the work; reach stays under 4.5
_STAND_MIN = 1.2           # never stand inside the block being worked on
_CLUSTER_GAP_TICKS = 2     # a mistake's break lands exactly two ticks after a placement;
                           # anything slower is a separate act with its own stand
_GLANCE_MIN_FILLER = 20    # a companion glance needs a real pause, not a beat
_SWEEP_MIN_FILLER = 30     # an evaluation sweep needs a thinking pause
_GLANCE_HOLD_TICKS = (6, 10)
_TAIL_IDLE_TICKS = 2       # quiet steps closing the recording after the final look
_COMPANION_RADIUS = 10.0   # the virtual watcher circles the build this far out
_COMPANION_DEG_PER_TICK = 0.4
_COMPANION_ENTITY = "minecraft:player"
_AIR = CrosshairTarget(block_pos=None, face=None, entity=None)
_COMPANION_LOOK = CrosshairTarget(block_pos=None, face=None, entity=_COMPANION_ENTITY)
_WALK_KEYS = ("forward",)
_SPRINT_KEYS = ("forward", "sprint")


@dataclass(frozen=True)
class MotionStyle:
    """How carefully a builder moves and looks — the mode latent's gaze signature."""

    turn_cap_deg: float                    # ordinary head-turn speed, degrees per tick
    pre_dwell_ticks: tuple[int, int]       # settle on the target before acting
    confirm_dwell_ticks: tuple[int, int]   # linger on a block just placed
    verify_dwell_ticks: tuple[int, int]    # stare at the spot a block was broken from
    sweep_targets: int                     # built cells visited per evaluation sweep
    sweep_hold_ticks: int                  # how long the gaze rests on each swept cell
    companion_glance_chance: float         # chance a long pause includes a social glance
    # The fastest the frozen schedule may force this builder to move, blocks per tick.
    # Reach is the hard invariant (the builder always arrives before acting), so when a
    # plan's pacing outruns vanilla sprint, speed gives way — the careful builder never
    # needs more than sprint; the hasty one scrambles, and that too is a mode signature.
    dash_cap: float


DELIBERATE_GAZE = MotionStyle(
    turn_cap_deg=12.0,
    pre_dwell_ticks=(4, 8),
    confirm_dwell_ticks=(4, 8),
    verify_dwell_ticks=(4, 6),
    sweep_targets=4,
    sweep_hold_ticks=6,
    companion_glance_chance=0.2,
    dash_cap=_SPRINT_SPEED,
)
SHORTCUT_GAZE = MotionStyle(
    turn_cap_deg=30.0,
    pre_dwell_ticks=(1, 3),
    confirm_dwell_ticks=(1, 2),
    verify_dwell_ticks=(1, 2),
    sweep_targets=1,
    sweep_hold_ticks=2,
    companion_glance_chance=0.08,
    dash_cap=0.75,
)


@dataclass(frozen=True)
class TickPose:
    """The builder's body and attention for one tick of a choreographed session.

    Yaw is continuous (no wrap-around jump at ±180), so per-tick deltas read
    honestly downstream — the capture contract puts no range on yaw.
    """

    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    keys: tuple[str, ...]
    crosshair: CrosshairTarget


@dataclass(frozen=True)
class _Action:
    """Every block change sharing one tick, treated as a single act of attention."""

    tick: int
    changes: tuple["ScriptedPlacement", ...]


@dataclass(frozen=True)
class _GapWish:
    """How the builder would spend the ticks before an action, given endless time."""

    confirm: int    # lingering on the previous action's spot
    settle: int     # turning to the next target plus the locked pre-dwell
    walk_slow: int  # ticks to reach the next stand at walking pace
    walk_fast: int  # the same trip at a sprint


@dataclass(frozen=True)
class _GapPlan:
    """The wish fitted into the ticks actually available."""

    confirm: int
    filler: int
    walk: int
    settle: int
    keys: tuple[str, ...]


def choreograph(build: "ScriptedBuild") -> tuple[TickPose, ...]:
    """One pose per tick (index == tick), covering the plan plus a closing look."""
    return _Choreographer(build).track()


def _squeeze(gap: int, wish: _GapWish) -> _GapPlan:
    """Fit the wished phases into the gap. Free looking-around absorbs any slack;
    when time is short the aim settle gives way first (a flick can save the aim),
    then the lingering, then walking speeds up to a sprint."""
    if gap <= 0:
        return _GapPlan(confirm=0, filler=0, walk=0, settle=0, keys=_WALK_KEYS)
    confirm, settle, walk, keys = wish.confirm, wish.settle, wish.walk_slow, _WALK_KEYS
    short = confirm + walk + settle - gap
    if short > 0:
        give = min(short, max(0, settle - 2))
        settle -= give
        short -= give
    if short > 0:
        give = min(short, confirm)
        confirm -= give
        short -= give
    if short > 0 and wish.walk_fast < walk:
        short -= walk - wish.walk_fast
        walk, keys = wish.walk_fast, _SPRINT_KEYS
    if short > 0:
        # Walking keeps every tick it needs; the approach itself lands the aim.
        give = min(short, settle)
        settle -= give
        short -= give
    if short > 0:
        walk = max(0, walk - short)
    filler = max(0, gap - confirm - walk - settle)
    return _GapPlan(confirm=confirm, filler=filler, walk=walk, settle=settle, keys=keys)


class _Choreographer:
    """Walks one build plan start to finish, emitting exactly one pose per tick."""

    def __init__(self, build: "ScriptedBuild"):
        self.style = build.motion
        self.rng = Random(f"motion:{build.session_id}")
        self.actions = _group_actions(build.placements)
        self.companion_center = _footprint_center(build.placements)
        self.companion_phase = self.rng.uniform(0.0, 360.0)
        self.stands = _plan_stands(self.actions, self.companion_center)
        self.built: list[BlockPos] = []
        self.poses: list[TickPose] = []
        self.target: BlockPos | None = None
        start = self.stands[0] if self.stands else (0.0, _GROUND_Y, 0.0)
        self.x, self.y, self.z = start
        # The session opens already facing the work — no artificial whip at tick 0.
        self.yaw, self.pitch = (
            self._aim_from_eye(_center(self.actions[0].changes[0].pos))
            if self.actions else (0.0, 0.0)
        )

    def track(self) -> tuple[TickPose, ...]:
        if not self.actions:
            for _ in range(_TAIL_IDLE_TICKS + 1):
                self._emit((), _AIR)
            return tuple(self.poses)
        previous = None
        for action, stand in zip(self.actions, self.stands):
            self._bridge(previous, action, stand)
            self._act(action)
            previous = action
        self._tail(previous)
        return tuple(self.poses)

    def _bridge(self, previous: _Action | None, action: _Action, stand) -> None:
        """Spend the ticks before an action: linger, look around, walk, take aim."""
        self.target = action.changes[0].pos
        gap = action.tick - len(self.poses)
        plan = _squeeze(gap, self._wish(previous, stand))
        self._linger(previous, plan.confirm)
        self._look_around(plan.filler)
        if plan.settle > 0:
            self._walk(stand, plan.walk, plan.keys)
            self._take_aim(plan.settle)
        else:
            self._walk_and_land(stand, plan.walk, plan.keys)

    def _wish(self, previous: _Action | None, stand) -> _GapWish:
        confirm = 0 if previous is None else self._dwell_wish(previous)
        distance = math.dist((self.x, self.y, self.z), stand)
        eye = (stand[0], stand[1] + _EYE_HEIGHT, stand[2])
        turn = _angle_between((self.yaw, self.pitch), _aim_at(eye, _center(self.target)))
        low, high = self.style.pre_dwell_ticks
        settle = math.ceil(turn / self.style.turn_cap_deg) + self.rng.randint(low, high)
        return _GapWish(
            confirm=confirm,
            settle=settle,
            walk_slow=math.ceil(distance / _WALK_SPEED),
            walk_fast=math.ceil(distance / _SPRINT_SPEED),
        )

    def _dwell_wish(self, previous: _Action) -> int:
        op = previous.changes[0].op
        low, high = (self.style.verify_dwell_ticks if op is BlockOp.BREAK
                     else self.style.confirm_dwell_ticks)
        return self.rng.randint(low, high)

    def _linger(self, previous: _Action | None, ticks: int) -> None:
        """Hold the gaze where the last action landed — confirming a placement, or
        staring at the emptied spot after a break (blank crosshair: the block is gone)."""
        if ticks <= 0 or previous is None:
            return
        first = previous.changes[0]
        crosshair = _AIR if first.op is BlockOp.BREAK else _on_block(first.pos)
        for _ in range(ticks):
            self._emit((), crosshair)

    def _look_around(self, ticks: int) -> None:
        """A thinking pause: maybe a glance at the companion, maybe an evaluation
        sweep over what is already built, then rest."""
        remaining = ticks
        if (remaining >= _GLANCE_MIN_FILLER
                and self.rng.random() < self.style.companion_glance_chance):
            remaining = self._glance(remaining)
        if remaining >= _SWEEP_MIN_FILLER and self.built:
            remaining = self._sweep(remaining)
        for _ in range(remaining):
            self._emit((), _AIR)

    def _glance(self, remaining: int) -> int:
        """Turn to the watching companion and hold eye contact for a moment. The
        entity tag appears only once the gaze has settled on them."""
        hold = self.rng.randint(*_GLANCE_HOLD_TICKS)
        while remaining > hold and not self._turned(self._companion_aim(), self.style.turn_cap_deg):
            self._emit((), _AIR)
            remaining -= 1
        if remaining <= hold:
            return remaining
        for _ in range(hold):
            self._emit((), _COMPANION_LOOK)
            remaining -= 1
        return remaining

    def _sweep(self, remaining: int) -> int:
        """Evaluate the build: rest the gaze on a few already-built cells in turn."""
        count = min(self.style.sweep_targets, len(self.built))
        for cell in self.rng.sample(self.built, count):
            aim = self._aim_from_eye(_center(cell))
            while remaining > 0 and not self._turned(aim, self.style.turn_cap_deg):
                self._emit((), _AIR)
                remaining -= 1
            hold = min(self.style.sweep_hold_ticks, remaining)
            if hold <= 0:
                break
            for _ in range(hold):
                self._emit((), _on_block(cell))
                remaining -= 1
        return remaining

    def _walk(self, stand, ticks: int, keys: tuple[str, ...]) -> None:
        """Move to the next stand in even steps, gaze drifting toward the work."""
        if ticks <= 0:
            self.x, self.y, self.z = stand
            return
        for k in range(ticks, 0, -1):
            self._step_to(stand, k)
            self._turned(self._aim_from_eye(_center(self.target)), self.style.turn_cap_deg)
            self._emit(keys, _AIR)

    def _walk_and_land(self, stand, ticks: int, keys: tuple[str, ...]) -> None:
        """A dead-run approach — the whole bridge is travel, so the gaze must land,
        flick included, on the run's final step, crosshair on the target."""
        if ticks <= 0:
            self.x, self.y, self.z = stand
            return
        for k in range(ticks, 0, -1):
            self._step_to(stand, k)
            aim = self._aim_from_eye(_center(self.target))
            landed = self._turned(aim, self.style.turn_cap_deg if k > 1 else _FLICK_CAP_DEG)
            self._emit(keys, _on_block(self.target) if landed and k == 1 else _AIR)

    def _step_to(self, stand, k: int) -> None:
        self.x += (stand[0] - self.x) / k
        self.y += (stand[1] - self.y) / k
        self.z += (stand[2] - self.z) / k

    def _take_aim(self, ticks: int) -> None:
        """Settle the gaze onto the next block: ordinary turning first, and if the
        schedule is tight, one hard flick on the last tick before the action. The
        crosshair names the target from the moment the gaze lands."""
        aim = self._aim_from_eye(_center(self.target))
        landed = False
        for k in range(ticks, 0, -1):
            if not landed:
                cap = self.style.turn_cap_deg if k > 1 else _FLICK_CAP_DEG
                landed = self._turned(aim, cap)
            self._emit((), _on_block(self.target) if landed else _AIR)

    def _act(self, action: _Action) -> None:
        """The action tick itself: stationary, aimed, crosshair on the block."""
        self._emit((), _on_block(action.changes[0].pos))
        for change in action.changes:
            if change.op is BlockOp.BREAK:
                if change.pos in self.built:
                    self.built.remove(change.pos)
            else:
                self.built.append(change.pos)

    def _tail(self, previous: _Action) -> None:
        """Run past the last action: one final confirming look, then a quiet beat."""
        self._linger(previous, self._dwell_wish(previous))
        for _ in range(_TAIL_IDLE_TICKS):
            self._emit((), _AIR)

    def _emit(self, keys: tuple[str, ...], crosshair: CrosshairTarget) -> None:
        self.poses.append(TickPose(x=self.x, y=self.y, z=self.z, yaw=self.yaw,
                                   pitch=self.pitch, keys=keys, crosshair=crosshair))

    def _turned(self, aim: tuple[float, float], cap: float) -> bool:
        """One bounded step of head turning toward an aim; True once the gaze is there.
        A landing snap is always followed by at least one emitted hold tick, so no
        two turning steps ever fold into a single pose delta."""
        target_yaw = _unwrap(aim[0], near=self.yaw)
        step_yaw, step_pitch = target_yaw - self.yaw, aim[1] - self.pitch
        step = max(abs(step_yaw), abs(step_pitch))
        if step <= cap:
            self.yaw, self.pitch = target_yaw, aim[1]
            return True
        self.yaw += step_yaw * (cap / step)
        self.pitch += step_pitch * (cap / step)
        return False

    def _aim_from_eye(self, point) -> tuple[float, float]:
        return _aim_at((self.x, self.y + _EYE_HEIGHT, self.z), point)

    def _companion_aim(self) -> tuple[float, float]:
        """Where the slowly drifting companion stands right now, as aim angles."""
        angle = math.radians(self.companion_phase
                             + _COMPANION_DEG_PER_TICK * len(self.poses))
        watcher = (self.companion_center[0] + _COMPANION_RADIUS * math.cos(angle),
                   _GROUND_Y + _EYE_HEIGHT,
                   self.companion_center[1] + _COMPANION_RADIUS * math.sin(angle))
        return self._aim_from_eye(watcher)


def _group_actions(placements) -> tuple[_Action, ...]:
    by_tick: dict[int, list] = {}
    for placement in placements:
        by_tick.setdefault(placement.tick, []).append(placement)
    return tuple(_Action(tick, tuple(changes)) for tick, changes in sorted(by_tick.items()))


def _footprint_center(placements) -> tuple[float, float]:
    if not placements:
        return (0.0, 0.0)
    xs = [placement.pos.x + 0.5 for placement in placements]
    zs = [placement.pos.z + 0.5 for placement in placements]
    return (sum(xs) / len(xs), sum(zs) / len(zs))


def _plan_stands(actions, start_xz) -> tuple[tuple[float, float, float], ...]:
    """One stand per action; actions close in time share one (a placement and the
    mistake broken right after it get a single vantage point)."""
    stands: list[tuple[float, float, float]] = []
    previous = start_xz
    for cluster in _clusters(actions):
        stand = _stand_for(cluster, previous)
        stands.extend([stand] * len(cluster))
        previous = (stand[0], stand[2])
    return tuple(stands)


def _clusters(actions) -> list[list[_Action]]:
    groups: list[list[_Action]] = []
    for action in actions:
        if groups and action.tick - groups[-1][-1].tick <= _CLUSTER_GAP_TICKS:
            groups[-1].append(action)
        else:
            groups.append([action])
    return groups


def _stand_for(cluster, previous) -> tuple[float, float, float]:
    """A spot within working reach of every block the cluster touches, moved no
    farther from the previous spot than it has to be."""
    points = [_center(change.pos) for action in cluster for change in action.changes]
    anchor = _anchor_xz(points, previous)
    x, z = _fit_xz(anchor, [(point[0], point[2]) for point in points])
    feet_y = max(_GROUND_Y, max(point[1] for point in points) - 1.5)
    return (x, feet_y, z)


def _anchor_xz(points, previous) -> tuple[float, float]:
    if len(points) == 1:
        return _toward((points[0][0], points[0][2]), previous)
    far_a, far_b = _farthest_pair(points)
    mid = ((far_a[0] + far_b[0]) / 2.0, (far_a[2] + far_b[2]) / 2.0)
    spread = math.hypot(far_a[0] - far_b[0], far_a[2] - far_b[2]) / 2.0
    slack = max(0.0, (_STAND_RADIUS - spread) * 0.9)
    return _step_toward(mid, previous, slack)


def _toward(target, previous) -> tuple[float, float]:
    """The point on the target's stand ring closest to where the builder already is —
    walk only as far as the work demands."""
    dx, dz = previous[0] - target[0], previous[1] - target[1]
    distance = math.hypot(dx, dz)
    if distance < 1e-9:
        return (target[0] + _STAND_RADIUS / 2.0, target[1])
    reach = min(max(distance, _STAND_MIN), _STAND_RADIUS)
    return (target[0] + dx / distance * reach, target[1] + dz / distance * reach)


def _step_toward(origin, goal, limit: float) -> tuple[float, float]:
    dx, dz = goal[0] - origin[0], goal[1] - origin[1]
    distance = math.hypot(dx, dz)
    if distance < 1e-9 or limit <= 0.0:
        return origin
    fraction = min(1.0, limit / distance)
    return (origin[0] + dx * fraction, origin[1] + dz * fraction)


def _fit_xz(candidate, targets) -> tuple[float, float]:
    """Nudge a stand until every target is within radius and none is underfoot."""
    x, z = candidate
    for _ in range(3):
        for tx, tz in targets:
            dx, dz = x - tx, z - tz
            distance = math.hypot(dx, dz)
            if distance < 1e-9:
                x += _STAND_MIN
            elif distance < _STAND_MIN:
                x, z = tx + dx / distance * _STAND_MIN, tz + dz / distance * _STAND_MIN
            elif distance > _STAND_RADIUS:
                x, z = tx + dx / distance * _STAND_RADIUS, tz + dz / distance * _STAND_RADIUS
    return (x, z)


def _farthest_pair(points):
    best = (points[0], points[0])
    best_distance = -1.0
    for index, a in enumerate(points):
        for b in points[index + 1:]:
            distance = math.hypot(a[0] - b[0], a[2] - b[2])
            if distance > best_distance:
                best, best_distance = (a, b), distance
    return best


def _center(pos: BlockPos) -> tuple[float, float, float]:
    return (pos.x + 0.5, pos.y + 0.5, pos.z + 0.5)


def _aim_at(eye, point) -> tuple[float, float]:
    """Minecraft-convention angles: yaw 0 faces +z, pitch positive looks down."""
    dx, dy, dz = point[0] - eye[0], point[1] - eye[1], point[2] - eye[2]
    yaw = math.degrees(math.atan2(-dx, dz))
    pitch = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    return (yaw, pitch)


def _unwrap(yaw: float, near: float) -> float:
    """The representation of a yaw angle closest to a reference — keeps yaw continuous."""
    return yaw + 360.0 * round((near - yaw) / 360.0)


def _angle_between(current, aim) -> float:
    delta_yaw = abs(_unwrap(aim[0], near=current[0]) - current[0])
    return max(delta_yaw, abs(aim[1] - current[1]))


def _on_block(pos: BlockPos) -> CrosshairTarget:
    return CrosshairTarget(block_pos=pos, face="top", entity=None)
