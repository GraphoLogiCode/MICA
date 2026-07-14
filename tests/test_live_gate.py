"""The D5 live gate runner: demo config holds, degradation is a state, the human
cell tracking is actor-honest, and the Arm-0 neutral derivation does what its
docstring says."""
import json

import pytest

from decoder_fixtures import fused_record
from mica.capture.synthetic import ScriptedPlacement  # noqa: F401 (fixture parity)
from mica.contracts.b0 import BlockEvent, BlockOp, BlockPos
from mica.gate import commit
from mica.gate.live_loop import LiveGateRunner, gate_ready
from mica.intent.tracker import uniform_belief


@pytest.fixture()
def runner(tmp_path):
    if not gate_ready():
        pytest.skip("no decoder/gate freeze on this machine")
    from mica.intent import heads_v1

    gate = LiveGateRunner(str(tmp_path / "trace.jsonl"), heads_v1.tracker_params(),
                          demo=True)
    yield gate
    gate.close()


def test_demo_config_disables_placement(runner):
    assert runner.fsm.config.place_low_risk_enabled is False
    assert runner.fsm.config.execute_chunk_enabled is False


def test_no_correction_yet_reads_as_observe(runner):
    block = runner.read(uniform_belief(), None, {})
    assert block["state"] == "observe" and "no corrections" in block["reason"]


def test_degraded_read_still_writes_a_trace_row(runner, tmp_path):
    runner.read(uniform_belief(), None, {"tick": 5})
    rows = [json.loads(line) for line in
            (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["reason"] == "no corrections yet"
    assert rows[-1]["chosen_state"] == "observe"
    assert rows[-1]["committed_actions"] == []


def test_between_correction_reads_use_a_time_advanced_belief(runner):
    # The belief only moves at corrections; a read a minute later must gate on
    # what the kernels say is true NOW, not on the frozen correction-time mass.
    belief = uniform_belief()
    peak = next(iter(belief))
    rest = 0.1 / (len(belief) - 1)
    peaked = {key: (0.9 if key == peak else rest) for key in belief}
    at_correction = runner._materialized(peaked, {"tick": 100, "belief_snapshot_id": 100})
    assert at_correction == peaked            # same tick: bit-identical, replay-safe
    minute_later = runner._materialized(
        peaked, {"tick": 100 + 20 * 60, "belief_snapshot_id": 100})
    assert minute_later[peak] < peaked[peak]  # the peak decayed toward uniform
    assert abs(sum(minute_later.values()) - 1.0) < 1e-9
    no_live_tick = runner._materialized(peaked, {"belief_snapshot_id": 100})
    assert no_live_tick == peaked             # replay drivers carry no live tick


def test_read_produces_a_trace_row_and_status_block(runner, tmp_path):
    status = {"current_behavior": "idle", "player_pos": (30.0, 64.0, 30.0),
              "focus_block": None, "belief_snapshot_id": 100}
    block = runner.read(uniform_belief(), fused_record(), status)
    assert block["state"] in ("observe", "suggest", "preview", "yield")
    assert block["state"] != "place_low_risk"
    runner.trace.flush()
    assert (tmp_path / "trace.jsonl").read_text(encoding="utf-8").strip()


def test_human_cell_tracking_ignores_the_agent(runner):
    runner.ingest_events([
        BlockEvent(event_id=0, pos=BlockPos(1, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.PLACE, actor="HumanBuilder"),
        BlockEvent(event_id=1, pos=BlockPos(2, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.PLACE, actor="MICA_AI"),
        BlockEvent(event_id=2, pos=BlockPos(1, 64, 1), block_type="minecraft:stone",
                   op=BlockOp.BREAK, actor="HumanBuilder"),
    ])
    assert runner.human_cells == set()      # placed then broken; agent's never entered


# --- the place config (§9 amendment 2026-07-12) ---------------------------------

def _peaked_belief(p_top=0.66, z1_share=0.6):
    """A belief with a chosen top-goal mass and mode split. The defaults leave the
    discounted conf BELOW theta_place (the dormant regime); the directive tests
    pass a deliberative, confident shape instead — the confidence route is the
    ONLY live licensing route since the declared route's retirement (2026-07-13)."""
    belief = {key: 0.0 for key in uniform_belief()}
    rest = (1.0 - p_top) / 8
    for key in belief:
        belief[key] = rest
    belief[("habitation", 1)] = p_top * z1_share
    belief[("habitation", 0)] = p_top * (1 - z1_share)
    total = sum(belief.values())
    return {key: value / total for key, value in belief.items()}


@pytest.fixture()
def place_runner(tmp_path, monkeypatch):
    if not gate_ready():
        pytest.skip("no decoder/gate freeze on this machine")
    from mica.contracts.b5 import ProposalChunk
    from mica.decoder.grammar import Place
    from mica.gate import live_loop
    from mica.intent import heads_v1

    proposal = ProposalChunk(
        actions=(Place(dx=1, dy=0, dz=1, block="minecraft:oak_planks"),),
        token_conf=(1.0,), rationale_goal="habitation")
    monkeypatch.setattr(live_loop.decoder_model, "propose",
                        lambda model, ctx, n: (proposal, None))
    gate = LiveGateRunner(str(tmp_path / "trace.jsonl"), heads_v1.tracker_params(),
                          session_id="test-session", place=True)
    gate.model = object()                    # already "loaded": skip the real torch load
    gate.origin = (0, 64, 0)
    # agent body well outside the human's proximal radius (the F4 veto is real:
    # putting it at the human's feet turns every read into YIELD)
    gate.materials_source = lambda: ({"oak_planks": 8}, {}, (18.0, 64.0, 6.0))
    yield gate
    gate.close()


def _safe_status():
    return {"current_behavior": "idle", "player_pos": (30.0, 64.0, 30.0),
            "focus_block": None, "belief_snapshot_id": 100}


def test_place_config_arms_the_gate_but_never_the_declared_route(place_runner):
    assert place_runner.fsm.config.place_low_risk_enabled is True
    # Retired 2026-07-13 (user decision): live behavior comes from inference
    # alone — nothing may arm the declared route in a live runner.
    assert place_runner.fsm.config.declared_place_enabled is False
    assert place_runner.fsm.config.execute_chunk_enabled is False


def test_a_mid_session_declaration_changes_nothing(place_runner):
    # Even if a declaration exists for the session (entered by mistake), the
    # retired route must stay dead: conf below theta_place means NO directive.
    place_runner._declared = lambda: {"goal": "habitation", "subtype": "cabin"}
    belief = _peaked_belief()                        # conf below theta_place
    blocks = [place_runner.read(belief, fused_record(), _safe_status())
              for _ in range(4)]
    assert all(b["place"] is None for b in blocks)
    assert all(b["state"] != "place_low_risk" for b in blocks)


def test_confidence_route_emits_one_directive_then_suppresses_the_filled_cell(
        place_runner, tmp_path):
    belief = _peaked_belief(p_top=0.8, z1_share=0.15)   # deliberative + confident
    blocks = [place_runner.read(belief, fused_record(), _safe_status())
              for _ in range(4)]
    # hysteresis (M=3) holds the first reads; the directive appears exactly once
    directives = [b["place"] for b in blocks if b["place"]]
    assert len(directives) == 1
    directive = directives[0]
    assert directive["block"] == "oak_planks"
    assert directive["cell"] == [1, 64, 1]           # origin (0,64,0) + (1,0,1)
    assert directive["route"] == "confidence"        # the only live route
    # the cell is remembered: later reads of the same proposal emit nothing
    later = place_runner.read(belief, fused_record(), _safe_status())
    assert later["place"] is None
    place_runner.trace.flush()
    rows = [json.loads(line) for line in
            (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["authority"] == "place" for row in rows)
    committed = [row for row in rows if row["committed_actions"]]
    assert len(committed) == 1
    action = committed[0]["committed_actions"][0]
    assert action["reversible"] is True and action["route"] == "confidence"
    assert committed[0]["belief_snapshot_id"] is not None
    assert any("already filled by the agent" in row["reason"] for row in rows)


def test_directive_ids_never_collide_across_runner_restarts(place_runner, tmp_path):
    # The body outlives a re-attached mind and remembers executed ids — a fresh
    # runner restarting its read counter at 1 must still mint globally new ids
    # (review 2026-07-13 F2).
    from mica.intent import heads_v1

    belief = _peaked_belief(p_top=0.8, z1_share=0.15)
    first_ids = [b["place"]["id"] for b in
                 (place_runner.read(belief, fused_record(), _safe_status())
                  for _ in range(4)) if b["place"]]
    second = LiveGateRunner(str(tmp_path / "trace2.jsonl"), heads_v1.tracker_params(),
                            session_id="test-session", place=True)
    second.model = object()
    second.origin = (0, 64, 0)
    second.materials_source = place_runner.materials_source
    second._run_token = place_runner._run_token + 1   # a later start, deterministic
    second_ids = [b["place"]["id"] for b in
                  (second.read(belief, fused_record(), _safe_status())
                   for _ in range(4)) if b["place"]]
    second.close()
    assert first_ids and second_ids
    assert not set(first_ids) & set(second_ids)


def test_place_directive_counts_into_the_materials_report(place_runner, tmp_path):
    belief = _peaked_belief(p_top=0.8, z1_share=0.15)
    for _ in range(4):
        place_runner.read(belief, fused_record(), _safe_status())
    place_runner.close()
    report = json.loads(
        (tmp_path / "trace.materials_report.json").read_text(encoding="utf-8"))
    assert report["committed_places"] == 1


def test_demo_reads_carry_no_directive_and_demo_authority(runner, tmp_path):
    status = {"current_behavior": "idle", "player_pos": (30.0, 64.0, 30.0),
              "focus_block": None, "belief_snapshot_id": 100}
    block = runner.read(uniform_belief(), fused_record(), status)
    assert block["place"] is None
    runner.trace.flush()
    rows = [json.loads(line) for line in
            (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["authority"] == "demo"


def test_agent_events_feed_the_repropose_guard(runner):
    runner.ingest_events([
        BlockEvent(event_id=3, pos=BlockPos(5, 64, 5), block_type="minecraft:stone",
                   op=BlockOp.PLACE, actor="MICA_AI"),
    ])
    assert (5, 64, 5) in runner.agent_cells


# --- the async wrapper (proof-grade fix 2026-07-13): gate reads off-thread ------

class _FakeRunner:
    """Stands in for LiveGateRunner: records what reached it, returns numbered
    blocks, and can hold a read open so coalescing is observable."""

    place = False

    def __init__(self):
        import threading
        self.events = []
        self.read_statuses = []
        self.closed = False
        self.hold = threading.Event()
        self.hold.set()                     # default: reads return immediately

    def ingest_events(self, block_events):
        self.events.extend(block_events)

    def read(self, belief, fused, status):
        self.hold.wait(timeout=5.0)
        self.read_statuses.append(status["n"])
        return {"state": "observe", "n": status["n"]}

    def close(self):
        self.closed = True


def _wait_for(condition, timeout=5.0):
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if condition():
            return True
        _time.sleep(0.01)
    return False


def test_async_runner_computes_off_thread_and_publishes_latest():
    from mica.gate.live_loop import AsyncGateRunner

    fake = _FakeRunner()
    runner = AsyncGateRunner(fake)
    assert runner.latest() is None          # nothing finished yet
    runner.request(None, None, {"n": 1})
    assert _wait_for(lambda: runner.latest() is not None)
    assert runner.latest()["n"] == 1
    runner.close()
    assert fake.closed


def test_async_runner_coalesces_to_the_freshest_request():
    from mica.gate.live_loop import AsyncGateRunner

    fake = _FakeRunner()
    fake.hold.clear()                       # hold the first read open
    runner = AsyncGateRunner(fake)
    runner.request(None, None, {"n": 1})
    _wait_for(lambda: len(fake.read_statuses) == 0)   # worker is inside read 1
    for n in (2, 3, 4):                     # these arrive while 1 is in flight
        runner.request(None, None, {"n": n})
    fake.hold.set()
    assert _wait_for(lambda: runner.latest() is not None
                     and runner.latest()["n"] == 4)
    # reads 2 and 3 were REPLACED, never executed: freshest-wins, no stale queue
    assert fake.read_statuses in ([1, 4], [4])
    runner.close()


def test_async_runner_drains_events_before_the_read_in_order():
    from mica.gate.live_loop import AsyncGateRunner

    fake = _FakeRunner()
    runner = AsyncGateRunner(fake)
    runner.ingest_events(["a", "b"])
    runner.ingest_events(["c"])
    runner.request(None, None, {"n": 1})
    assert _wait_for(lambda: runner.latest() is not None)
    assert fake.events == ["a", "b", "c"]
    runner.close()


def test_async_runner_survives_a_gate_crash():
    from mica.gate.live_loop import AsyncGateRunner

    class _Crashing(_FakeRunner):
        def read(self, belief, fused, status):
            if status["n"] == 1:
                raise RuntimeError("boom")
            return super().read(belief, fused, status)

    fake = _Crashing()
    runner = AsyncGateRunner(fake)
    runner.request(None, None, {"n": 1})
    assert _wait_for(lambda: runner.latest() is not None)
    assert "gate error" in runner.latest()["reason"]
    runner.request(None, None, {"n": 2})    # the worker is still alive
    assert _wait_for(lambda: runner.latest().get("n") == 2)
    runner.close()


def test_arm0_neutral_derivation_sides():
    assert commit.derive_arm0_neutral(0.7, theta_1=0.5) == 0.52
    assert commit.derive_arm0_neutral(0.1, theta_1=0.5) == 0.2
    assert commit.legacy_gate_mean([0.6, 0.6, 0.1], [0.2, 0.9, 0.2], 0.5) == pytest.approx(1 / 3)
    commit.set_arm0_neutral(0.52)
    try:
        assert commit.arm_p_star("arm0_zero", None, 0.9, 1,
                                 __import__("mica.intent.tracker",
                                            fromlist=["TrackerParams"]).TrackerParams(),
                                 1.0) == 0.52
    finally:
        commit.set_arm0_neutral(None)
