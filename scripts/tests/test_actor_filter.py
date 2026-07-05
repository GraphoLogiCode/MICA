"""The A7 / do-operator filter: the agent's own block events are interventions, never
human evidence. Verified end to end on a mixed-actor session — D1 never scores or
consumes them, D2's feature world never contains them, the human's evidence is exactly
what a human-only session would produce, and the snapshot monitor's shadow world (which
must match the REAL world) keeps them."""
from mica.capture.sample_builds import assisted_build
from mica.capture.synthetic import generate_session
from mica.contracts.b0 import AGENT_ACTOR, is_agent_actor
from mica.contracts.b1 import MacroAction
from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks
from mica.intent.tracker import TrackerParams
from mica.perception.evidence2d import Evidence2DStream, evidence_stream
from mica.perception.evidence3d import Evidence3DStream, build_evidence3d
from mica.perception.voxel_replay import ReplayWorld, SnapshotMonitor, region_around_events


def _session():
    return generate_session(assisted_build())


def _split_ids(session):
    human, agent = [], []
    for packet in session.packets:
        for event in packet.server.block_events:
            (agent if is_agent_actor(event.actor) else human).append(event.event_id)
    return human, agent


def test_the_pinned_actor_names_are_the_predicate():
    assert is_agent_actor(AGENT_ACTOR)             # MICA_AI, the agent's username
    assert is_agent_actor("MICA_AI_1")             # replicas: MICA_AI_1, _2, ...
    assert is_agent_actor("MICA_AI_12")
    assert not is_agent_actor("MICA_AIX")          # suffix must follow an underscore
    assert not is_agent_actor("HumanBuilder")      # the rig's pinned human username
    assert not is_agent_actor("Player940")         # older real captures' human actor
    assert not is_agent_actor("fire")              # world dynamics
    assert not is_agent_actor("synthetic-assisted")


def test_agent_events_are_never_scored_or_consumed():
    session = _session()
    human_ids, agent_ids = _split_ids(session)
    assert agent_ids                                # the fixture really is mixed
    records = list(evidence_stream(session.packets))
    consumed = [eid for r in records for eid in r.event_ids]
    assert sorted(consumed) == sorted(human_ids)    # every human event, exactly once
    assert not set(consumed) & set(agent_ids)       # and no agent event, ever
    # the human's five placements are five PLACE corrections — the agent's three
    # interleaved blocks created no extra build actions
    places = [r for r in records if r.scored and r.a_hat is MacroAction.PLACE]
    assert len(places) == 5


def test_feature_world_excludes_agent_cells_but_monitor_keeps_them():
    session = _session()
    corrections = tuple(r for r in evidence_stream(session.packets) if r.scored)
    events = {e.event_id: e for p in session.packets for e in p.server.block_events}
    region = region_around_events(session)
    d2 = Evidence3DStream(ReplayWorld(region, {}))
    d2.remember(events.values())
    for correction in corrections:
        d2.on_correction(correction)
    built = d2._world.built()
    assert all(z != 3 for (_, _, z), _ in [(c, b) for c, b in built.items()])  # agent row absent
    assert len(built) == 5                          # exactly the human's wall row
    assert d2.pending_event_ids == ()               # no agent ids parked as "unconsumed"

    monitor = SnapshotMonitor(region, {})
    for packet in session.packets:
        monitor.apply_packet(packet)
    monitor.drain_pending()                         # stream over (live events hold for compares)
    shadow_built = monitor.world.built()
    assert len(shadow_built) == 8                   # human 5 + agent 3: reality has both


def test_mixed_session_evidence_equals_the_human_only_evidence():
    # same client behavior, same human events -> the B1 stream must be identical in
    # everything except the event-id numbering (ids are global across actors)
    mixed = _session()
    human_only = generate_session(assisted_build())
    stripped_packets = []
    import dataclasses
    for packet in human_only.packets:
        server = dataclasses.replace(
            packet.server,
            block_events=tuple(e for e in packet.server.block_events
                               if not is_agent_actor(e.actor)))
        stripped_packets.append(dataclasses.replace(packet, server=server))
    mixed_records = list(evidence_stream(mixed.packets))
    clean_records = list(evidence_stream(stripped_packets))
    assert len(mixed_records) == len(clean_records)
    for a, b in zip(mixed_records, clean_records):
        assert (a.tick_range, a.a_hat, a.scored, a.idle, a.state_feats, a.focus) == \
               (b.tick_range, b.a_hat, b.scored, b.idle, b.state_feats, b.focus)
        assert len(a.event_ids) == len(b.event_ids)   # same consumption shape


def test_live_pipeline_end_to_end_on_the_mixed_session():
    session = _session()
    human_ids, agent_ids = _split_ids(session)
    d2 = Evidence3DStream(ReplayWorld(region_around_events(session), {}))
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=RecordSinks(), d2=d2))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        pipeline.on_moment(packet, None)
    summary = pipeline.finish()
    assert summary["consumed_event_ids"] == len(human_ids)
    assert summary["unconsumed_event_ids"] == []      # agent events never entered the pool
    assert summary["records"]["corrections"] == summary["records"]["scored"]
