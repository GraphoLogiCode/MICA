"""The failure mode of session fabric-20260704-022944, pinned as tests: when the
build happens outside the capture region, every structure feature reads zero — and
the pipeline must now SAY so (crop_escapes in status and summary) instead of leaving
silent zeros. Plus the snapshot monitor's divergence cap: a blown region can diverge
on hundreds of thousands of cells; memory stays bounded while every count (and the
quarantine verdict) stays exact."""
from mica.capture.sample_builds import wall_row_build
from mica.capture.synthetic import generate_session
from mica.intent.tracker import TrackerParams
from mica.live_pipeline import LivePipeline, PipelineConfig, RecordSinks
from mica.perception.evidence3d import Evidence3DStream
from mica.perception.voxel_replay import Region, ReplayWorld, SnapshotMonitor


def _far_away_region():
    # The build happens near (0..4, 64, 0); this region is centered hundreds of
    # blocks away — exactly what a spawn-anchored region looks like after a teleport.
    return Region(1000, 0, 1000, 1048, 98, 1048)


def test_build_outside_region_reads_zero_and_reports_escapes():
    session = generate_session(wall_row_build())
    d2 = Evidence3DStream(ReplayWorld(_far_away_region(), {}))
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=RecordSinks(), d2=d2))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        pipeline.on_moment(packet, None)
    summary = pipeline.finish()
    # blind: nothing built as far as the features can see
    assert d2._world.built() == {}
    # ...and loudly reported: all five placements escaped the crop
    assert summary["crop_escapes"] == 5
    assert pipeline.status()["crop_escapes"] == 5


def test_build_inside_region_reports_no_escapes():
    session = generate_session(wall_row_build())
    d2 = Evidence3DStream(ReplayWorld(Region(-8, 56, -8, 12, 72, 8), {}))
    pipeline = LivePipeline(PipelineConfig(params=TrackerParams(), sinks=RecordSinks(), d2=d2))
    for packet in sorted(session.packets, key=lambda p: p.tick):
        pipeline.on_moment(packet, None)
    assert pipeline.finish()["crop_escapes"] == 0
    assert len(d2._world.built()) == 5


def test_monitor_divergences_are_capped_but_counts_stay_exact():
    region = Region(0, 0, 0, 20, 20, 20)
    monitor = SnapshotMonitor(region, {})
    # A snapshot claiming 250 water cells (class a) and one stone cell (class c)
    # where the replayed world has air — far more than the example cap.
    cells = {(x, y, 0): "minecraft:water" for x in range(20) for y in range(13)}
    cells[(1, 1, 1)] = "minecraft:stone"
    found = monitor.on_snapshot(100, cells)
    assert len(found) == 261
    assert monitor.divergence_count == 261
    assert monitor.class_counts == {"a": 260, "b": 0, "c": 1}
    assert len(monitor.divergences) == SnapshotMonitor._DIVERGENCE_EXAMPLES
    # the class-c may lie beyond the kept examples — the verdict must not care
    assert monitor.quarantined


def test_monitor_quarantine_from_class_c_past_the_cap():
    region = Region(0, 0, 0, 30, 30, 30)
    monitor = SnapshotMonitor(region, {})
    water = {(x, y, 0): "minecraft:water" for x in range(30) for y in range(10)}
    monitor.on_snapshot(50, water)              # 300 class-a: fills the example cap
    assert not monitor.quarantined
    monitor.on_snapshot(90, {(2, 2, 2): "minecraft:stone"})   # class-c, past the cap
    assert monitor.quarantined
    assert monitor.class_counts["c"] == 1
