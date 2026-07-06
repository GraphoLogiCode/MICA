"""The session-path resolver: date parsing, nested vs legacy-flat resolution, and
the relocation-tolerant frame lookup."""
import os

import pytest

from mica.capture import session_store


def test_session_date_parses_the_fabric_id():
    assert session_store.session_date("fabric-20260706-001126") == "2026-07-06"
    assert session_store.session_date("fabric-20260101-235959") == "2026-01-01"


def test_session_date_rejects_non_dated_ids():
    for bad in ("livetest-0001", "scripted-cabin-7000", "fabric-bad"):
        with pytest.raises(ValueError):
            session_store.session_date(bad)


def test_nested_layout_resolves_when_present(tmp_path):
    root = str(tmp_path)
    nested = tmp_path / "2026-07-06" / "fabric-20260706-001126"
    nested.mkdir(parents=True)
    got = session_store.session_dir("fabric-20260706-001126", root)
    assert got == str(nested)
    assert session_store.session_file("fabric-20260706-001126", ".evidence2d.jsonl", root) \
        == str(nested / "fabric-20260706-001126.evidence2d.jsonl")


def test_legacy_flat_fallback(tmp_path):
    root = str(tmp_path)
    (tmp_path / "fabric-20260706-001126.jsonl").write_text("{}", encoding="utf-8")
    # no dated dir exists -> resolve to the flat root
    assert session_store.session_dir("fabric-20260706-001126", root) == root


def test_new_dated_session_defaults_to_organized(tmp_path):
    root = str(tmp_path)
    # nothing on disk yet -> a fresh dated session resolves to its organized dir
    got = session_store.session_dir("fabric-20260706-090000", root)
    assert got == os.path.join(root, "2026-07-06", "fabric-20260706-090000")


def test_non_dated_id_always_flat(tmp_path):
    root = str(tmp_path)
    assert session_store.session_dir("livetest-0001", root) == root


def test_resolve_frame_prefers_baked_then_relocated(tmp_path):
    session = "fabric-20260706-001126"
    capture_dir = tmp_path / "2026-07-06" / session
    frames = capture_dir / session / "frames"
    frames.mkdir(parents=True)
    (frames / "0.png").write_bytes(b"x")

    # a stale baked path (old flat location) still resolves via the frames dir
    stale = str(tmp_path / session / "frames" / "0.png")
    assert session_store.resolve_frame(stale, session, str(capture_dir)) \
        == str(frames / "0.png")
    # a live baked path is returned as-is
    live = str(frames / "0.png")
    assert session_store.resolve_frame(live, session, str(capture_dir)) == live
    # a frame that exists nowhere resolves to None
    assert session_store.resolve_frame(
        str(tmp_path / "gone" / "9.png"), session, str(capture_dir)) is None
