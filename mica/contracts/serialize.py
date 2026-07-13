"""One home for turning contract records into the dicts the JSONL logs store.

Every stage writes its records as JSON lines, and two different writers (the offline
scripts and the live runner) must produce byte-identical lines for the same record —
that is the golden-equivalence rule. So the record→dict shaping lives here, once,
instead of being copied into each writer. Field order matters: json.dumps keeps dict
insertion order, and the banked artifacts were written in exactly this order.

The reverse direction (dict→record) stays where it always was: packets in
capture/jsonl_ingest.py, fused records in contracts/b3.py (fuse_dicts). The one
exception is packet_to_dict below — Python normally only READS packets (the mod
writes them), but the wire-format test fixtures need to play the mod's part.
"""
from __future__ import annotations

import math


def packet_to_dict(packet) -> dict:
    """One ObservationPacket as the JSON dict the mod would have written for it.

    Exact inverse of jsonl_ingest.packet_from_dict: round-tripping a packet through
    the two gives the same packet back. Used by the synthetic wire encoder so live
    protocol tests can run without a game.
    """
    c, s = packet.client, packet.server
    return {
        "tick": packet.tick,
        "wallclock_ms": packet.wallclock_ms,
        "client": {
            "capture_wallclock_ms": c.capture_wallclock_ms,
            "pov_frame": None if c.pov_frame is None else {
                "path": c.pov_frame.path, "width": c.pov_frame.width, "height": c.pov_frame.height},
            "input_state": None if c.input_state is None else {
                "keys": list(c.input_state.keys),
                "mouse_buttons": list(c.input_state.mouse_buttons),
                "mouse_dx": c.input_state.mouse_dx, "mouse_dy": c.input_state.mouse_dy},
            "yaw": c.yaw,
            "pitch": c.pitch,
            "crosshair_target": None if c.crosshair_target is None else {
                "block_pos": None if c.crosshair_target.block_pos is None else [
                    c.crosshair_target.block_pos.x, c.crosshair_target.block_pos.y,
                    c.crosshair_target.block_pos.z],
                "face": c.crosshair_target.face, "entity": c.crosshair_target.entity},
            "held_item": c.held_item,
            "hotbar": None if c.hotbar is None else list(c.hotbar),
            "gui_open": c.gui_open,
        },
        "server": {
            "player_pos": None if s.player_pos is None else [
                s.player_pos.x, s.player_pos.y, s.player_pos.z],
            "block_events": [
                {"event_id": e.event_id, "pos": [e.pos.x, e.pos.y, e.pos.z],
                 "block_type": e.block_type, "op": e.op.value, "actor": e.actor}
                for e in s.block_events],
            "inventory_delta": [[item, count] for item, count in s.inventory_delta],
            "dimension": s.dimension,
            "biome": s.biome,
        },
    }


def evidence2d_to_dict(ev) -> dict:
    """A B1 record as its JSONL line dict (one scored or context Evidence2D)."""
    sf, f = ev.state_feats, ev.focus
    state = {
        "held_item": sf.held_item, "hotbar": list(sf.hotbar), "pos_delta": list(sf.pos_delta),
        "yaw_delta": sf.yaw_delta, "pitch_delta": sf.pitch_delta, "recent_actions": list(sf.recent_actions),
    }
    if sf.inventory is not None:
        # Written only when a sample exists: an old capture's regenerated line stays
        # byte-identical to its banked one (the golden-equivalence proof cares).
        state["inventory"] = [[item, count] for item, count in sf.inventory]
        state["inventory_tick"] = sf.inventory_tick
    return {
        "tick_range": list(ev.tick_range),
        "a_hat": ev.a_hat.value, "a_hat_conf": ev.a_hat_conf, "idle": ev.idle, "scored": ev.scored,
        "event_ids": list(ev.event_ids),
        "state_feats": state,
        "focus": {"block": None if f.block is None else [f.block.x, f.block.y, f.block.z],
                  "dwell_ticks": f.dwell_ticks},
        "h2d": list(ev.h2d) if ev.h2d is not None else None,
        "s_goal": list(ev.s_goal) if ev.s_goal is not None else None,
    }


def evidence3d_to_dict(record) -> dict:
    """A B2 record as its JSONL line dict (one Evidence3D per correction)."""
    g = record.global_feats
    return {
        "tick": record.tick,
        "event_ids": list(record.event_ids),
        "scored": record.scored,
        "per_goal": {
            goal: {
                "comp": feats.comp, "edit_distance": feats.edit_distance, "fit": feats.fit,
                "pose": {"dx": feats.pose.dx, "dz": feats.pose.dz, "rot": feats.pose.rot},
                "subtype": feats.subtype,
                "delta_comp": feats.delta_comp,
            }
            for goal, feats in record.per_goal.items()
        },
        "global": {
            "built_count": g.built_count, "bbox": list(g.bbox) if g.bbox else None,
            "centroid": list(g.centroid) if g.centroid else None,
            "planar_runs": g.planar_runs, "has_enclosure": g.has_enclosure,
            "symmetry": g.symmetry, "symmetry_support": g.symmetry_support,
            "edit_locality": g.edit_locality,
        },
        "voxel_patch": None,   # recomputable by replay; materialized when D3 trains the encoder
        "h3d": list(record.h3d) if record.h3d is not None else None,
    }


def fused_to_dict(fused) -> dict:
    """A B3 record as a JSONL line dict — the live runner's fused proof log.

    Same sub-shapes as the B1/B2 lines above, so anything that can read those can
    read this. The B1 channels ride along in full; per_goal/global keep B2's form.
    """
    sf, f, g = fused.state_feats, fused.focus, fused.global_feats
    state = {
        "held_item": sf.held_item, "hotbar": list(sf.hotbar), "pos_delta": list(sf.pos_delta),
        "yaw_delta": sf.yaw_delta, "pitch_delta": sf.pitch_delta, "recent_actions": list(sf.recent_actions),
    }
    if sf.inventory is not None:
        state["inventory"] = [[item, count] for item, count in sf.inventory]
        state["inventory_tick"] = sf.inventory_tick
    return {
        "tick": fused.tick,
        "event_ids": list(fused.event_ids),
        "a_hat": fused.a_hat.value,
        "idle": fused.idle,
        "state_feats": state,
        "focus": {"block": None if f.block is None else [f.block.x, f.block.y, f.block.z],
                  "dwell_ticks": f.dwell_ticks},
        "global": {
            "built_count": g.built_count, "bbox": list(g.bbox) if g.bbox else None,
            "centroid": list(g.centroid) if g.centroid else None,
            "planar_runs": g.planar_runs, "has_enclosure": g.has_enclosure,
            "symmetry": g.symmetry, "symmetry_support": g.symmetry_support,
            "edit_locality": g.edit_locality,
        },
        "s_goal": list(fused.s_goal) if fused.s_goal is not None else None,
        "per_goal": {
            goal: {
                "comp": feats.comp, "edit_distance": feats.edit_distance, "fit": feats.fit,
                "pose": {"dx": feats.pose.dx, "dz": feats.pose.dz, "rot": feats.pose.rot},
                "subtype": feats.subtype,
                "delta_comp": feats.delta_comp,
            }
            for goal, feats in fused.per_goal.items()
        },
        "h2d": list(fused.h2d) if fused.h2d is not None else None,
        "h3d": list(fused.h3d) if fused.h3d is not None else None,
    }


def belief_to_dict(tick: int, kind: str, belief: dict, normalizer: float | None = None) -> dict:
    """One belief log line: the full ten-number belief plus the read-off summaries.

    `kind` says what produced the line: "correction" (an observation moved the belief)
    or "drift" (a display-only prediction between observations — never fed back).
    The belief's keys are (goal, mode) pairs; JSON keys must be strings, so they are
    written as "goal|mode". The summaries are recomputed by plain sums here rather
    than imported from the tracker, because contracts must not depend on upper layers.
    """
    goals = sorted({g for g, _ in belief})
    modes = sorted({z for _, z in belief})
    by_goal = {g: sum(belief[(g, z)] for z in modes) for g in goals}
    top_goal = max(by_goal, key=by_goal.get)
    line = {
        "tick": tick,
        "kind": kind,
        "belief": {f"{g}|{z}": belief[(g, z)] for g in goals for z in modes},
        "top_goal": top_goal,
        "p_top_goal": by_goal[top_goal],
        "p_z1": sum(belief[(g, 1)] for g in goals) if 1 in modes else 0.0,
        "entropy": -sum(p * math.log(p) for p in belief.values() if p > 0.0),
    }
    if normalizer is not None:
        line["normalizer"] = normalizer
    return line
