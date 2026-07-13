"""The assist layer's material math — pure data in, pure data out (D7 §4)."""
from mica.assist.sufficiency import (
    GATHER_CAP, gather_plan, sufficiency, template_requirements,
)
from mica.perception.templates import REQ_SOLID, instance


def test_requirements_come_from_the_smallest_instance():
    needs = template_requirements("animal_husbandry")
    # fence pen (16 fence cells) is smaller than the post-and-rail pen
    assert needs == {"oak_fence": len(instance("fence pen").cells)}
    house = template_requirements("house")
    assert house == {REQ_SOLID: len(instance("treehouse").cells)}   # smallest house


def test_definitions_only_subtype_is_honestly_none():
    assert template_requirements("mining_excavation") is None
    assert sufficiency(None) is None


def test_sufficiency_pools_both_inventories_and_names_whats_missing():
    needs = {"oak_fence": 16, REQ_SOLID: 10}
    result = sufficiency(needs,
                         player_inventory=[("minecraft:oak_fence", 10), ("minecraft:stone", 4)],
                         agent_inventory={"oak_fence": 2, "cobblestone": 3, "iron_pickaxe": 1})
    assert result["have"] == {"oak_fence": 12, REQ_SOLID: 7}     # pickaxe never counts
    assert result["missing"] == {"oak_fence": 4, REQ_SOLID: 3}
    assert result["can_help"] and 0 < result["agent_share"] < 1


def test_exact_needs_draw_before_the_solid_budget():
    # 4 stone in stock, 4 exact-stone cells + 4 any-solid cells: the exact cells
    # take the stone, the budget goes wanting — never double-counted
    result = sufficiency({"stone": 4, REQ_SOLID: 4}, player_inventory={"stone": 4})
    assert result["have"] == {"stone": 4, REQ_SOLID: 0}
    assert result["missing"] == {REQ_SOLID: 4}


def test_declared_targets_never_reach_evidence_training_or_heads():
    # The D7 quarantine, made structural (review checklist item 5): a declaration is
    # the answer sheet, so no evidence builder, trainer, or likelihood head may even
    # MENTION its store. It feeds eval, material planning, and diagnostics only.
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    for rel in ("mica/data/training_pairs.py", "mica/data/source_b.py",
                "mica/intent/heads_v0.py", "mica/intent/heads_v1.py",
                "mica/intent/features.py", "mica/intent/tracker.py",
                "mica/perception/evidence2d.py", "mica/perception/evidence3d.py",
                "scripts/train_heads.py", "scripts/train_arm1.py"):
        source = (root / rel).read_text(encoding="utf-8")
        assert "declared_target" not in source, f"{rel} touches the declared target"


def test_gather_plan_is_whitelisted_capped_and_honest():
    plan = gather_plan({"oak_fence": 100, REQ_SOLID: 5, "minecraft:diamond_block": 2})
    fence = next(e for e in plan if e["for"] == "oak_fence")
    assert fence["gather"] == "oak_log" and fence["count"] == GATHER_CAP   # raw source, capped
    solid = next(e for e in plan if e["for"] == REQ_SOLID)
    assert solid["gather"] == "cobblestone"
    diamond = next(e for e in plan if e["for"] == "minecraft:diamond_block")
    assert diamond["gather"] is None and "human" in diamond["note"]        # never silently dropped
    assert plan[0]["for"] == "oak_fence"                                   # biggest shortfall first
