"""Migrate labels.json subtypes to taxonomy v3 (one-time, 2026-07-10).

    python scripts/migrate_labels_v3.py            # dry run: print the diff, change nothing
    python scripts/migrate_labels_v3.py --apply    # write the migrated labels.json

The mapping is the Taxonomy v3 vault note's table ("Taxonomy v3 - Subtype Expansion"
§3). Every entry keeps its original spelling in a new `subtype_v2` field, so nothing
about the old labels is lost. A subtype the table does not cover is FLAGGED and left
untouched — the user decides those; this script never guesses a label.

labels.json is only the REAL captures' store; the scripted corpus regenerates its own
labels with v3 subtypes (scripts/make_scripted_corpus.py, the standing no-flag command).
"""
from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LABELS = os.path.join(_ROOT, "capture", "raw", "labels.json")

# Old spelling (lowercased for lookup) -> v3 subtype. The vault note's table.
_MAPPING = {
    "cabin": "house", "longhouse": "house", "treehouse": "house", "house": "house",
    "railed bridge": "bridge", "flat span": "bridge", "bridge": "bridge",
    "road": "path_road",
    "crop field": "crop_farm", "crop_field": "crop_farm", "bordered plot": "crop_farm",
    "fence pen": "animal_husbandry", "post-and-rail pen": "animal_husbandry",
    "square tower": "watchtower", "square_tower": "watchtower",
    "pillar tower": "watchtower",
    "perimeter wall": "perimeter_wall",
    "flower garden": "landscaping_garden",
    "fountain": "fountain_water_feature",
}


def migrate(labels: dict) -> tuple[dict, list[tuple[str, str, str]], list[tuple[str, str]]]:
    """(migrated labels, [(sid, old, new)] changes, [(sid, old)] flagged). Pure, so
    the dry run and the real run cannot disagree."""
    changed, flagged = [], []
    out = {}
    for sid, entry in labels.items():
        entry = dict(entry)
        old = entry.get("subtype", "")
        new = _MAPPING.get(old.strip().lower())
        if new is None:
            flagged.append((sid, old))
        elif "subtype_v2" in entry:
            pass                                     # already migrated — never twice
        else:
            entry["subtype_v2"] = old
            entry["subtype"] = new
            changed.append((sid, old, new))
        out[sid] = entry
    return out, changed, flagged


def main() -> int:
    apply = "--apply" in sys.argv
    with open(_LABELS, encoding="utf-8") as handle:
        labels = json.load(handle)
    migrated, changed, flagged = migrate(labels)

    print(f"labels.json: {len(labels)} entries")
    for sid, old, new in changed:
        print(f"  {sid}: '{old}' -> '{new}'")
    for sid, old in flagged:
        print(f"  FLAGGED {sid}: '{old}' has no v3 mapping — decide it in the "
              "labeling tool (left untouched)")
    if not changed:
        print("  nothing to migrate")
        return 0
    if not apply:
        print("\ndry run — nothing written. Re-run with --apply to write.")
        return 0

    # Write-then-rename, like every writer of this file: a crash mid-write must
    # never corrupt the one store the labeling chain starts from.
    tmp = _LABELS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(migrated, handle, indent=1)
    os.replace(tmp, _LABELS)
    print(f"\nwrote {len(changed)} migrated entries ({len(flagged)} flagged, untouched)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
