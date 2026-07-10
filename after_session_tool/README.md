# After-Session Tool

A small **desktop app** for the one manual step left in the MICA pipeline: giving
each played session its **goal + subtype** label. The rig automation already does
everything else (evidence, agent scan, report graphs, cascade-readiness) when a
session ends; this window just makes the label a click instead of a terminal command.

## Run

```
python after_session_tool/gui.py
```

Stdlib only — Tkinter ships with Python, so there are no dependencies. It reads the
pipeline's files and, on a label, shells out to `scripts/after_game.py` (the
canonical path); it never changes any pipeline code.

## Tabs

- **Inbox** — a list of every session still needing a label with its chain-state
  colour, and a detail pane for the selected one. A **"What was built"** section shows
  the *in-game POV screenshot* (present even for unprocessed captures) next to the
  *voxel build render* — click either to open it full size, or **Open frames** /
  **Open all graphs** for the folders. Then facts, the matcher verdict once labeled,
  and the label form (goal dropdown + subtype with taxonomy preset chips + a
  VLM-cross-check toggle). Buttons: **Label** (unlocks once a session is *ready*),
  **Skip** (throwaways), and — for a *not processed* capture — **Process evidence**
  (runs `after_game --evidence-only` to make it ready).
- **Status** — matcher-agreed capture count and cascade readiness. The cascade retrain
  stays a manual terminal step by design; the tab shows the command, no button.
- **History** — an audit view for the whole batch before a cascade. Select a **labeled**
  session to see its build pictures + matcher verdict, with the form enabled to
  **Re-label** it if it's wrong (re-runs the matcher). Skipped sessions list below with
  **Un-skip**.

## What a label does

While a label (or Process evidence) runs, a bottom activity bar shows an animated
indeterminate progress + a live elapsed-seconds timer (there is no honest percentage —
the matcher's model-load + match doesn't report a fraction — so it shows "working, and
for how long", not a made-up number). It clears and the verdict appears when done.

Submitting runs `after_game.py --session <id> --goal <g> --subtype <s>` in the
background (untick *VLM cross-check* to skip the local qwen2.5vl step when ollama is
off). The card shows *labeling…*, then the matcher verdict (agrees / contests, pairs
written); the session moves to History and Status updates.

## What a label does

Pressing **Label** runs `after_game.py --session <id> --goal <g> --subtype <s>` in
the background (untick *VLM cross-check* to skip the local qwen2.5vl step when ollama
is off). The state shows *labeling…*, then the matcher verdict (agrees / contests,
pairs written); the session moves to History and Status updates.

## Files

- `gui.py` — the Tkinter desktop window (Inbox / Status / History).
- `sessions.py` — data + logic; imports `scripts/after_game.py` for all pipeline
  facts. Pure helpers (`derive_state`, `inbox_from`, `subtype_presets`) are unit-tested.
  UI-agnostic, so a different front-end is just another caller.
- `skipped.json` — the tool's own list of set-aside sessions (created on first skip).

## Tests

```
python -m pytest after_session_tool/tests/
```
