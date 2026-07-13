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
canonical path); it never re-derives pipeline facts. All loading runs in a
background thread as one pass over the files, so the window never freezes on a
refresh.

## Tabs

- **Inbox** — a list of every session still needing a label with its chain-state
  colour, and a detail pane for the selected one. A **"What was built"** section shows
  the *in-game POV screenshot* (present even for unprocessed captures) next to the
  *voxel build render* — click either to open it full size, or **Open frames** /
  **Open all graphs** for the folders. Then facts, the matcher verdict once labeled,
  and the label form. The subtype chips come in two groups: **matcher templates**
  (the styles the 3D matcher can actually recognize) and **more styles** in grey —
  the wider documented reach of each category. A grey style is a fine label:
  matcher agreement is category-level, so it still counts toward the cascade.
  Free text is allowed too. Buttons: **Label** (one session now), **Add to queue**
  (stage it for the batch), **Skip** (throwaways), and — for a *not processed*
  capture — **Process evidence** (runs `after_game --evidence-only`).
- **Status** — matcher-agreed capture count and cascade readiness, then two panels:
  **Models on disk** (which trained models are live — heads, arm1, decoder, gate —
  with their training date and key numbers, plus the `pre_cascade_a` "before" bank)
  and the **Cascade checklist** — every session sorted into *feeds the cascade*
  (labeled + kept + matcher-agreed, with its pair count) or *excluded, with the
  reason* (no label yet, skipped, quarantined, gate failed, contested, discarded).
  Nothing unlabeled can slip into a retrain unseen. The cascade retrain itself
  stays a manual terminal step by design; the tab shows the command, no button.
- **History** — an audit view for the whole batch before a cascade. Select a **labeled**
  session to see its build pictures + matcher verdict, with the form enabled to
  **Re-label** it if it's wrong (re-runs the matcher). Skipped sessions list below with
  **Un-skip**.

## Batch labeling — one matcher run for many sessions

Every single label reloads the matcher's models, which is what makes a label take
minutes. The **batch queue** (under the Inbox session list) fixes that: pick goal +
subtype for a session, press **Add to queue**, repeat, then **Label all (N)** — the
staged labels are saved and the matcher runs **once** for the whole queue
(`after_game.py --batch`). One model load instead of N. Sessions whose evidence
isn't regenerated yet are refused with a printed reason, never silently sent
through GPU work.

## What a label does

Pressing **Label** runs `after_game.py --session <id> --goal <g> --subtype <s>` in
the background (untick *VLM cross-check* to skip the local qwen2.5vl step when ollama
is off). While it runs, the bottom activity bar shows an animated indeterminate
progress + a live elapsed timer (there is no honest percentage — the matcher doesn't
report a fraction — so it shows "working, and for how long", not a made-up number).
Then the matcher verdict appears (agrees / contests, pairs written); the session
moves to History and Status updates.

## Files

- `gui.py` — the Tkinter desktop window (Inbox / Status / History).
- `sessions.py` — data + logic; imports `scripts/after_game.py` for all pipeline
  facts. Pure helpers (`derive_state`, `inbox_from`, `subtype_presets`,
  `batch_problem`, `checklist_from`) are unit-tested. UI-agnostic, so a different
  front-end is just another caller.
- `skipped.json` — the tool's own list of set-aside sessions (created on first skip).
- `staged_labels.json` — the last batch queue handed to `after_game.py --batch`.

## Tests

```
python -m pytest after_session_tool/tests/
```
