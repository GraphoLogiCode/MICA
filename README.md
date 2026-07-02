# MICA — implementation

Minecraft Intention-Aware Collaborative Agent. Design and math live in the
Obsidian vault (`MICA Research/`); this repo is the code.

## Where this is in the plan

Build order is **B0 first, then the perception streams** (D1/D2), because both
streams consume B0. This commit lays down B0 — the `ObservationPacket` capture
contract — plus the verification that proves a captured stream carries every
field the perception streams need.

- `mica/contracts/` — the B0 schema, the tick master-clock, the session manifest.
- `mica/capture/` — a synthetic B0 generator (scripted builds), so the contract
  and validators run end-to-end before a real Fabric/MineRL rig exists.
- `mica/validation/` — `coverage` (does B0 supply what D1/D2 consume?) and
  `sync_report` (is a stream temporally well-formed and lossless?).

## Run

```sh
python -m pytest                 # contract + validation tests
python scripts/make_proof_logs.py   # regenerate proof_logs/ (D0 "no log, not done")
```

## Conventions

Clean-code rules in `.agents/rules/research-code.md` apply. Indexing discipline
from the math review: no variable named `t` — only `tick` and the evidence-step
index `k`.
