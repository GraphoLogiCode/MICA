"""The D5 live gate loop: one gate read per ~1 Hz display tick, in-world.

This is the last D5 piece — the same staircase + FSM the counterfactual run proved,
now fed by the LIVE belief. The runner is deliberately dumb plumbing: run_live hands
it the pipeline's authoritative belief, the last fused correction, and the status
read-offs; it asks the decoder for a proposal, runs the frozen staircase and FSM,
flushes one gate_trace row, and returns the small "gate" block live_status.json
carries so the embodiment can render the state.

Demo authority (D5 §9 pin): PLACE_LOW_RISK is CONFIG-DISABLED by default — the demo
posture is observe/suggest/preview only. The §9 amendment (2026-07-12, user decision)
lifts the pin behind run_live --place: the runner then emits at most ONE placement
directive per read — the first committed action, only when it is a Place — and the
BODY executes it (this module still executes nothing; it authorizes). Live placement
is licensed by conf >= theta_place ONLY: the declared-place route was retired the
next day (2026-07-13, user decision — live behavior comes from inference alone;
declarations are post-session labels, never live inputs).

Degradation is a state, not a crash: no decoder on disk, or no correction yet, reads
as OBSERVE with the reason saying so.
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time

from ..assist import sufficiency as assist
from ..capture import session_store
from ..contracts.b0 import BlockOp, is_agent_actor
from ..decoder import context as context_builder
from ..decoder import model as decoder_model
from ..decoder.grammar import Place, Say
from ..intent.tracker import predict
from . import commit, fsm, materials, reversibility

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_GATE_META = os.path.join(_ROOT, "models", "gate_v1.json")
_MAX_ACTIONS = 8


def gate_ready() -> bool:
    return decoder_model.available() and os.path.exists(_GATE_META)


CONSENT_FRESH_MS = 60000       # a "yes" licenses the proposal for this long
CONSENT_ENTOMB_R = 1.5         # never place within this of the human's body


def consent_veto(consent, consumed, agent_cells, player_pos, inventory,
                 now_s: float) -> str | None:
    """Why a relayed consent may NOT become a directive right now — or None when
    it may. The checks are the D5 §10 consent rules, in refusal-first order:
    every returned string lands in the trace row, so a session's unhonored
    yeses are diagnosable afterward.

    The consent names its own OFFER — the {cell, block} the body voiced — and
    that offer is what gets honored (adjusted 2026-07-19, same day: the first
    cut required the CURRENT proposal to still target the consented cell, but
    the decoder re-proposes every read and can wander in the seconds between
    the offer and the yes; a valid yes then died on "different cell"). The
    current proposal plays no part here."""
    if (not isinstance(consent, dict) or consent.get("ts") is None
            or consent.get("cell") is None or not consent.get("block")):
        return "consent malformed (needs ts + cell + block)"
    if consent["ts"] in consumed:
        return "consent already honored (one block per yes)"
    if now_s * 1000 - consent["ts"] > CONSENT_FRESH_MS:
        return "consent expired"
    cell = tuple(consent["cell"])
    if cell in agent_cells:
        return "the consented cell is already filled by the agent"
    if player_pos is None:
        return "human position unknown — not placing near a person I cannot see"
    dx = cell[0] + 0.5 - player_pos[0]
    dy = cell[1] + 0.5 - player_pos[1]
    dz = cell[2] + 0.5 - player_pos[2]
    if (dx * dx + dy * dy + dz * dz) ** 0.5 <= CONSENT_ENTOMB_R:
        return "target is inside the human's personal space"
    block = materials.normalize(consent["block"])
    if inventory is not None and inventory.get(block, 0) < 1:
        return f"no {block} in stock (toss one over and say yes again)"
    return None


def proposal_summary(proposal, target, held_k: int, top_goal: str) -> str | None:
    """The advisory line's noun phrase — the body says "I could add {this}".

    Names WHAT and WHERE whenever the decoder proposed a placement. SUGGEST is by
    construction the K_commit == 0 state, so the old held_k >= 1 requirement made
    every voiced suggestion say the generic fallback and log a null summary (R-6,
    2026-07-18 review) — the whole session, nobody ever heard what the agent
    actually wanted to place."""
    if proposal is None:
        return None
    first = proposal.actions[0]
    if target is not None and isinstance(first, Place):
        summary = f"a {materials.normalize(first.block)} block at {target}"
        if held_k >= 1:
            summary += f" ({held_k} action(s) committed toward {top_goal})"
        return summary
    if held_k >= 1:
        # non-placement proposals keep the old commit-count line
        return f"{held_k} action(s) toward {top_goal}, first at {target}"
    return None


class LiveGateRunner:
    """Feed me events as they arrive and call read() once per display tick."""

    def __init__(self, trace_path: str, params, demo: bool = True,
                 preload: bool = False, materials_source=None,
                 session_id: str | None = None, gather: bool = False,
                 place: bool = False, consent_place: bool = False):
        if place:
            demo = False                          # --place IS the pin lift (§9 amendment)
        with open(_GATE_META, encoding="utf-8") as handle:
            meta = json.load(handle)
        self.staircase = commit.GateThresholds(**meta["thresholds"])
        self.delta_hat = meta["delta_hat"]["delta_hat_seconds"]
        self.fsm = fsm.GateFsm(fsm.FsmConfig(
            theta_suggest=meta["fsm"]["theta_suggest"],
            theta_place=meta["fsm"]["theta_place"],
            m_consecutive=meta["thresholds"]["m_consecutive"],
            place_low_risk_enabled=not demo,      # the §9 demo pin
            gather_enabled=gather,                # D5 §4 amendment: off unless asked
            # The declared-place route is RETIRED for live sessions (user decision
            # 2026-07-13: "no declaring in live session" — live behavior must come
            # from inference alone; declarations are post-session labels). Nothing
            # arms it: live placement is licensed by conf >= theta_place only. The
            # FSM mechanism stays defined and unit-tested for a possible future
            # pre-declared workflow, which would need its own pin.
            declared_place_enabled=False))
        self.place = place
        # The consent route (D5 §10, answered 2026-07-19): an explicit chat "yes"
        # relayed by the body licenses ONE placement of the current proposal,
        # whatever the confidence. Separate arming from --place on purpose: this
        # route never reads a threshold, so the confidence route's blockers do
        # not apply — and it must not arm them either (the FSM stays demo).
        self.consent_place = consent_place
        self._consent_consumed: set = set()       # each consent ts licenses once
        self._agent_consent: dict | None = None   # the newest relayed consent
        self._agent_inventory: dict | None = None  # the same snapshot's inventory
        self.session_id = session_id              # keys the declared-target lookup
        self.params = params                      # MUST match the live belief's knobs
        self.hysteresis = commit.CommitHysteresis()
        self.model = None
        self.origin: tuple[int, int, int] | None = None
        self.human_cells: set[tuple[int, int, int]] = set()
        # Cells this session's agent already filled (or was directed to fill). A7
        # keeps agent placements out of the evidence, so the decoder can re-propose
        # them forever — this memory is what stops a re-issued directive. Optimistic:
        # a directive's cell joins the set the read it is issued, so a slow walk to
        # the spot never triggers a duplicate directive behind it.
        self.agent_cells: set[tuple[int, int, int]] = set()
        self.directives = 0                       # placement directives issued (§7 count)
        # Directive ids must survive a mind restart: the body process outlives a
        # re-attached run_live and remembers executed ids, so a fresh runner's
        # counter restarting at 1 must not collide with the previous run's ids
        # (review 2026-07-13 F2). The start-time token makes each run's ids unique.
        self._run_token = int(time.time() * 1000)
        self.reads = 0
        self.last_decode_ms: float | None = None   # this read's decoder wall time
        # The materials constraint (D5 §4, 2026-07-06): where the agent's inventory
        # and substitution grants come from. Default: the newest agent status file
        # next to the trace. A None snapshot (replays, counterfactual, agent gone)
        # leaves the constraint INACTIVE and behavior bit-unchanged.
        self.trace_path = trace_path
        self.materials_source = materials_source or (lambda: materials.read_agent_snapshot(
            os.path.dirname(os.path.abspath(trace_path))))
        self.account = materials.MaterialsAccount()
        self._materials_block: dict | None = None
        self._gather_block: dict | None = None
        # One runner, one trace file, from the top. Append mode here once let a
        # re-attached run stack its rows under a previous run's (run_live now
        # banks the older files aside, and a repeated replay starts over).
        self.trace = open(trace_path, "w", encoding="utf-8")
        if preload and decoder_model.available():
            # Load torch + the decoder NOW, before the socket attaches. Lazy-loading
            # on the first read stalls the single pipeline thread for seconds —
            # longer than the mod's drop-oldest buffer holds — so a live session
            # would open with a burst of dropped moments at its first correction.
            self._ensure_model()

    def _ensure_model(self) -> None:
        if self.model is None:
            import torch

            self.model, _, _ = decoder_model.load()
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def ingest_events(self, block_events) -> None:
        """Track the human-built standing cells (reversibility's world half), and
        the agent's own placements (the re-propose guard's ground truth)."""
        for event in block_events:
            cell = (event.pos.x, event.pos.y, event.pos.z)
            if is_agent_actor(event.actor):
                if event.op is BlockOp.PLACE:
                    self.agent_cells.add(cell)
                continue
            if event.op is BlockOp.PLACE:
                self.human_cells.add(cell)
            else:
                self.human_cells.discard(cell)

    def _declared(self) -> dict | None:
        """The session's declared build target, if the human recorded one — read
        from its OWN store (never labels.json), per the D7 quarantine."""
        if self.session_id is None:
            return None
        try:
            path = os.path.join(session_store.RAW_ROOT, "declared_targets.json")
            with open(path, encoding="utf-8") as handle:
                return json.load(handle).get(self.session_id)
        except (OSError, ValueError):
            return None

    def _gather_inputs(self, slot, fused, snapshot, declared) -> tuple[bool, bool]:
        """(gather_wanted, gather_declared) + the status/trace gather block. The
        target is the DECLARED subtype when one exists, else the belief winner's
        style read; a definitions-only subtype yields no requirements and honestly
        no gather. Player stock comes from the evidence's leak-safe inventory
        sample; agent stock from its status snapshot. `declared` arrives from the
        caller so one gate read does one declared-target lookup."""
        self._gather_block = None
        if declared:
            subtype = declared.get("subtype")
        elif slot["top_goal"] and slot["top_goal"] in fused.per_goal:
            subtype = fused.per_goal[slot["top_goal"]].subtype
        else:
            return False, False
        result = assist.sufficiency(assist.template_requirements(subtype),
                                    fused.state_feats.inventory,
                                    snapshot[0] if snapshot else None)
        if not result or not result["missing"]:
            return False, False
        self._gather_block = {"target": subtype, "declared": declared is not None,
                              "missing": result["missing"],
                              "plan": result["gather_next"],
                              "can_help": result["can_help"]}
        return True, declared is not None

    def _decide(self, belief, fused, status):
        """(gate_read, proposal, flags, positions, raw_k, held_k) for one tick."""
        self.last_decode_ms = None             # None until this read actually decodes
        self._materials_block = None           # None until an inventory is observed
        slot = context_builder.arm3_slot(belief)
        if self.origin is None:
            self.origin = context_builder.build_origin([fused])
        proposal = reject = None
        held_k, raw_k, positions, flags, target = 0, 0, [], [], None
        prefix_reversible = None
        # One status-file read serves materials, gather, AND the agent-proximity
        # veto; one declared-target lookup serves gather AND the place bar.
        agent_snapshot = self.materials_source()
        declared = self._declared()
        gather_wanted, gather_declared = self._gather_inputs(
            slot, fused, agent_snapshot, declared)
        # Older test doubles hand back (inventory, grants); the live reader now adds
        # the agent's position as a third element (D5 §10 Q1 live half).
        agent_pos = (agent_snapshot[2]
                     if agent_snapshot is not None and len(agent_snapshot) > 2 else None)
        if decoder_model.available() and self.origin is not None:
            self._ensure_model()
            ctx = context_builder.control_context(fused, self.reads, "arm3", slot,
                                                  self.origin)
            decode_started = time.perf_counter()
            proposal, reject = decoder_model.propose(self.model, ctx, _MAX_ACTIONS)
            self.last_decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if proposal is not None:
                origin = self.origin
                human_rel = frozenset((c[0] - origin[0], c[1] - origin[1],
                                       c[2] - origin[2]) for c in self.human_cells)
                flags = reversibility.prefix_flags(proposal.actions, human_rel)
                raw_k, positions = commit.k_commit(
                    proposal, flags, "arm3", belief, slot["p_top"], self.params,
                    self.delta_hat, self.staircase)
                # The materials constraint (D5 §4): the prefix may not extend past
                # the first placement the agent has no stock for — a shortage
                # SHRINKS the build; substitution needs an explicit chat grant.
                if agent_snapshot is not None:
                    # Consent + inventory ride the same snapshot (one file read
                    # per gate read); the consent branch in read() needs both.
                    self._agent_consent = (agent_snapshot[3]
                                           if len(agent_snapshot) > 3 else None)
                    inventory, grants = agent_snapshot[0], agent_snapshot[1]
                    self._agent_inventory = inventory
                    mat_flags, missing, substituted = materials.feasibility_flags(
                        proposal.actions, inventory, grants)
                    feasible_prefix = (mat_flags.index(True) if any(mat_flags)
                                       else len(proposal.actions))
                    capped = feasible_prefix < raw_k
                    if capped:
                        raw_k = feasible_prefix
                    self.account.observe(proposal.actions, missing, substituted,
                                         capped, inventory, grants)
                    ask = None
                    if missing:
                        short_block = sorted(missing)[0]
                        candidate = materials.propose_substitute(short_block,
                                                                 inventory, grants)
                        if candidate is not None:
                            ask = {"block": short_block, "short": missing[short_block],
                                   "substitute": candidate}
                    self._materials_block = {"feasible_prefix": feasible_prefix,
                                             "missing": missing, "ask": ask}
                held_k = self.hysteresis.read(raw_k)
                if held_k >= 1:
                    prefix_reversible = not any(flags[:held_k])
                first = proposal.actions[0]
                if not isinstance(first, Say):
                    target = (first.dx + origin[0], first.dy + origin[1],
                              first.dz + origin[2])
        behavior = status.get("current_behavior")
        gate_read = fsm.GateRead(
            top_goal=slot["top_goal"], p_top=slot["p_top"],
            entropy_nats=slot["entropy_nats"], p_z1=slot["p_z1"],
            idle=behavior in (None, "idle"),
            player_pos=tuple(status["player_pos"]) if status.get("player_pos") else None,
            focus_block=tuple(status["focus_block"]) if status.get("focus_block") else None,
            target_cell=target, k_commit=held_k,
            prefix_fully_reversible=prefix_reversible,
            nothing_to_do=reject == decoder_model.NOTHING_TO_DO,
            gather_wanted=gather_wanted, gather_declared=gather_declared,
            # The agent-proximity veto is part of the place-mode posture (§9
            # amendment); demo runs stay comparable to their validated traces.
            agent_pos=agent_pos if self.place else None,
            place_declared=declared is not None)
        return gate_read, proposal, positions, raw_k, held_k, target

    def _materialized(self, belief, status: dict):
        """The belief as of THIS read, not as of the last correction. The live belief
        only advances when the player acts; a read seconds later must not reuse that
        confidence as if no time had passed. So advance a COPY over the elapsed ticks
        (predict is pure; nothing feeds back) — the same rule the belief log's drift
        lines use. Replay drivers pass no live tick, so they are untouched."""
        tick_now = status.get("tick")
        snapshot = status.get("belief_snapshot_id")
        if tick_now is None or snapshot is None or tick_now <= snapshot:
            return belief
        return predict(belief, (tick_now - snapshot) / 20.0, self.params)

    def _authority(self) -> str:
        # Every trace row declares which authority wrote it; the auditor judges the
        # row by that declaration (demo commits nothing, consent commits only on
        # route "consent", place runs the full §9 rules).
        return "place" if self.place else ("consent" if self.consent_place else "demo")

    def _degraded(self, reason: str, status: dict) -> dict:
        """A read that could not gate still leaves its trace row (one row per read,
        D5 §7) — the log must show the gate was alive and observing, not absent."""
        row = {"tick": status.get("tick"), "k": self.reads,
               "belief_snapshot_id": status.get("belief_snapshot_id"),
               "K_commit": 0, "idle_state": None,
               "chosen_state": "observe", "candidate_state": "observe",
               "reason": reason,
               "authority": self._authority(),
               "committed_actions": []}
        self.trace.write(json.dumps(row) + "\n")
        self.trace.flush()
        return {"state": "observe", "reason": reason, "k_commit": 0,
                "authority": self._authority()}

    def read(self, belief, fused, status: dict) -> dict:
        """One gate read. Returns the status-file "gate" block; flushes a trace row."""
        self.reads += 1
        if fused is None:
            return self._degraded("no corrections yet", status)
        if not decoder_model.available():
            return self._degraded("no decoder on disk", status)
        gate_read, proposal, positions, raw_k, held_k, target = self._decide(
            self._materialized(belief, status), fused, status)
        decision, candidate = self.fsm.read(gate_read)
        summary = proposal_summary(proposal, target, held_k, gate_read.top_goal)
        first_block = (materials.normalize(proposal.actions[0].block)
                       if (proposal is not None and target is not None
                           and isinstance(proposal.actions[0], Place))
                       else None)
        # The placement directive (§9 amendment 2026-07-12): at most ONE block per
        # read — the first committed action, only when it is a Place, only at a cell
        # the agent has not already filled. The body executes; this only authorizes.
        directive = None
        committed = []
        why_no_place = None
        if decision.state.value == "place_low_risk" and proposal is not None and held_k >= 1:
            first = proposal.actions[0]
            if not isinstance(first, Place):
                why_no_place = "first committed action is not a placement (v1 executes Place only)"
            elif tuple(target) in self.agent_cells:
                why_no_place = "target already filled by the agent; waiting for new human evidence"
            else:
                conf_now = gate_read.p_top * (1 - gate_read.p_z1)
                route = ("confidence" if conf_now >= self.fsm.config.theta_place
                         else "declared")
                block_name = materials.normalize(first.block)
                self.agent_cells.add(tuple(target))
                self.directives += 1
                directive = {"id": f"{self._run_token}-{self.reads}",
                             "cell": list(target),
                             "block": block_name, "route": route}
                committed = [{"action": {"type": "place", "cell": list(target),
                                         "block": block_name},
                              "actor": "agent", "reversible": True, "route": route}]
        # The consent route (D5 §10, 2026-07-19): the human's relayed "yes" licenses
        # ONE placement of the OFFER they consented to — {cell, block} as voiced —
        # whatever the confidence and whatever the decoder proposes NOW (it
        # re-proposes every read and may have wandered since the offer). YIELD
        # still emits nothing, a consumed consent never fires twice, and the
        # veto's reason is written into the row so an unhonored yes is
        # diagnosable afterward.
        if (directive is None and self.consent_place
                and self._agent_consent is not None
                and decision.state.value != "yield"):
            veto = consent_veto(self._agent_consent, self._consent_consumed,
                                self.agent_cells, gate_read.player_pos,
                                self._agent_inventory, time.time())
            if veto is None:
                consented_cell = tuple(self._agent_consent["cell"])
                block_name = materials.normalize(self._agent_consent["block"])
                self._consent_consumed.add(self._agent_consent["ts"])
                self.agent_cells.add(consented_cell)
                self.directives += 1
                directive = {"id": f"{self._run_token}-{self.reads}",
                             "cell": list(consented_cell),
                             "block": block_name, "route": "consent"}
                committed = [{"action": {"type": "place",
                                         "cell": list(consented_cell),
                                         "block": block_name},
                              "actor": "agent", "reversible": True,
                              "route": "consent"}]
            elif veto != "consent already honored (one block per yes)":
                why_no_place = f"consent: {veto}"
        row = {"tick": fused.tick, "k": self.reads,
               "belief_snapshot_id": status.get("belief_snapshot_id"),
               "inputs_snapshot": {
                   "top_goal": decision.inputs_snapshot.top_goal,
                   "p_top_goal": round(decision.inputs_snapshot.p_top_goal, 4),
                   "belief_entropy": round(decision.inputs_snapshot.belief_entropy, 4),
                   "p_z1": round(decision.inputs_snapshot.p_z1, 4),
                   "proximity": (None if decision.inputs_snapshot.proximity is None
                                 else round(decision.inputs_snapshot.proximity, 2)),
                   "reversibility": decision.inputs_snapshot.reversibility},
               "K_commit": held_k, "raw_k_commit": raw_k, "per_position": positions,
               "idle_state": "idle" if gate_read.idle else "active",
               "chosen_state": decision.state.value,
               "candidate_state": candidate.value, "reason": decision.reason,
               "decode_ms": (round(self.last_decode_ms, 1)
                             if self.last_decode_ms is not None else None),
               "feasible_prefix": (self._materials_block or {}).get("feasible_prefix"),
               "missing": (self._materials_block or {}).get("missing"),
               "gather": self._gather_block,   # schema-additive (D5 §4 amendment)
               # The decoder's first proposed placement, recorded on EVERY read that
               # has one — voiced or not. This is what makes the acceptance signal's
               # CONTROL group computable (D9 §3): reads where a proposal existed
               # but was never surfaced give the base rate at which the human does
               # the proposed thing anyway. Schema-additive.
               "proposal_first": ({"block": first_block, "cell": list(target)}
                                  if first_block is not None else None),
               # The auditor judges each trace by its own recorded authority: demo
               # traces must show zero placements, place traces at most 1 per read.
               "authority": self._authority(),
               "committed_actions": committed}
        if why_no_place:
            row["reason"] += f" | no directive: {why_no_place}"
        self.trace.write(json.dumps(row) + "\n")
        self.trace.flush()
        return {"state": decision.state.value, "reason": decision.reason,
                "k_commit": held_k,
                "conf": round(gate_read.p_top * (1 - gate_read.p_z1), 4),
                "p_star": round(gate_read.p_top, 4),
                # The body speaks up about shortages only when placement is armed —
                # asking for materials the config would never let it place is noise.
                "authority": self._authority(),
                "target_cell": list(target) if target else None,
                # The proposal's block name rides with the cell so the body's
                # consent can name the full OFFER {cell, block} — the mind honors
                # exactly that pair even if its proposal moves before the yes.
                "target_block": first_block,
                "proposal_summary": summary,
                "materials": self._materials_block,
                # The agent executes a gather errand ONLY when state == "gather";
                # the block rides along on other reads so FlowViz can show the plan.
                "gather": self._gather_block,
                # The body places ONLY while this block is present (and its id is
                # new); every other read carries no directive at all.
                "place": directive}

    def close(self) -> None:
        self.trace.close()
        if self.account.last_inventory is not None:
            # The constraint was live at least once: leave the session's material
            # ledger — usage, remaining, missing, compromises (D5 §4 report).
            if ".gate_trace" in self.trace_path:
                base = self.trace_path[: self.trace_path.index(".gate_trace")]
            else:
                base = os.path.splitext(self.trace_path)[0]
            report_path = base + ".materials_report.json"
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(self.account.report(committed_places=self.directives),
                          handle, indent=2)
            print(f"  materials report -> {os.path.basename(report_path)}")


class AsyncGateRunner:
    """The live gate OFF the ingest thread (the proof-grade fix, 2026-07-13).

    The measured problem: one gate read costs 200–360 ms mean on this machine
    (decoder forward + staircase), and it ran INSIDE the socket-draining loop —
    six-plus ticks of ingest blocked every second, which is exactly where the
    gap ticks that spoil proof-grade sessions came from (every recent session:
    gate mean 205–361 ms, gaps 13–184, stale ≈ gaps).

    This wrapper runs the SAME runner on one worker thread. The ingest loop
    hands over the freshest (belief, fused, status) and keeps draining; the
    worker computes at its own pace; the newest finished gate block is picked
    up at the next status write. A request that arrives while a read is in
    flight REPLACES the waiting one — the gate wants the freshest moment,
    never a queue of stale ones (so trace rows may be sparser than status
    ticks when reads run long; each row is still one real read, D5 §7).

    Block events buffer through a deque and are drained by the worker right
    before each read, so the runner's cell sets have exactly ONE mutating
    thread — handing the sets to two threads was a set-changed-during-
    iteration crash waiting for a busy session.
    """

    def __init__(self, runner: LiveGateRunner, on_timing=None):
        self._runner = runner
        self._on_timing = on_timing            # e.g. a StallMeter's .add
        self._events: collections.deque = collections.deque()
        self._lock = threading.Lock()
        self._request = None
        self._wake = threading.Event()
        self._block: dict | None = None
        self._closing = False
        self._thread = threading.Thread(target=self._loop, name="gate-read",
                                        daemon=True)
        self._thread.start()

    @property
    def place(self) -> bool:
        return self._runner.place

    def ingest_events(self, block_events) -> None:
        """Ingest-thread side: buffer only. deque.append is atomic under the GIL."""
        for event in block_events:
            self._events.append(event)

    def request(self, belief, fused, status: dict) -> None:
        """Hand the worker the freshest inputs; never blocks."""
        with self._lock:
            self._request = (belief, fused, status)
        self._wake.set()

    def latest(self) -> dict | None:
        """The newest COMPLETED gate block (None until the first read lands)."""
        with self._lock:
            return self._block

    def _loop(self) -> None:
        while True:
            # The bounded wait makes lost wakeups impossible to hang on: worst
            # case the worker notices a request (or the close) half a second late.
            self._wake.wait(timeout=0.5)
            with self._lock:
                request, self._request = self._request, None
                self._wake.clear()
                closing = self._closing
            if request is None:
                if closing:
                    return
                continue
            drained = []
            while self._events:
                drained.append(self._events.popleft())
            if drained:
                self._runner.ingest_events(drained)
            started = time.perf_counter()
            try:
                block = self._runner.read(*request)
            except Exception as error:          # a gate crash must not kill the worker
                block = {"state": "observe", "reason": f"gate error: {error}",
                         "k_commit": 0,
                         "authority": "place" if self._runner.place else "demo"}
            if self._on_timing is not None:
                self._on_timing((time.perf_counter() - started) * 1000.0)
            with self._lock:
                self._block = block

    def close(self) -> None:
        """Finish the read in flight, stop the worker, close the runner."""
        with self._lock:
            self._closing = True
        self._wake.set()
        self._thread.join(timeout=30.0)
        self._runner.close()


def replay_gate(jsonl_path: str) -> int:
    """The pre-demo proof: run the LIVE runner over a recorded session's banked
    evidence, one read per correction, and show the state mix. Writes
    <session>.gate_trace.replay.jsonl — never the live trace's name."""
    from ..capture.jsonl_ingest import JsonlSource
    from ..contracts.b3 import fuse_streams
    from ..data import decoder_corpus
    from ..intent import heads_v1

    base = jsonl_path[:-6]                     # strip .jsonl
    b1 = [json.loads(l) for l in open(base + ".evidence2d.jsonl", encoding="utf-8")
          if l.strip()]
    b2 = [json.loads(l) for l in open(base + ".evidence3d.jsonl", encoding="utf-8")
          if l.strip()]
    fused = fuse_streams(b1, b2, os.path.basename(base))
    beliefs = decoder_corpus.replay_beliefs(fused)     # the calibrated v1 replay
    packets = JsonlSource(jsonl_path, base + ".manifest.json").load().packets
    positions = {p.tick: (p.server.player_pos.x, p.server.player_pos.y,
                          p.server.player_pos.z)
                 for p in packets if p.server.player_pos is not None}
    events_by_tick: dict[int, list] = {}
    for packet in packets:
        if packet.server.block_events:
            events_by_tick[packet.tick] = list(packet.server.block_events)

    runner = LiveGateRunner(base + ".gate_trace.replay.jsonl",
                            heads_v1.tracker_params(), demo=True)
    fed_through = 0
    states: dict[str, int] = {}
    last_pos = None
    ticks_sorted = sorted(events_by_tick)
    for record, belief in zip(fused, beliefs):
        while fed_through < len(ticks_sorted) and ticks_sorted[fed_through] <= record.tick:
            runner.ingest_events(events_by_tick[ticks_sorted[fed_through]])
            fed_through += 1
        for at in sorted(positions):
            if at <= record.tick:
                last_pos = positions[at]
        status = {"current_behavior": "idle" if record.idle else record.a_hat.value,
                  "player_pos": last_pos,
                  "focus_block": ((record.focus.block.x, record.focus.block.y,
                                   record.focus.block.z) if record.focus.block else None),
                  "belief_snapshot_id": record.tick}
        block = runner.read(belief, record, status)
        states[block["state"]] = states.get(block["state"], 0) + 1
    runner.close()
    total = sum(states.values())
    print(f"gate replay over {total} reads (demo config, v1 belief):")
    for state, count in sorted(states.items(), key=lambda kv: -kv[1]):
        print(f"  {state:<15} {count:>5}  ({count / total:.0%})")
    if states.get("place_low_risk"):
        print("  !! PLACE_LOW_RISK appeared under the demo config — this is a bug")
        return 1
    print(f"  trace -> {os.path.basename(base)}.gate_trace.replay.jsonl")
    return 0
