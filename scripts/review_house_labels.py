"""The Source C review gate: approve or reject every mined label in the browser.

    python scripts/review_house_labels.py [--port 8330]

Serves the sheet make_review_sheet.py built. The labels live on TWO levels, kept
strictly apart (the same line the D2 design draws: a wall or roof is a stage
inside a build, never a build category):

  CLIP VERDICT (one per clip — this is the gate):
      HOUSE      this is a house-building clip -> kept in the approved subset
      NOT_HOUSE  it is not                     -> rejected
      UNSURE     cannot tell                   -> excluded, marked for revisit

  FRAME STAGE TAGS (optional, one per frame — annotation, never the gate):
      EXTERIOR / WALL_BUILD / ROOF_BUILD / INTERIOR / OTHER — what this frame
      shows. A house tutorial naturally passes through walls, roof, and interior;
      tagging frames records that WITHOUT competing with the clip verdict.

Every decision saves immediately to capture/youtube/review_decisions.json, and
capture/youtube/house_subset_approved.json is rebuilt on the spot — clips whose
verdict is HOUSE, carrying the human verdict plus the frame stage tags. THAT file
is the only Source C artifact anything downstream may read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "capture", "youtube")
_REVIEW = os.path.realpath(os.path.join(_OUT, "review"))
_ITEMS = os.path.join(_OUT, "review_items.json")
_DECISIONS = os.path.join(_OUT, "review_decisions.json")
_APPROVED = os.path.join(_OUT, "house_subset_approved.json")

_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>MICA - Source C label review</title>
<style>
 body { background:#12151c; color:#d7dce6; font:14px/1.5 system-ui, sans-serif;
        margin:0; padding:24px; }
 h1 { font-size:17px; margin:0 0 4px; }
 .scope { color:#8a93a6; font-size:12px; max-width:900px; }
 #progress { margin:10px 0 18px; color:#9fb7d8; }
 .card { background:#1a1f2b; border:1px solid #2a3145; border-radius:10px;
         padding:14px 16px; margin-bottom:16px; max-width:1080px; }
 .card.decided-keep { border-color:#2f6f4f; }
 .card.decided-reject { border-color:#7a3a3a; opacity:.75; }
 .card.decided-unsure { border-color:#8a7a3a; }
 .title { font-weight:600; } .title a { color:#7fb3ff; text-decoration:none; }
 .meta { color:#8a93a6; font-size:12px; margin:2px 0 8px; }
 .frames { display:flex; gap:10px; flex-wrap:wrap; margin:8px 0; }
 .frame { width:250px; }
 .frame img { width:100%; border-radius:6px; display:block; }
 .frame .t { color:#8a93a6; font-size:11px; margin-top:2px; }
 .frame .win { color:#c9d4e6; font-size:11px; font-style:italic; }
 .stagerow { display:flex; gap:3px; flex-wrap:wrap; margin-top:4px; }
 .stage { border:1px solid #3c4a6b; background:transparent; color:#9fb7d8;
          border-radius:4px; padding:1px 6px; font-size:10.5px; cursor:pointer; }
 .stage.picked { background:#3c4a6b; color:#fff; font-weight:600; }
 .evidence { background:#141824; border-left:3px solid #3c4a6b; padding:6px 10px;
             margin:6px 0; font-size:12.5px; }
 .evidence b { color:#9fb7d8; font-weight:600; }
 .rule { color:#b9a15e; font-size:12px; margin:6px 0; }
 .verdict { margin-top:12px; padding-top:10px; border-top:1px solid #2a3145;
            display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
 .verdict .q { color:#9fb7d8; font-weight:600; margin-right:4px; }
 .verdict button { border:0; border-radius:6px; padding:7px 16px; font-weight:700;
                   cursor:pointer; color:#0d1017; }
 .b-house { background:#5fd08a; } .b-not { background:#e07a7a; }
 .b-unsure { background:#8a93a6; }
 .verdict button.picked { outline:3px solid #ffffff66; }
 .decision-note { color:#8a93a6; font-size:12px; margin-left:6px; }
 .migrated { color:#b9a15e; font-size:11.5px; margin-left:6px; }
 .autobadge { background:#3c4a6b; color:#cdd9f0; border-radius:4px; padding:1px 7px;
              font-size:11px; font-weight:700; margin-left:6px; }
 .vlmhint { color:#8fb0d8; font-size:11.5px; margin-left:6px; font-style:italic; }
 .auditnote { color:#d8a15e; font-size:11.5px; margin-left:6px; font-weight:600; }
</style></head><body>
<h1>Source C — manual label review</h1>
<div class="scope" id="scope"></div>
<div id="progress"></div>
<div id="cards"></div>
<script>
const STAGES = ["EXTERIOR","WALL_BUILD","ROOF_BUILD","INTERIOR","INVENTORY_UI","OTHER"];
// The concrete definitions (2026-07-04, user decision) — same text the model gets.
const STAGE_DEFS = {
  EXTERIOR: "camera OUTSIDE the build, looking at it - the structure visible from outside",
  WALL_BUILD: "working on the SIDE of the build - a wall face up close, blocks placed against it",
  ROOF_BUILD: "building on TOP - roof or top-layer work, camera at/above the upper edge",
  INTERIOR: "camera INSIDE the build - its walls/floor/ceiling surround the view",
  INVENTORY_UI: "a menu covers the view (inventory, crafting, chest) - this wins when present",
  OTHER: "ONLY when nothing above clearly fits - terrain, sky, title cards, no build in sight",
};
let DATA = null;
async function load() {
  DATA = await (await fetch('/items')).json();
  document.getElementById('scope').textContent = DATA.scope;
  render();
}
function render() {
  const decided = Object.values(DATA.decisions).filter(d => d.label).length;
  document.getElementById('progress').textContent =
    decided + " / " + DATA.items.length + " clip verdicts - approved subset = clips whose " +
    "verdict matches their category label (frame tags ride along as annotations)";
  const root = document.getElementById('cards');
  root.innerHTML = "";
  for (const item of DATA.items) {
    const decision = DATA.decisions[item.id] || {};
    const label = item.proposed_label || "HOUSE";
    const card = document.createElement('div');
    card.className = "card" + (decision.label === label ? " decided-keep" :
      (decision.label || "").startsWith("NOT_") ? " decided-reject" :
      decision.label === "UNSURE" ? " decided-unsure" : "");
    let frames = "";
    for (const f of item.frames) {
      const tagged = (decision.frames || {})[f.file];
      let chips = "";
      for (const stage of STAGES) {
        chips += `<button class="stage${tagged === stage ? " picked" : ""}"
          title="${STAGE_DEFS[stage]}"
          onclick="tag('${item.id}','${f.file}','${stage}')">${stage}</button>`;
      }
      frames += `<div class="frame"><img src="/frame?f=${encodeURIComponent(f.file)}">
        <div class="t">@ ${f.t}s${f.window ? "" : " (evenly spaced - no caption here)"}</div>
        ${f.window ? `<div class="win">&ldquo;${f.window}&rdquo;</div>` : ""}
        <div class="stagerow">${chips}</div></div>`;
    }
    let evidence = item.windows.length
      ? item.windows.map(w => `<div class="evidence"><b>@${w.t}s</b> &ldquo;${w.text}&rdquo;</div>`).join("")
      : `<div class="evidence">no transcript available - judge from the frames + title</div>`;
    let verdictButtons = "";
    for (const [verdict, cls] of [[label,"b-house"],["NOT_"+label,"b-not"],["UNSURE","b-unsure"]]) {
      const picked = decision.label === verdict ? " picked" : "";
      verdictButtons += `<button class="${cls}${picked}" onclick="decide('${item.id}','${verdict}')">${verdict}</button>`;
    }
    card.innerHTML = `
      <div class="title"><a href="${item.link}" target="_blank">${item.title}</a></div>
      <div class="meta">category: <b>${item.category || "habitation"}</b> ·
        proposed_label: <b>${item.proposed_label}</b> ·
        confidence ${item.confidence} · ${Math.round(item.duration)}s ·
        tag frames below (what does each frame SHOW), then give the clip verdict</div>
      <div class="frames">${frames}</div>
      ${evidence}
      <div class="rule">rule: ${item.rule}</div>
      <div class="verdict"><span class="q">Is this a ${label}-building clip?</span>
        ${verdictButtons}
        <span class="decision-note">${decision.label ? "verdict: " + decision.label : "pending"}</span>
        ${decision.decided_by === "auto" ? `<span class="autobadge">AUTO</span>
          <span class="vlmhint">${(decision.vlm || {}).reason || ""} - click to override</span>` : ""}
        ${decision.recheck ? `<button class="b-house" style="margin-left:6px"
            onclick="decide('${item.id}','${decision.label}')">CONFIRM ${decision.label} &#10003;</button>
          <span class="auditnote">RECHECK: ${decision.recheck}</span>` : ""}
        ${decision.audit_of_auto && !decision.label ? `<span class="auditnote">AUDIT: the auto
          tier would say ${decision.audit_of_auto} - your verdict grades it</span>` : ""}
        ${decision.vlm && !decision.label && !decision.audit_of_auto ?
          `<span class="vlmhint">VLM opinion: ${decision.vlm.verdict}
           - ${decision.vlm.reason || ""}</span>` : ""}
        ${decision.migrated_stage ? `<span class="migrated">migrated from pre-split review
          (you had picked ${decision.migrated_stage} - now a frame tag, refine below)</span>` : ""}
      </div>`;
    root.appendChild(card);
  }
}
async function decide(id, label) {
  await fetch('/decide', { method:'POST', headers:{'Content-Type':'application/json'},
                           body: JSON.stringify({ id, label }) });
  DATA.decisions[id] = { ...(DATA.decisions[id] || {}), label, decided_by: "human" };
  delete DATA.decisions[id].migrated_stage;
  delete DATA.decisions[id].recheck;
  render();
}
async function tag(id, file, stage) {
  const current = ((DATA.decisions[id] || {}).frames || {})[file];
  const next = current === stage ? null : stage;    // click again to clear
  await fetch('/tag', { method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({ id, file, stage: next }) });
  DATA.decisions[id] = DATA.decisions[id] || {};
  DATA.decisions[id].frames = DATA.decisions[id].frames || {};
  if (next) DATA.decisions[id].frames[file] = next;
  else delete DATA.decisions[id].frames[file];
  render();
}
load();
</script></body></html>"""


def _load_items() -> dict:
    with open(_ITEMS, encoding="utf-8") as handle:
        return json.load(handle)


def _load_decisions() -> dict:
    if os.path.exists(_DECISIONS):
        with open(_DECISIONS, encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def _save_decisions(decisions: dict) -> None:
    with open(_DECISIONS, "w", encoding="utf-8") as handle:
        json.dump(decisions, handle, indent=2)
    _write_approved(_load_items(), decisions)


def _tag_provenance(decision: dict) -> dict:
    """frame_stages with per-tag provenance: a tag is provably HUMAN when its clip
    has no VLM record at all, or when it differs from what the model proposed —
    the same heuristic --restage protects with. Downstream uses that depend on
    stages (the model only matches human stage tags ~54%) filter to by=human."""
    vlm_stages = decision.get("vlm", {}).get("stages", {})
    has_vlm = "vlm" in decision
    return {file: {"stage": stage,
                   "by": "human" if not has_vlm or stage != vlm_stages.get(file) else "auto"}
            for file, stage in decision.get("frames", {}).items()}


def _write_approved(items: dict, decisions: dict) -> None:
    """The gate's output: HOUSE-verdict clips with their frame stage tags. Each entry
    says WHO decided it (human, or the two-signal auto tier) so any downstream use
    that needs human-only labels can filter without re-reviewing."""
    approved = []
    for item in items["items"]:
        decision = decisions.get(item["id"], {})
        # Approved = the verdict matches the clip's own proposed category label
        # (HOUSE for habitation, PRODUCTION for production, ...). The category
        # field is the bridge into the goal taxonomy (contracts/goals.py); frame
        # stages never map to categories — stages are not goals.
        label = item.get("proposed_label", "HOUSE")
        if decision.get("label") == label:
            approved.append({**{k: item[k] for k in
                                ("id", "title", "link", "duration", "rule", "confidence")},
                             "human_label": label,
                             "category": item.get("category", "habitation"),
                             "decided_by": decision.get("decided_by", "human"),
                             "frame_stages": _tag_provenance(decision)})
    decided = [d for d in decisions.values() if d.get("label")]
    with open(_APPROVED, "w", encoding="utf-8") as handle:
        json.dump({"scope": items["scope"],
                   "label_scheme": {"clip_verdicts": "<LABEL> | NOT_<LABEL> | UNSURE, per category",
                                    "clip_verdict_category_map": {
                                        "HOUSE": "habitation", "PRODUCTION": "production",
                                        "INFRASTRUCTURE": "infrastructure",
                                        "DEFENSE": "defense", "DECORATIVE": "decorative"},
                                    "frame_stages": ["EXTERIOR", "WALL_BUILD", "ROOF_BUILD",
                                                     "INTERIOR", "INVENTORY_UI", "OTHER"],
                                    "frame_stage_provenance": "each tag carries by: human|auto;"
                                        " stage-dependent uses filter to human (auto stages"
                                        " match human ~54%)"},
                   "reviewed": len(decided),
                   "by_human": sum(1 for d in decided if d.get("decided_by", "human") == "human"),
                   "by_auto": sum(1 for d in decided if d.get("decided_by") == "auto"),
                   "approved": approved}, handle, indent=2)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            self._send(_PAGE.encode(), "text/html; charset=utf-8")
        elif self.path.startswith("/items"):
            payload = _load_items()
            payload["decisions"] = _load_decisions()
            self._send(json.dumps(payload).encode(), "application/json")
        elif self.path.startswith("/frame?f="):
            from urllib.parse import unquote
            relative = unquote(self.path.split("=", 1)[1])
            path = os.path.realpath(os.path.join(_REVIEW, relative))
            if not path.startswith(_REVIEW) or not os.path.exists(path):
                return self.send_error(404)
            with open(path, "rb") as handle:
                self._send(handle.read(), "image/jpeg")
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        decisions = _load_decisions()
        entry = decisions.setdefault(body["id"], {})
        if self.path.startswith("/decide"):
            if entry.get("decided_by") == "auto" and entry.get("label") != body["label"]:
                entry["overridden_auto"] = entry["label"]   # grades the auto tier
            entry["label"] = body["label"]
            entry["decided_by"] = "human"
            entry["ts"] = int(time.time() * 1000)
            entry.pop("migrated_stage", None)      # a fresh verdict clears the note
            entry.pop("recheck", None)             # ... and settles a recheck
        elif self.path.startswith("/tag"):
            frames = entry.setdefault("frames", {})
            if body.get("stage"):
                frames[body["file"]] = body["stage"]
            else:
                frames.pop(body["file"], None)
        else:
            return self.send_error(404)
        _save_decisions(decisions)
        self._send(b'{"ok": true}', "application/json")


def main() -> int:
    if not os.path.exists(_ITEMS):
        print("no review sheet - run scripts/make_review_sheet.py first")
        return 1
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8330
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"review the labels at  http://localhost:{port}")
    print(f"decisions -> {os.path.relpath(_DECISIONS, _ROOT)}"
          f"  approved subset -> {os.path.relpath(_APPROVED, _ROOT)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
