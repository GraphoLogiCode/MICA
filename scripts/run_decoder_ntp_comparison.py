"""The pre-registered real-session NTP comparison — one command, regenerable.

    python scripts/run_decoder_ntp_comparison.py [--skip-train]

Question (vault note "2026-07-19 - Decoder Real-Session NTP Pretraining"): the
decoder trains on scripted builds yet proposes live on real free-builds — does
pretraining its next-token stage on REAL sessions' action streams (goal-free, so
contested and discarded sessions carry zero label risk) close that domain gap?

Chain:
  1. requires the corpus's real groups (make_decoder_corpus.py --real) and the
     production decoder the cascade just trained (the CONTROL — no second
     scripted-only training is needed).
  2. train the PRETRAINED arm -> models/decoder_cmp_real.* (stage A on
     scripted + real_pretrain rows; stage B scripted-only; no pin logic).
     Production decoder_v1.* untouched (checksum-asserted).
  3. eval BOTH final models with the same fixed draw on the two holdouts:
     the pre-registered four-session REAL holdout (primary) and the scripted
     holdout (guard). Print + bank capture/raw/decoder_ntp_comparison.json.

Decision rule (pre-registered, printed with the result): the pretraining is
adopt-worthy ONLY if real-holdout one-step NLL STRICTLY improves AND scripted-
holdout one-step NLL stays within 2% of control's. A win changes nothing by
itself: adoption is two-step — the user approves flipping the recipe (dated
amendment), and the NEXT cascade's full OQ1 adjudication must pass on the
pretrained recipe before anything ships live.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from mica.data import decoder_corpus                                  # noqa: E402
from mica.decoder import model as decoder_model                       # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
_OUT = os.path.join(_ROOT, "capture", "raw", "decoder_ntp_comparison.json")

SCRIPTED_NLL_TOLERANCE = 0.02      # relative: within 2% of control (the OQ1 bar)
CMP_PREFIX = "decoder_cmp_real"


def _sha(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _load_arm(weights_path: str, meta_path: str, device):
    import torch

    model, _, meta = decoder_model.load(weights_path, meta_path)
    model.to(device)
    model.eval()
    return model, meta


def _nll(model, rows, device) -> float:
    from train_decoder import _BATCH, _SEED, _holdout_nll, _prepare

    return round(_holdout_nll(model, _prepare(rows), device, _SEED), 4)


def main() -> int:
    import torch

    real_holdout = decoder_corpus.load_group("real_holdout")
    if not real_holdout:
        print("no real_holdout rows — run make_decoder_corpus.py --real first")
        return 1
    if not decoder_model.available():
        print("no production decoder on disk — the cascade trains it first")
        return 1
    guarded = {name: _sha(os.path.join(_ROOT, "models", name))
               for name in ("decoder_v1.pt", "decoder_v1.json")}

    if "--skip-train" not in sys.argv:
        print("=== training the PRETRAINED arm (stage A += real_pretrain rows) ===")
        code = subprocess.run([sys.executable,
                               os.path.join(_SCRIPTS, "train_decoder.py"),
                               "--real-pretrain", "--out-prefix", CMP_PREFIX]
                              ).returncode
        if code != 0:
            raise SystemExit(f"pretrained-arm training exited {code}")

    for name, digest in guarded.items():
        assert _sha(os.path.join(_ROOT, "models", name)) == digest, \
            f"PRODUCTION MODEL {name} CHANGED — comparison is invalid, investigate"
    print("production decoder_v1.* checksums unchanged — the arm trained to its own files")

    _, scripted_eval, labels = decoder_corpus.load_rows()
    scripted_holdout = [row for row in scripted_eval
                        if labels[row["session"]]["group"] == "holdout"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    control, _ = _load_arm(decoder_model.WEIGHTS_PATH, decoder_model.META_PATH, device)
    pretrained, _ = _load_arm(os.path.join(_ROOT, "models", f"{CMP_PREFIX}.pt"),
                              os.path.join(_ROOT, "models", f"{CMP_PREFIX}.json"),
                              device)

    results = {}
    for arm_name, model in (("control", control), ("pretrained", pretrained)):
        results[arm_name] = {
            "real_holdout_token_nll": _nll(model, real_holdout, device),
            "scripted_holdout_token_nll": _nll(model, scripted_holdout, device),
        }
    ctrl, pre = results["control"], results["pretrained"]
    real_wins = pre["real_holdout_token_nll"] < ctrl["real_holdout_token_nll"]
    scripted_ok = (pre["scripted_holdout_token_nll"]
                   <= ctrl["scripted_holdout_token_nll"] * (1 + SCRIPTED_NLL_TOLERANCE))
    verdict = real_wins and scripted_ok

    report = {
        "arms": results,
        "eval": {"real_holdout_rows": len(real_holdout),
                 "real_holdout_sessions": sorted({r["session"] for r in real_holdout}),
                 "scripted_holdout_rows": len(scripted_holdout)},
        "decision_rule": {"registered": "2026-07-19",
                          "real_nll_strictly_below_control": True,
                          "scripted_nll_relative_tolerance": SCRIPTED_NLL_TOLERANCE},
        "verdict": ("REAL-NTP PRETRAINING WINS — adoption is the user's two-step "
                    "call (dated amendment + next cascade's OQ1 on the new recipe)"
                    if verdict else "scripted-only recipe STANDS"),
    }
    with open(_OUT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)

    print(f"\n{'metric':<28}  {'control':>10}  {'pretrained':>10}")
    for key in ("real_holdout_token_nll", "scripted_holdout_token_nll"):
        print(f"{key:<28}  {ctrl[key]:>10}  {pre[key]:>10}")
    print(f"\nPRE-REGISTERED VERDICT: {report['verdict']}")
    print(f"report -> {os.path.relpath(_OUT, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
