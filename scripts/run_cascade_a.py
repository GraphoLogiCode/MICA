"""Cascade A: the full retrain, one command — fire it when the batch is ready.

    python scripts/run_cascade_a.py            # run the whole chain
    python scripts/run_cascade_a.py --dry-run  # print the steps, change nothing

This is the deliberate, user-invoked retrain that consumes the accumulated batch of
matcher-agreed captures (watch `scripts/after_game.py --status` to decide when it is
worth it). It banks the current models to `models/pre_cascade_a/` for the before/after
comparison, regenerates the motion corpora (removing their provenance pins), retrains
the heads / arm1 / decoder, reruns the four-arm table + decoder eval + counterfactual
gate, and stamps `capture/raw/cascade_status.json` so the readiness baseline resets.

Every step is one of the existing scripts as a subprocess — this wrapper only
sequences them and fails fast if any step exits non-zero, so a broken retrain never
silently ships a half-updated model set.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
from after_game import _agreed_and_pairs, _LEDGER                   # noqa: E402
from mica.capture import session_store                              # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
_MODELS = os.path.join(_ROOT, "models")
_RAW = session_store.RAW_ROOT

# (script, args) in order. Each is idempotent and reproducible on its own.
_CHAIN = [
    ("make_scripted_corpus.py", ["--unpin"]),   # motion corpus (pin removed)
    ("label_finished_builds.py", []),           # relabel -> source_b pairs
    ("train_heads.py", []),                      # adapter + likelihood heads v1
    ("run_tracker.py", ["--real", "--heads", "v1"]),        # belief on real, new heads
    ("calibration_report.py", ["--real", "--heads", "v1", "--held-out"]),
    ("train_arm1.py", []),                       # implicit-classifier arm
    ("run_arms.py", []),                          # four-arm comparison (fresh arm2 queries)
    ("make_decoder_corpus.py", ["--unpin", "--real"]),  # decoder corpus + the
                                                 # real NTP groups (pre-reg 07-19)
    ("train_decoder.py", []),                    # Stage A NTP -> Stage B MTP
    ("run_decoder_eval.py", []),                 # OQ1 + gate freeze
    ("run_gate.py", []),                          # counterfactual gate + traces
    # The contested-pairs comparison rides every cascade (rule v2, re-registered
    # 2026-07-19 in the vault note): it writes the contested pairs itself (the
    # relabel above writes agreed pairs only), trains two extra head sets against
    # the fresh corpus, and prints + banks the pre-registered verdict. It never
    # touches production models -- a "win" is a separate dated D3 amendment for
    # the user to approve.
    ("run_contested_comparison.py", ["--no-vlm"]),
    # The real-session NTP comparison (pre-registered 2026-07-19): one extra
    # decoder training (stage A += goal-free real rows) against the production
    # decoder just trained above; verdict printed and banked. Adoption stays a
    # two-step user decision -- nothing ships from here.
    ("run_decoder_ntp_comparison.py", []),
]

_PRE_MODELS = ["heads_v1.npz", "heads_v1.json", "arm1.npz", "arm1.json",
               "decoder_v1.pt", "decoder_v1.json", "decoder_v1_stage_a.pt", "gate_v1.json"]


def _bank_pre_cascade() -> None:
    dest = os.path.join(_MODELS, "pre_cascade_a")
    os.makedirs(dest, exist_ok=True)
    for name in _PRE_MODELS:
        src = os.path.join(_MODELS, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"banked current models -> models/pre_cascade_a/ ({len(_PRE_MODELS)} files)")


def _stamp_ledger() -> None:
    agreed, pairs = _agreed_and_pairs()
    ledger = {"last_cascade": {"date": time.strftime("%Y-%m-%d"),
                               "pair_pool": pairs, "agreed_sessions": agreed},
              "note": "stamped by run_cascade_a.py"}
    with open(_LEDGER, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=1)
    print(f"readiness baseline reset: {len(agreed)} agreed captures, {pairs} pairs")


def main() -> int:
    dry = "--dry-run" in sys.argv
    print("cascade A chain:")
    for script, args in _CHAIN:
        print(f"  {script} {' '.join(args)}")
    if dry:
        print("\n(dry run — nothing executed, ledger untouched)")
        return 0

    _bank_pre_cascade()
    for script, args in _CHAIN:
        print(f"\n=== {script} {' '.join(args)} ===")
        code = subprocess.run([sys.executable, os.path.join(_SCRIPTS, script), *args]).returncode
        if code != 0:
            print(f"\nSTOP: {script} exited {code} — cascade halted, "
                  "models/pre_cascade_a preserved, ledger NOT stamped")
            return code
    _stamp_ledger()
    print("\ncascade A complete. Record the before/after tables (models/pre_cascade_a "
          "holds the 'before'); commit when ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
