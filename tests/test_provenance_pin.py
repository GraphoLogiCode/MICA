"""The corpus provenance pins: while a bank predates the motion-default generator,
the no-flag regeneration commands must refuse rather than silently orphan every
model trained on that bank (decision B, 2026-07-06)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import make_decoder_corpus  # noqa: E402
import make_scripted_corpus  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pin_reader_returns_none_without_a_pin(tmp_path):
    assert make_scripted_corpus.provenance_pin(str(tmp_path)) is None
    assert make_decoder_corpus.provenance_pin(str(tmp_path)) is None


def test_a_pin_when_present_carries_the_load_bearing_fields():
    # Cascade A (2026-07-06) legitimately removes the pins — banks then match the
    # current generator again. The invariant that survives: a pin that EXISTS must
    # be well-formed, or the refusal message would be garbage exactly when needed.
    for module, directory in ((make_scripted_corpus, os.path.join(_ROOT, "capture", "scripted")),
                              (make_decoder_corpus, os.path.join(_ROOT, "capture", "decoder"))):
        if not os.path.isdir(directory):
            continue
        pin = module.provenance_pin(directory)
        if pin is None:
            continue
        for field in ("generator", "verified_mismatch",
                      "consumers_trained_on_this_bank", "regeneration"):
            assert pin.get(field), f"pin in {directory} missing {field!r}"


def test_generators_refuse_while_a_pin_stands(monkeypatch, tmp_path):
    # main() must exit 1 before generating anything when a pin exists and --unpin
    # was not passed. Exercised against a synthetic pin so the test outlives the
    # banked pins' own lifecycle.
    pin_body = {"generator": "STATIC (test)", "verified_mismatch": "test",
                "consumers_trained_on_this_bank": ["test"], "regeneration": "test"}
    for module in (make_scripted_corpus, make_decoder_corpus):
        pinned_dir = tmp_path / module.__name__
        pinned_dir.mkdir()
        (pinned_dir / "PROVENANCE_PIN.json").write_text(
            __import__("json").dumps(pin_body), encoding="utf-8")
        monkeypatch.setattr(module, "_OUT", str(pinned_dir))
        monkeypatch.setattr(sys, "argv", [f"{module.__name__}.py"])
        assert module.main() == 1
