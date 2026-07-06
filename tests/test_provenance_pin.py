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


def test_banked_corpora_are_pinned_with_the_load_bearing_fields():
    for module, directory in ((make_scripted_corpus, os.path.join(_ROOT, "capture", "scripted")),
                              (make_decoder_corpus, os.path.join(_ROOT, "capture", "decoder"))):
        if not os.path.isdir(directory):
            continue
        pin = module.provenance_pin(directory)
        assert pin is not None, f"{directory} lost its provenance pin"
        for field in ("generator", "verified_mismatch",
                      "consumers_trained_on_this_bank", "regeneration"):
            assert pin.get(field), f"pin in {directory} missing {field!r}"
        assert "STATIC" in pin["generator"]


def test_generators_refuse_while_the_pin_stands(monkeypatch):
    # main() must exit 1 before generating anything when the pin exists and
    # --unpin was not passed
    monkeypatch.setattr(sys, "argv", ["make_scripted_corpus.py"])
    assert make_scripted_corpus.main() == 1
    monkeypatch.setattr(sys, "argv", ["make_decoder_corpus.py"])
    assert make_decoder_corpus.main() == 1
