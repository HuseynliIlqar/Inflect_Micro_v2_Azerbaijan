"""The device check's reference writer -- its helpers, no model loaded."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "calibration_reference", ROOT / "tools" / "calibration_reference.py"
)
calibration_reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(calibration_reference)

GENERATED = ROOT / "web" / "js" / "engine" / "calibration-reference.js"


def test_tokenise_intersperses_blanks_like_inputs_js() -> None:
    symbols = ["_", "a", "b"]
    assert calibration_reference.tokenise("ab", symbols) == [0, 1, 0, 2, 0]


def test_tokenise_skips_unknown_symbols() -> None:
    assert calibration_reference.tokenise("a?b", ["_", "a", "b"]) == [0, 1, 0, 2, 0]


def test_envelope_is_rms_per_whole_window() -> None:
    wave = [0.5, -0.5, 0.25, -0.25, 1.0]
    assert calibration_reference.envelope(wave, window=2) == [0.5, 0.25]


def test_rms_of_nothing_is_zero() -> None:
    assert calibration_reference.rms([]) == 0.0


def _generated_reference() -> dict:
    source = GENERATED.read_text(encoding="utf-8")
    body = re.search(r"Object\.freeze\((\{.*\})\);", source, re.S)
    assert body, "calibration-reference.js is not in the generated shape"
    return json.loads(body.group(1))


def test_generated_module_matches_the_tool() -> None:
    reference = _generated_reference()
    assert reference["phonemes"] == calibration_reference.PHONEMES
    assert reference["text"] == calibration_reference.TEXT
    assert reference["window"] == calibration_reference.WINDOW


def test_generated_envelopes_cover_their_clips() -> None:
    reference = _generated_reference()
    for precision in ("fp32", "int8"):
        print_ = reference[precision]
        assert len(print_["envelope"]) == print_["samples"] // reference["window"]


def test_render_round_trips() -> None:
    reference = {"version": "0" * 16, "fp32": {"samples": 1, "rms": 0.1, "envelope": []}}
    source = calibration_reference.render(reference)
    body = re.search(r"Object\.freeze\((\{.*\})\);", source, re.S)
    assert json.loads(body.group(1)) == reference
