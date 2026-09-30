"""Which decoder layers stay fp32 in the int8 build -- by name, no model loaded."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "quantize_decoder", Path(__file__).resolve().parent.parent / "tools" / "quantize_decoder.py"
)
quantize_decoder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(quantize_decoder)
kept = quantize_decoder.nodes_kept_in_fp32

VOCODER = (
    ["/decoder/conv_pre/Conv", "/decoder/conv_post/Conv"]
    + [f"/decoder/ups.{s}/ConvTranspose" for s in range(4)]
    + [f"/decoder/resblocks.{b}/convs1.{k}/Conv" for b in range(12) for k in range(3)]
)
FLOW = ["/flow/flows.0/pre/Conv", "/flow/flows.6/post/Conv"]


def test_flow_stays_fp32() -> None:
    assert set(FLOW) <= set(kept(FLOW + VOCODER))


def test_vocoder_edges_stay_fp32() -> None:
    assert {"/decoder/conv_pre/Conv", "/decoder/conv_post/Conv"} <= set(kept(VOCODER))


def test_last_stage_stays_fp32() -> None:
    names = set(kept(VOCODER))
    assert "/decoder/ups.3/ConvTranspose" in names
    assert all(f"/decoder/resblocks.{b}/convs1.0/Conv" in names for b in (9, 10, 11))


def test_earlier_stages_are_quantised() -> None:
    names = set(kept(VOCODER))
    assert not any(f"/decoder/ups.{s}/ConvTranspose" in names for s in (0, 1, 2))
    assert not any(f"/decoder/resblocks.{b}/convs1.0/Conv" in names for b in range(9))


def test_resblock_1_is_not_mistaken_for_10_or_11() -> None:
    # "/decoder/resblocks.1/" must not match the last stage's prefixes.
    assert "/decoder/resblocks.1/convs1.0/Conv" not in kept(VOCODER)


def test_calibration_set_is_not_empty() -> None:
    assert len(quantize_decoder.CALIBRATION) >= 20
