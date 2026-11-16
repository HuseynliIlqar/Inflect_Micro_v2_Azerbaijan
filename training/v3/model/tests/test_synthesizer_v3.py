"""v3's synthesiser must reproduce v2 exactly while its new parts are at their starting values.

Uses the vendored runtime code in model/runtime (read-only, never the weights) with a tiny
configuration, so it runs on the CPU in about a second.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))

from monotonic_align import maximum_path  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402

TINY = {"inter_channels": 8, "hidden_channels": 16, "filter_channels": 32, "n_heads": 2, "n_layers": 2,
        "kernel_size": 3, "p_dropout": 0.1, "resblock": "1", "resblock_kernel_sizes": [3, 7],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], "upsample_rates": [8, 8, 2, 2],
        "upsample_initial_channel": 32, "upsample_kernel_sizes": [16, 16, 4, 4], "use_sdp": False}
N_VOCAB, SPEC, SEGMENT_FRAMES = 20, 33, 8


def _reference_path(value: np.ndarray) -> np.ndarray:
    """VITS' Cython maximum_path, line for line."""
    value = value.copy()
    t_y, t_x = value.shape
    path = np.zeros_like(value)
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = -1e9 if x == y else value[y - 1, x]
            v_prev = (0.0 if y == 0 else -1e9) if x == 0 else value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    index = t_x - 1
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index -= 1
    return path


def test_alignment_matches_the_vits_recurrence() -> None:
    rng = np.random.default_rng(0)
    for t_y, t_x in [(9, 4), (12, 12), (30, 7), (5, 1)]:
        value = rng.normal(size=(t_y, t_x)).astype(np.float32)
        neg_cent = torch.from_numpy(value)[None]
        got = maximum_path(neg_cent, torch.ones(1, t_y, t_x))[0].numpy()
        assert np.array_equal(got, _reference_path(value.astype(np.float64)))
        assert np.all(got.sum(axis=1) == 1)  # one token per frame


def test_alignment_respects_padding() -> None:
    neg_cent = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, 4)
    mask[0] = 1
    mask[1, :6, :3] = 1
    path = maximum_path(neg_cent, mask)
    assert path[1, 6:].sum() == 0 and path[1, :, 3:].sum() == 0
    assert path[1, :6].sum() == 6


@pytest.fixture(scope="module")
def runtime():
    rt = load_runtime(REPO / "model" / "runtime")
    rt.models.monotonic_align = type("MAS", (), {"maximum_path": staticmethod(maximum_path)})
    return rt


def _pair(runtime):
    torch.manual_seed(0)
    reference = runtime.models.SynthesizerTrn(N_VOCAB, SPEC, SEGMENT_FRAMES, **copy.deepcopy(TINY))
    v3 = SynthesizerV3(runtime, N_VOCAB, SPEC, SEGMENT_FRAMES, copy.deepcopy(TINY), maximum_path)
    new = v3.load_v2_state(reference.state_dict())
    return reference.eval(), v3.eval(), new


def _inputs():
    torch.manual_seed(5)
    x = torch.randint(1, N_VOCAB, (2, 6))
    x_lengths = torch.tensor([6, 4])
    y = torch.randn(2, SPEC, 24)
    y_lengths = torch.tensor([24, 18])
    return x, x_lengths, y, y_lengths


def test_runtime_import_does_not_leak_module_names(runtime) -> None:
    assert "models" not in sys.modules or sys.modules["models"] is not runtime.models


def test_v2_weights_load_and_only_new_parts_are_left(runtime) -> None:
    _, _, new = _pair(runtime)
    assert {key.split(".")[0] for key in new} == {"lang_emb", "pitch", "prior_f0", "source", "source_convs"}


def test_training_forward_equals_v2(runtime) -> None:
    reference, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    torch.manual_seed(7)
    o, l_length, attn, ids_slice, _, _, (z, z_p, m_p, logs_p, m_q, logs_q) = reference(x, x_lengths, y, y_lengths)
    torch.manual_seed(7)
    out = v3(x, x_lengths, y, y_lengths)
    assert torch.equal(ids_slice, out["ids_slice"])
    assert torch.equal(attn, out["attn"])
    for name, expected in {"audio": o, "l_length": l_length, "z": z, "z_p": z_p, "m_p": m_p, "logs_p": logs_p,
                           "m_q": m_q, "logs_q": logs_q}.items():
        assert torch.allclose(expected, out[name], atol=1e-6), name


def test_duration_outputs_are_what_v2_trains_on(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    out = v3(x, x_lengths, y, y_lengths)
    frames = (torch.exp(out["logw_target"]) * out["x_mask"]).sum(dim=2).squeeze(1).round()
    assert torch.equal(frames, y_lengths.float())  # every frame belongs to one token
    expected = torch.sum((out["logw"] - out["logw_target"]) ** 2, [1, 2]) / torch.sum(out["x_mask"])
    assert torch.allclose(out["l_length"], expected)
    assert out["hidden"].shape == (2, TINY["hidden_channels"], 6)


def test_inference_equals_v2(runtime) -> None:
    reference, v3, _ = _pair(runtime)
    x, x_lengths, _, _ = _inputs()
    torch.manual_seed(9)
    expected, expected_attn, _, _ = reference.infer(x, x_lengths, noise_scale=0.3, length_scale=1.1)
    torch.manual_seed(9)
    got, attn, _, f0 = v3.infer(x, x_lengths, noise_scale=0.3, length_scale=1.1)
    assert torch.equal(expected_attn, attn)
    assert torch.allclose(expected, got, atol=1e-6)  # the zero-initialised pitch shift changes nothing yet
    assert f0.shape[-1] == attn.shape[2]


def test_language_vector_changes_the_output_once_trained(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, _, _ = _inputs()
    base = v3.encode_text(x, x_lengths)[0]
    with torch.no_grad():
        v3.lang_emb.weight[1].normal_()
    assert torch.equal(base, v3.encode_text(x, x_lengths, torch.zeros(2, dtype=torch.long))[0])
    assert not torch.allclose(base, v3.encode_text(x, x_lengths, torch.ones(2, dtype=torch.long))[0])


def _true_f0(y_lengths: torch.Tensor, frames: int) -> torch.Tensor:
    torch.manual_seed(11)
    f0 = 180.0 * torch.pow(2.0, torch.randn(len(y_lengths), frames) / 4)
    f0[:, ::5] = 0.0  # some unvoiced frames
    for i, n in enumerate(y_lengths.tolist()):
        f0[i, n:] = 0.0
    return f0


def test_training_forward_with_true_pitch_still_equals_v2(runtime) -> None:
    reference, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    f0 = _true_f0(y_lengths, y.size(2))  # before seeding: it draws random numbers itself
    torch.manual_seed(7)
    o, _, _, _, _, _, (_, _, m_p, _, _, _) = reference(x, x_lengths, y, y_lengths)
    torch.manual_seed(7)
    out = v3(x, x_lengths, y, y_lengths, f0=f0)
    assert torch.allclose(o, out["audio"], atol=1e-6)
    assert torch.allclose(m_p, out["m_p"], atol=1e-6)


def test_pitch_outputs_are_masked_frame_tracks(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    out = v3(x, x_lengths, y, y_lengths, f0=_true_f0(y_lengths, y.size(2)))
    for key in ("semitones", "voiced", "semitones_pred", "voicing_logit"):
        assert out[key].shape == (2, 1, y.size(2)), key
        assert torch.all(out[key][1, :, 18:] == 0), key
    assert torch.all(out["semitones"][out["voiced"] == 0] == 0)


def test_trained_prior_shift_moves_the_prior_with_the_pitch(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    f0 = _true_f0(y_lengths, y.size(2))
    torch.manual_seed(7)
    before = v3(x, x_lengths, y, y_lengths, f0=f0)["m_p"]
    with torch.no_grad():
        v3.prior_f0.proj.weight.normal_()
    torch.manual_seed(7)
    after = v3(x, x_lengths, y, y_lengths, f0=f0)["m_p"]
    assert not torch.allclose(before, after)
    assert torch.all(after[1, :, 18:] == 0)


def test_pitch_range_and_shift_act_on_the_predicted_contour(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, _, _ = _inputs()
    torch.manual_seed(9)
    _, attn, y_mask, _ = v3.infer(x, x_lengths)
    hidden, _, _, _ = v3.encode_text(x, x_lengths)
    st, voiced, hz = v3.predict_pitch(hidden, attn, y_mask)
    st2, voiced2, _ = v3.predict_pitch(hidden, attn, y_mask, pitch_shift=2.0, pitch_range=1.5)
    assert torch.equal(voiced, voiced2)
    assert torch.allclose(st2, (st * 1.5 + 2.0) * voiced, atol=1e-6)
    assert torch.all((hz > 0) == (voiced > 0))


def test_decode_with_pitch_equals_v2_decoder_until_trained(runtime) -> None:
    _, v3, _ = _pair(runtime)
    z = torch.randn(2, TINY["inter_channels"], SEGMENT_FRAMES)
    f0 = torch.full((2, 1, SEGMENT_FRAMES), 190.0)
    assert torch.allclose(v3.dec(z), v3.decode(z), atol=1e-6)
    assert torch.allclose(v3.dec(z), v3.decode(z, f0), atol=1e-6)
    with torch.no_grad():
        for conv in v3.source_convs:
            conv.weight.normal_(std=0.1)
    assert not torch.allclose(v3.dec(z), v3.decode(z, f0), atol=1e-4)


def test_training_forward_slices_the_pitch_with_the_latent(runtime) -> None:
    _, v3, _ = _pair(runtime)
    x, x_lengths, y, y_lengths = _inputs()
    f0 = _true_f0(y_lengths, y.size(2))
    out = v3(x, x_lengths, y, y_lengths, f0=f0)
    for b, start in enumerate(out["ids_slice"].tolist()):
        assert torch.equal(out["f0_slice"][b, 0], f0[b, start: start + SEGMENT_FRAMES])


def test_full_size_inference_network_fits_the_budget(runtime) -> None:
    import json
    config = json.loads((REPO / "model" / "config.json").read_text(encoding="utf-8"))
    model = {k: v for k, v in config["model"].items() if k != "inference_only"}
    full = SynthesizerV3(runtime, 178, config["data"]["filter_length"] // 2 + 1,
                         config["train"]["segment_size"] // config["data"]["hop_length"], model, maximum_path)
    inference = sum(p.numel() for name, p in full.named_parameters() if not name.startswith("enc_q."))
    added = sum(p.numel() for name, p in full.named_parameters()
                if name.split(".")[0] in ("lang_emb", "pitch", "prior_f0", "source", "source_convs"))
    assert inference <= 12_000_000
    assert added <= 1_000_000
