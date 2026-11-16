"""Check that the full-size v3 inference graph exports to ONNX, matches PyTorch and is fast enough.

    python onnx_check.py --out D:/ktts/avaz-train/onnx_check

Needs torch, onnx and onnxruntime (not the repository's test environment). It uses v2's real
dimensions with random weights: export, parity and speed do not depend on trained values. The
source injections get random weights so the harmonic path is exercised. Noise is switched off on
both sides so the outputs can be compared sample for sample. Speed is ONNX Runtime on one thread,
against the same graph without the pitch part and the source (v2's computation).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from monotonic_align import maximum_path  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402


class TextToAudio(torch.nn.Module):
    def __init__(self, net: SynthesizerV3, use_pitch: bool) -> None:
        super().__init__()
        self.net, self.use_pitch = net, use_pitch

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, length_scale: torch.Tensor) -> torch.Tensor:
        audio, _, _, _ = self.net.infer(x, x_lengths, noise_scale=0.0, length_scale=length_scale,
                                        use_pitch=self.use_pitch, source_noise=False)
        return audio


def _export(graph: TextToAudio, path: Path, x, x_lengths, length_scale) -> None:
    torch.onnx.export(graph, (x, x_lengths, length_scale), str(path), dynamo=False, opset_version=17,
                      input_names=["x", "x_lengths", "length_scale"], output_names=["audio"],
                      dynamic_axes={"x": {1: "tokens"}, "audio": {2: "samples"}})


def _rtf(session: ort.InferenceSession, feeds: dict, repeats: int = 3) -> float:
    session.run(None, feeds)
    started = time.perf_counter()
    for _ in range(repeats):
        audio = session.run(None, feeds)[0]
    return (time.perf_counter() - started) / repeats / (audio.shape[-1] / 24000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    config = json.loads((REPO / "model" / "config.json").read_text(encoding="utf-8"))
    model = {k: v for k, v in config["model"].items() if k != "inference_only"}
    runtime = load_runtime(REPO / "model" / "runtime")
    torch.manual_seed(0)
    net = SynthesizerV3(runtime, 178, 513, 64, model, maximum_path).eval()
    with torch.no_grad():
        for conv in net.source_convs:
            conv.weight.normal_(std=0.05)
        net.prior_f0.proj.weight.normal_(std=0.05)
    for module in net.modules():  # export the plain weights, as the v2 export does
        if hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)

    x = torch.randint(1, 178, (1, 120))  # about 60 phonemes with blanks: one long sentence
    x_lengths = torch.tensor([120])
    length_scale = torch.tensor(1.0)
    report = {}
    for name, use_pitch in (("v3", True), ("v2_path", False)):
        graph = TextToAudio(net, use_pitch).eval()
        path = args.out / f"{name}.onnx"
        _export(graph, path, x, x_lengths, length_scale)
        with torch.no_grad():
            expected = graph(x, x_lengths, length_scale).numpy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        feeds = {"x": x.numpy(), "x_lengths": x_lengths.numpy(), "length_scale": length_scale.numpy()}
        got = session.run(None, feeds)[0]
        report[name] = {"samples": int(got.shape[-1]), "max_abs_diff": float(np.max(np.abs(got - expected)))
                        if got.shape == expected.shape else "shape mismatch",
                        "rtf_1_thread": round(_rtf(session, feeds), 4), "mb": round(path.stat().st_size / 1e6, 1)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
