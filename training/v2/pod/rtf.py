"""CPU speed of the exported package: real-time factor (seconds of compute per second of audio)
for PyTorch and ONNX Runtime at 1 and 4 threads, on 12 validation sentences. The pod's server
CPU is faster than a phone's, so read the numbers relative to each other.

    python rtf.py   (training venv with onnxruntime)
Writes /workspace/asr_eval/rtf.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

PACKAGE = Path("/workspace/final/avaz-v2")
sys.path.insert(0, str(PACKAGE))
from inference import InflectTTS  # noqa: E402

ROWS = json.loads(Path("/workspace/asr_eval/rows.json").read_text(encoding="utf-8"))[:12]
_DTYPES = {"tensor(int64)": np.int64, "tensor(float)": np.float32}


def feed(session, values: dict) -> dict:
    return {i.name: np.asarray(values[i.name], dtype=_DTYPES[i.type]) for i in session.get_inputs()}


def onnx_rtf(engine, threads: int) -> float:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    duration = ort.InferenceSession(str(PACKAGE / "onnx" / "duration.onnx"), options, providers=["CPUExecutionProvider"])
    decode = ort.InferenceSession(str(PACKAGE / "onnx" / "decode.onnx"), options, providers=["CPUExecutionProvider"])
    compute = audio = 0.0
    for row in ROWS:
        tokens, lengths = engine._tokens(None, phonemes=row["phonemes"])
        start = time.perf_counter()
        m, logs, mask = duration.run(None, feed(duration, {"tokens": tokens.numpy(), "lengths": lengths.numpy(),
                                                           "length_scale": [1.0]}))
        noise = np.random.default_rng(0).standard_normal(m.shape).astype(np.float32)
        wave = decode.run(None, feed(decode, {"m_p_exp": m, "logs_p_exp": logs, "y_mask": mask,
                                              "zp_noise": noise, "noise_scale": [0.667]}))[0]
        compute += time.perf_counter() - start
        audio += wave.size / engine.sample_rate
    return round(compute / audio, 4)


def torch_rtf(engine, threads: int) -> float:
    torch.set_num_threads(threads)
    compute = audio = 0.0
    for row in ROWS:
        start = time.perf_counter()
        rate, wave = engine.synthesize(row["text"], phonemes=row["phonemes"])
        compute += time.perf_counter() - start
        audio += len(wave) / rate
    return round(compute / audio, 4)


def main() -> None:
    engine = InflectTTS(PACKAGE, device="cpu")
    engine.synthesize("Salam.")  # warm-up
    report = {"pytorch_1_thread": torch_rtf(engine, 1), "pytorch_4_threads": torch_rtf(engine, 4),
              "onnx_1_thread": onnx_rtf(engine, 1), "onnx_4_threads": onnx_rtf(engine, 4),
              "params_millions": round(sum(p.numel() for p in engine.model.parameters()) / 1e6, 2)}
    Path("/workspace/asr_eval/rtf.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
