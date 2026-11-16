"""Pick the best v3 export: render v2 and several exports on the same sentences, twice each, and measure them.

    python eval_candidates.py --run /workspace/runs/v3_full --base /workspace/base --prepared /workspace/prepared \
        --kit /workspace/kit --init /workspace/keep/polish-30k-slim.pth --steps 10000 15000 20000 22500 \
        --test /workspace/kit/v2_validation.jsonl --count 60 --out /workspace/eval_candidates

Sentences: the first --count of v2's held-out test sentences (never trained on) plus the 12 listening sentences.
Per file: buzz and ring (buzz_measure.py, as in every earlier Avaz comparison), the level of the fixed ~9 kHz
line, the intonation range from Praat, and SQUIM's reference-free PESQ/STOI estimates. The real recordings of
the test sentences are measured too, as the target. Writes <out>/summary.json and <out>/<name>/*.wav.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parents[1] / "v2" / "pod"))

from listen_v3 import VARIATION, load_frontend, tokens_for  # noqa: E402
from monotonic_align import maximum_path  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_fd import SynthesizerFD  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402

SEEDS = (0, 1)
F0_REF = 178.0


def line_level(x: np.ndarray, sr: int) -> float:
    """dB of the strongest 8.7-9.3 kHz bin over the 7.5-10.5 kHz median, on the mean speech spectrum."""
    frames = np.lib.stride_tricks.sliding_window_view(x, 2048)[::256] * np.hanning(2048)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(2048, 1 / sr)
    energy = 10 * np.log10(power.sum(axis=1) + 1e-12)
    spec = 10 * np.log10(power[energy > energy.max() - 45].mean(axis=0) + 1e-12)
    band, line = (freqs >= 7500) & (freqs <= 10500), (freqs >= 8700) & (freqs <= 9300)
    return float(spec[line].max() - np.median(spec[band]))


def measure(path: str) -> dict:
    import buzz_measure as bm
    import parselmouth

    m = bm.measure(path)
    x, sr = sf.read(path, dtype="float32")
    f0 = parselmouth.Sound(x, sr).to_pitch_ac(time_step=0.01, pitch_floor=75, pitch_ceiling=500).selected_array["frequency"]
    st = 12 * np.log2(f0[f0 > 0] / F0_REF) if (f0 > 0).sum() > 10 else np.zeros(2)
    lo, hi = np.percentile(st, [5, 95])
    return {"buzz": m["h2-4k"], "low": m["h1-2k"], "ring": m["ring_share"], "line_db": line_level(x, sr),
            "range": float(hi - lo), "seconds": m["seconds"]}


def build_model(args, symbols, device, cls=SynthesizerV3) -> SynthesizerV3:
    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = cls(load_runtime(args.base / "runtime"), len(symbols), config["data"]["filter_length"] // 2 + 1,
                          config["train"]["segment_size"] // config["data"]["hop_length"], options, maximum_path,
                          f0_ref_hz=F0_REF)
    model.load_v2_state(torch.load(args.init, map_location="cpu", weights_only=False)["generator"])
    return model.to(device).eval()


def render(model, tokens, folder: Path, use_pitch: bool, device) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    with torch.no_grad():
        for i, x in enumerate(tokens):
            for seed in SEEDS:
                torch.manual_seed(seed)
                x_dev = x.to(device)
                audio, _, _, _ = model.infer(x_dev, torch.tensor([x.size(1)], device=device), noise_scale=VARIATION,
                                             use_pitch=use_pitch)
                path = folder / f"{i:03d}-{seed}.wav"
                sf.write(path, audio[0, 0].float().cpu().numpy(), 24000, subtype="PCM_16")
                paths.append(str(path))
    return paths


def squim(paths: list[str], device) -> dict:
    import torchaudio
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    model = SQUIM_OBJECTIVE.get_model().to(device).eval()
    stoi, pesq = [], []
    with torch.no_grad():
        for p in paths:
            x, sr = sf.read(p, dtype="float32")
            wav = torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000).to(device)
            s, q, _ = model(wav)
            stoi.append(float(s))
            pesq.append(float(q))
    return {"stoi": statistics.mean(stoi), "pesq": statistics.mean(pesq)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "base", "prepared", "kit", "init", "test", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=[], help="v3 exports in --run")
    parser.add_argument("--fd-run", type=Path, help="a frame-decoder run (train_fdec.py)")
    parser.add_argument("--fd-steps", type=int, nargs="*", default=[], help="its exports to measure")
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = json.loads((args.prepared / "symbols.json").read_text(encoding="utf-8"))
    symbols = payload["symbols"] if isinstance(payload, dict) else payload
    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    add_blank = bool(config["data"].get("add_blank", True))
    frontend = load_frontend(args.kit)

    validation = {json.loads(l)["normalized_text"]: json.loads(l) for l in
                  (args.prepared / "validation.jsonl").read_text(encoding="utf-8").splitlines() if l}
    test_rows = [json.loads(l) for l in args.test.read_text(encoding="utf-8").splitlines() if l][: args.count]
    listen = json.loads((HERE / "listen_sentences_v3.json").read_text(encoding="utf-8"))
    texts = [r["normalized_text"] for r in test_rows] + listen
    tokens = [tokens_for(frontend.phonemize(frontend.normalize(t)), symbols, add_blank) for t in texts]
    real = [str(args.prepared / validation[r["normalized_text"]]["audio"]) for r in test_rows
            if r["normalized_text"] in validation]

    model = build_model(args, symbols, device)
    files = {"v2": render(model, tokens, args.out / "v2", use_pitch=False, device=device)}
    for step in args.steps:
        state = torch.load(args.run / "exports" / f"model-step-{step:08d}.pth", map_location="cpu",
                           weights_only=False)["model"]
        model.load_state_dict(state, strict=False)
        files[f"step-{step}"] = render(model, tokens, args.out / f"step-{step}", use_pitch=True, device=device)
        print("rendered", step, flush=True)
    if args.fd_steps:
        fd = build_model(args, symbols, device, cls=SynthesizerFD)
        for step in args.fd_steps:
            state = torch.load(args.fd_run / "exports" / f"model-step-{step:08d}.pth", map_location="cpu",
                               weights_only=False)["model"]
            result = fd.load_state_dict(state, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(f"fd step {step}: unexpected {result.unexpected_keys[:3]}")
            files[f"fd-{step}"] = render(fd, tokens, args.out / f"fd-{step}", use_pitch=True, device=device)
            print("rendered fd", step, flush=True)
    files["real"] = real

    n_test = len(test_rows) * len(SEEDS)
    summary = {}
    with ProcessPoolExecutor(6) as pool:
        for name, paths in files.items():
            rows = list(pool.map(measure, paths, chunksize=8))
            q = squim(paths, device)
            part = lambda rs: {k: round(float(np.nanmean([r[k] for r in rs])), 3)  # noqa: E731
                               for k in ("buzz", "low", "ring", "line_db", "range")}
            summary[name] = {"all": {**part(rows), **{k: round(v, 3) for k, v in q.items()}},
                             "test": part(rows[:n_test]) if name != "real" else part(rows),
                             "listen": part(rows[n_test:]) if name != "real" else None, "files": len(paths)}
            print(name, json.dumps(summary[name]["all"]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
