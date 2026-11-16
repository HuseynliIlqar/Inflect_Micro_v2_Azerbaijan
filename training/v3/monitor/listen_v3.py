"""Read the same sentences with every new v3 checkpoint, for the monitor's "Dinlə" section.

    python listen_v3.py --run /workspace/runs/v3_trial --base /workspace/base --prepared /workspace/prepared \
        --kit /workspace/kit --init /workspace/keep/polish-30k-slim.pth

Writes, once:
- <run>/listen/base/NN.wav: v2 (the --init weights, without the pitch part), the "before";
- <run>/listen/dataset_metrics.json: buzz and ring of 40 training recordings, the target;
- <run>/validation/before.wav, target.wav and sentence.json for the first validation clip;
then <run>/listen/step-XXXXXXXX/NN.wav for every exports/model-step-*.pth, with buzz and ring in
metrics.json. Sentences: listen_sentences_v3.json (v2's eight plus three yes/no questions and an
exclamation). Runs on the CPU with two threads, so training keeps the GPU and the data loaders.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import shutil
import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model"))
sys.path.insert(0, str(HERE.parents[1] / "v2" / "pod"))

from monotonic_align import maximum_path  # noqa: E402
from runtime import load_runtime  # noqa: E402
from synthesizer_v3 import SynthesizerV3  # noqa: E402

SEED = 0
VARIATION = 0.3
_EXPORT = re.compile(r"^model-step-(\d{8})\.pth$")


def load_frontend(kit: Path):
    spec = importlib.util.spec_from_file_location("az_stress_frontend_listen", kit / "az_stress_frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_frontend("az")


def tokens_for(phonemes: str, symbols: list[str], add_blank: bool) -> torch.Tensor:
    index = {s: i for i, s in enumerate(symbols)}
    ids = [index[s] for s in phonemes if s in index]
    if add_blank:
        expanded = [0] * (len(ids) * 2 + 1)
        expanded[1::2] = ids
        ids = expanded
    return torch.tensor([ids], dtype=torch.long)


def render(model: SynthesizerV3, folder: Path, tokens: list[torch.Tensor], use_pitch: bool) -> None:
    staging = folder.with_name(folder.name + ".partial")
    staging.mkdir(parents=True, exist_ok=True)
    model.eval()
    for i, x in enumerate(tokens):
        torch.manual_seed(SEED)
        audio, _, _, _ = model.infer(x, torch.tensor([x.size(1)]), noise_scale=VARIATION, use_pitch=use_pitch)
        sf.write(staging / f"{i:02d}.wav", audio[0, 0].numpy(), 24000, subtype="PCM_16")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)


def summary(paths: list[Path]) -> dict:
    import buzz_measure as bm  # the same measurement as every earlier Avaz comparison

    rows = [bm.measure(str(p)) for p in paths]
    finite = lambda key: [r[key] for r in rows if r[key] == r[key]]  # noqa: E731
    return {"h2_4k": statistics.median(finite("h2-4k")), "h1_2k": statistics.median(finite("h1-2k")),
            "ring_share": statistics.median(finite("ring_share")), "ring_hz": statistics.median(finite("ring_hz")),
            "clips": len(rows)}


def validation_reference(run: Path, prepared: Path, model: SynthesizerV3, symbols: list[str], add_blank: bool) -> None:
    folder = run / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "before.wav").exists():
        return
    row = json.loads((prepared / "validation.jsonl").read_text(encoding="utf-8").splitlines()[0])
    shutil.copyfile(prepared / row["audio"], folder / "target.wav")
    (folder / "sentence.json").write_text(json.dumps({"text": row.get("text")}, ensure_ascii=False), encoding="utf-8")
    torch.manual_seed(SEED)
    x = tokens_for(row["phonemes"], symbols, add_blank)
    audio, _, _, _ = model.infer(x, torch.tensor([x.size(1)]), noise_scale=VARIATION, use_pitch=False)
    sf.write(folder / "before.wav", audio[0, 0].numpy(), 24000, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--f0-ref", type=float, default=178.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    # The pod's CPU quota (about 7.6 CPUs) is shared with the trainer's data loaders: one thread, lowest priority.
    torch.set_num_threads(1)
    os.nice(19)

    config = json.loads((args.base / "config.json").read_text(encoding="utf-8"))
    payload = json.loads((args.prepared / "symbols.json").read_text(encoding="utf-8"))
    symbols = payload["symbols"] if isinstance(payload, dict) else payload
    add_blank = bool(config["data"].get("add_blank", True))
    options = {k: v for k, v in config["model"].items() if k != "inference_only"}
    model = SynthesizerV3(load_runtime(args.base / "runtime"), len(symbols), config["data"]["filter_length"] // 2 + 1,
                          config["train"]["segment_size"] // config["data"]["hop_length"], options, maximum_path,
                          f0_ref_hz=args.f0_ref)
    model.load_v2_state(torch.load(args.init, map_location="cpu", weights_only=False)["generator"])

    sentences = json.loads((HERE / "listen_sentences_v3.json").read_text(encoding="utf-8"))
    frontend = load_frontend(args.kit)
    tokens = [tokens_for(frontend.phonemize(frontend.normalize(text)), symbols, add_blank) for text in sentences]
    out = args.run / "listen"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")
    validation_reference(args.run, args.prepared, model, symbols, add_blank)
    if not (out / "base").is_dir():
        render(model, out / "base", tokens, use_pitch=False)
        print("rendered v2", flush=True)
    if not (out / "dataset_metrics.json").exists():
        clips = sorted((args.prepared / "audio").glob("*.wav"))
        sample = random.Random(0).sample(clips, min(40, len(clips)))
        (out / "dataset_metrics.json").write_text(json.dumps(summary(sample)), encoding="utf-8")
    while True:
        for path in sorted((args.run / "exports").glob("model-step-*.pth")):
            match = _EXPORT.match(path.name)
            if not match or (out / f"step-{match.group(1)}").is_dir():
                continue
            time.sleep(5)  # let the trainer finish writing
            state = torch.load(path, map_location="cpu", weights_only=False)["model"]
            result = model.load_state_dict(state, strict=False)
            stray = [k for k in result.missing_keys if not k.startswith("enc_q.")]
            if result.unexpected_keys or stray:
                raise RuntimeError(f"{path.name}: unexpected {result.unexpected_keys[:3]}, missing {stray[:3]}")
            render(model, out / f"step-{match.group(1)}", tokens, use_pitch=True)
            print(f"rendered step {match.group(1)}", flush=True)
        for folder in sorted(out.iterdir()):
            if folder.is_dir() and not folder.name.endswith(".partial") and not (folder / "metrics.json").exists():
                clips = sorted(folder.glob("[0-9][0-9].wav"))
                if clips:
                    (folder / "metrics.json").write_text(json.dumps(summary(clips)), encoding="utf-8")
        if args.once:
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
