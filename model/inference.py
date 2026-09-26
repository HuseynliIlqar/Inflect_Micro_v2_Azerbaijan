from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

import commons
import utils
from inflect_vits_frontend import run_vits_frontend
from models import SynthesizerTrn
from text import cleaned_text_to_sequence
from text.symbols import symbols


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:。！？；：])\s*", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":", "，", "；", "："))
            split_at = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28, "？": 0.28, "!": 0.24, "！": 0.24,
        ".": 0.22, "。": 0.22, ";": 0.16, "；": 0.16,
        ":": 0.13, "：": 0.13, ",": 0.09, "，": 0.09,
    }.get(ending, 0.08)


def edge_fade(
    waveform: np.ndarray,
    sample_rate: int,
    milliseconds: float = 5.0,
) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


class InflectTTS:
    def __init__(
        self,
        model_dir: str | Path = PACKAGE_ROOT,
        device: str = "cpu",
    ) -> None:
        self.root = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.hps = utils.get_hparams_from_file(str(self.root / "config.json"))
        self.model = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model,
        ).to(self.device).eval()
        self.n_speakers = int(getattr(self.hps.model, "n_speakers", 0) or 0)
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        try:
            root_logger.setLevel(logging.WARNING)
            utils.load_checkpoint(str(self.root / "model.pth"), self.model, None)
        finally:
            root_logger.setLevel(previous_level)
        self.sample_rate = int(self.hps.data.sampling_rate)

    def _tokens(
        self,
        text: str | None = None,
        *,
        phonemes: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = run_vits_frontend(text, phonemes=phonemes)
        sequence = cleaned_text_to_sequence(output.phoneme_text)
        if self.hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        if not sequence:
            raise ValueError("The deployment frontend produced no speakable tokens.")
        tokens = torch.LongTensor(sequence).to(self.device).unsqueeze(0)
        lengths = torch.LongTensor([tokens.size(1)]).to(self.device)
        return tokens, lengths

    @torch.inference_mode()
    def synthesize(
        self,
        text: str | None = None,
        *,
        phonemes: str | None = None,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
        speaker: int = 0,
    ) -> tuple[int, np.ndarray]:
        if self.n_speakers > 0:
            if not 0 <= speaker < self.n_speakers:
                raise ValueError(
                    f"speaker must be between 0 and {self.n_speakers - 1}"
                )
            sid = torch.LongTensor([speaker]).to(self.device)
        elif speaker:
            raise ValueError("This package contains a single voice; speaker must be 0.")
        else:
            sid = None
        if phonemes is None:
            normalized = " ".join((text or "").split())
            if not normalized:
                raise ValueError("Text must not be empty.")
            chunks: list[tuple[str | None, str | None]] = [
                (chunk, None) for chunk in split_text(normalized)
            ]
        else:
            chunks = [(text, phonemes)]
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if not 0.0 <= variation <= 1.0:
            raise ValueError("variation must be between 0.0 and 1.0")
        pieces: list[np.ndarray] = []
        for index, (chunk_text, chunk_phonemes) in enumerate(chunks):
            if index:
                previous = chunks[index - 1][0] or ""
                pieces.append(
                    np.zeros(
                        round(self.sample_rate * boundary_pause_seconds(previous)),
                        dtype=np.float32,
                    )
                )
            tokens, lengths = self._tokens(chunk_text, phonemes=chunk_phonemes)
            torch.manual_seed(seed + index)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed + index)
            waveform = self.model.infer(
                tokens,
                lengths,
                sid=sid,
                noise_scale=variation,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
                max_len=4000,
            )[0][0, 0].float().cpu().numpy()
            pieces.append(edge_fade(waveform, self.sample_rate))
        return self.sample_rate, np.clip(np.concatenate(pieces), -1.0, 1.0)

    def save(
        self,
        text: str | None,
        output: str | Path,
        **kwargs: object,
    ) -> Path:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample_rate, waveform = self.synthesize(text, **kwargs)
        sf.write(destination, waveform, sample_rate)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Inflect synthesis.")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE_ROOT)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--phonemes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    engine = InflectTTS(args.model_dir, args.device)
    engine.save(
        args.text,
        args.output,
        phonemes=args.phonemes,
        speed=args.speed,
        variation=args.variation,
        seed=args.seed,
    )
    print(f"wrote {args.output} at {engine.sample_rate} Hz")


if __name__ == "__main__":
    main()
