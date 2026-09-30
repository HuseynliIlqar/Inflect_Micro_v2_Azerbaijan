"""Azerbaijani text to speech: the working layer around the model.

The package in `model/` was written for an English pipeline: it phonemises the
text as it stands and only splits after 280 characters.  For Azerbaijani that
falls short in two places -- `II` is read as "ı ı", and the tail of a long
sentence flattens out.

This module closes that gap:

1. `normalize_az` -- rewrites digits, Roman numerals, percentages, units and
   abbreviations as the words a reader would say;
2. `chunk_text` -- cuts sentences into roughly fifteen-word chunks, because the
   9.36M model loses intonation well before the package's own 280-character
   limit;
3. `restress` (optional) -- thins out the stress eSpeak puts on nearly every
   word.

Before any of that, `censor_az` swaps obscenities for a bleep marker, and the
marker is spoken as a tone. It is on unless the caller passes `censor=False`.

Example::

    >>> from aztts import AzTTS
    >>> tts = AzTTS()
    >>> tts.save("Salam, necəsiniz?", "out/salam.wav")

The object is expensive (~0.3 s to load, weights stay resident) -- build it once
and reuse it.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from .az_chunk import DEFAULT_MAX_WORDS, chunk_text
from .az_profanity import BLEEP, bleep_segments, censor_az
from .az_prosody import restress
from .az_text import normalize_az

__all__ = ["AzTTS", "DEFAULT_MODEL_DIR", "ModelNotFoundError", "bleep"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"

Prosody = Literal["off", "safe", "wide"]

# The tone that stands in for an obscenity: the broadcast 1 kHz, quiet enough
# to sit under the voice, faded at both ends so it does not click.
BLEEP_HZ = 1000.0
BLEEP_SECONDS = 0.35
BLEEP_LEVEL = 0.25
BLEEP_FADE_SECONDS = 0.005


def bleep(sample_rate: int) -> np.ndarray:
    """The tone played in place of a censored word, as float32 mono."""
    length = round(sample_rate * BLEEP_SECONDS)
    time = np.arange(length, dtype=np.float32) / sample_rate
    tone = BLEEP_LEVEL * np.sin(2 * np.pi * BLEEP_HZ * time)
    fade = min(round(sample_rate * BLEEP_FADE_SECONDS), length // 2)
    envelope = np.ones(length, dtype=np.float32)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        envelope[:fade] = ramp
        envelope[-fade:] = ramp[::-1]
    return (tone * envelope).astype(np.float32)


class ModelNotFoundError(FileNotFoundError):
    """There is no `model.pth` in the `model/` directory."""


def _load_package(model_dir: Path):
    """Import the package's own `inference` module, with its runtime on the path."""
    if not (model_dir / "model.pth").is_file():
        raise ModelNotFoundError(
            f"model.pth not found: {model_dir}\n"
            "Check that the directory was copied in full (model/ should be ~37 MB)."
        )
    # The package ships its own runtime; put it first so it takes precedence.
    root = str(model_dir)
    if root not in sys.path:
        sys.path.insert(0, root)
    with warnings.catch_warnings():
        # The VITS runtime uses the old weight_norm API; harmless for synthesis.
        warnings.filterwarnings("ignore", message=".*weight_norm.*")
        import inference  # type: ignore[import-not-found]

    return inference


class AzTTS:
    """Offline TTS engine that speaks Azerbaijani."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "cpu",
    ) -> None:
        model_dir = Path(model_dir).resolve()
        package = _load_package(model_dir)
        self._package = package
        with warnings.catch_warnings():
            # The VITS runtime uses the old weight_norm API; harmless for synthesis.
            warnings.filterwarnings("ignore", message=".*weight_norm.*")
            self._tts = package.InflectTTS(str(model_dir), device=device)
        self.model_dir = model_dir
        self.device = device
        self.sample_rate: int = self._tts.sample_rate

    # -- internal helpers -------------------------------------------------

    def _phonemes_for(self, chunk: str, prosody: Prosody, drop: bool):
        """A restressed (text, phonemes) pair, or None when restressing is off."""
        if prosody == "off":
            return None
        from inflect_vits_frontend import run_vits_frontend  # type: ignore[import-not-found]

        output = run_vits_frontend(chunk)
        edited = restress(
            output.phoneme_text,
            output.normalized_text,
            level=prosody,
            to_secondary=not drop,
        )
        return output.normalized_text, edited

    def _speak_chunk(self, chunk: str, **options) -> np.ndarray:
        """Speak a chunk, playing a bleep wherever `censor_az` left one."""
        if BLEEP not in chunk:
            return self._speak_text(chunk, **options)
        pieces = [
            bleep(self.sample_rate)
            if segment == BLEEP
            else self._speak_text(segment, **options)
            for segment in bleep_segments(chunk)
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def _speak_text(
        self,
        chunk: str,
        *,
        speed: float,
        variation: float,
        seed: int,
        prosody: Prosody,
        prosody_drop: bool,
    ) -> np.ndarray:
        prepared = self._phonemes_for(chunk, prosody, prosody_drop)
        if prepared is None:
            return self._tts.synthesize(
                chunk, speed=speed, variation=variation, seed=seed
            )[1]
        text, phonemes = prepared
        return self._tts.synthesize(
            text, phonemes=phonemes, speed=speed, variation=variation, seed=seed
        )[1]

    # -- public API -------------------------------------------------------

    def prepare(
        self,
        text: str,
        *,
        normalize: bool = True,
        max_words: int = DEFAULT_MAX_WORDS,
        censor: bool = True,
    ) -> tuple[str, ...]:
        """Return the chunks that would reach the model, without synthesising.

        With `censor` (the default) each obscenity is a `BLEEP` in the chunks.
        """
        source = censor_az(text) if censor else text
        prepared = normalize_az(source) if normalize else " ".join(source.split())
        if not prepared:
            return ()
        if max_words <= 0:
            return (prepared,)
        return chunk_text(prepared, max_words=max_words)

    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 7,
        normalize: bool = True,
        max_words: int = DEFAULT_MAX_WORDS,
        prosody: Prosody = "off",
        prosody_drop: bool = False,
        censor: bool = True,
    ) -> np.ndarray:
        """Synthesise the text into a float32 mono array ([-1, 1], 24 kHz).

        Obscenities are bleeped unless `censor=False` is passed explicitly.
        """
        chunks = self.prepare(
            text, normalize=normalize, max_words=max_words, censor=censor
        )
        if not chunks:
            raise ValueError("The text is empty.")
        pause = self._package.boundary_pause_seconds
        pieces: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index:
                # The pause length comes from the previous chunk's final mark.
                silence = round(self.sample_rate * pause(chunks[index - 1]))
                pieces.append(np.zeros(silence, dtype=np.float32))
            pieces.append(
                self._speak_chunk(
                    chunk,
                    speed=speed,
                    variation=variation,
                    seed=seed + index,
                    prosody=prosody,
                    prosody_drop=prosody_drop,
                )
            )
        return np.clip(np.concatenate(pieces), -1.0, 1.0)

    def save(self, text: str, output: str | Path, **options) -> Path:
        """Synthesise the text and write it as a WAV file; return the path."""
        import soundfile as sf

        destination = Path(output)
        if destination.suffix.lower() != ".wav":
            destination = destination.with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        waveform = self.synthesize(text, **options)
        sf.write(destination, waveform, self.sample_rate)
        return destination
