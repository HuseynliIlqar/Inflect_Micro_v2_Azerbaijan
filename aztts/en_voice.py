"""The English base model, speaking through the Azerbaijani export's runtime.

`model-en/` holds the English checkpoint this project was adapted from
(Inflect-Micro-v2 by Owen Song, Apache-2.0), vendored so that a clone speaks
both languages with nothing to download. Only the weights and their config are
kept: the upstream package also ships `inference.py`, `models.py` and `text/`,
modules with the *same names* as the ones in `model/`, and importing both in one
process gives whichever was imported first, silently -- so this module never
imports the English package's runtime.

Instead it loads only the weights, into the runtime `model/` already ships:

- the architecture is identical (409 of 410 tensors were carried over), so the
  same `SynthesizerTrn` accepts either checkpoint;
- the symbol inventory is the full 178-symbol IPA set shared by both, so English
  phonemes are already representable;
- `model/`'s frontend is pinned to `language: "az"` by `frontend.json`, which is
  why the text is phonemised here, with eSpeak's English voice, and handed over
  as `phonemes=` -- the one path the runtime offers for exactly this.

What this is not: the English model is a different voice, trained by someone
else. It shares nothing with the Azerbaijani voice but its shape.

The one Azerbaijani layer it does share is `censor_az`: a visitor can type an
Azerbaijani obscenity with the English voice selected, and eSpeak would voice
it. Censoring only removes words -- it writes no Azerbaijani -- so it is safe
here where `normalize_az` is not. `censor=False` turns it off.

    >>> from aztts import EnVoice          # doctest: +SKIP
    >>> EnVoice().save("Hello there.", "out/en.wav")   # doctest: +SKIP
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .az_chunk import DEFAULT_MAX_WORDS, chunk_text
from .az_profanity import BLEEP, bleep_segments, censor_az
from .engine import DEFAULT_MODEL_DIR, ModelNotFoundError, _load_package, bleep

__all__ = [
    "DEFAULT_EN_MODEL_DIR",
    "EN_LANGUAGE",
    "EnVoice",
    "checkpoint_status",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EN_MODEL_DIR = PROJECT_ROOT / "model-en"

# eSpeak's American English voice. The base model was trained on English audio
# phonemised this way.
EN_LANGUAGE = "en-us"

# What a Git LFS pointer file starts with. A clone made without Git LFS gets
# these few bytes instead of the weights, and the failure that follows is
# otherwise unreadable.
_LFS_POINTER = b"version https://git-lfs"
# Anything this small cannot be a 37 MB checkpoint.
_POINTER_LIMIT = 4096

_MISSING = (
    "The English weights are not in {path}.\n"
    "They ship with this repository; if the directory is gone, restore it with "
    "`git checkout model-en`, or fetch the full upstream package with "
    "`python training/scripts/download_model.py` and point EnVoice at "
    "training/base-model."
)
_POINTER = (
    "{path} is a Git LFS pointer, not the weights.\n"
    "The clone was made without Git LFS. Fix it with:\n"
    "    git lfs install && git lfs pull"
)


def checkpoint_status(model_dir: str | Path = DEFAULT_EN_MODEL_DIR) -> str:
    """"ok", "missing" or "pointer" -- what is actually in the directory.

    The pointer case is worth naming: `git clone` without Git LFS leaves a
    120-byte text file called `model.pth`, which looks present and fails deep
    inside torch.
    """
    path = Path(model_dir) / "model.pth"
    if not path.is_file():
        return "missing"
    try:
        if path.stat().st_size <= _POINTER_LIMIT:
            with path.open("rb") as handle:
                if handle.read(len(_LFS_POINTER)) == _LFS_POINTER:
                    return "pointer"
    except OSError:
        return "missing"
    return "ok"


def _configure_espeak() -> None:
    """Point phonemizer at an eSpeak NG library, as the model package does."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    for candidate in (
        Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib64/libespeak-ng.so.1"),
    ):
        if candidate.is_file():
            os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(candidate))
            return
    import espeakng_loader

    os.environ.setdefault(
        "PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path()
    )
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
    espeakng_loader.make_library_available()


class EnVoice:
    """Offline TTS engine that speaks English, using the base checkpoint.

    The object is expensive to build (the weights stay resident); build it once
    and reuse it, the same as `AzTTS`.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_EN_MODEL_DIR,
        *,
        runtime_dir: str | Path = DEFAULT_MODEL_DIR,
        device: str = "cpu",
    ) -> None:
        model_dir = Path(model_dir).resolve()
        status = checkpoint_status(model_dir)
        if status == "missing":
            raise ModelNotFoundError(_MISSING.format(path=model_dir))
        if status == "pointer":
            raise ModelNotFoundError(_POINTER.format(path=model_dir / "model.pth"))
        package = _load_package(Path(runtime_dir).resolve())
        self._package = package
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*weight_norm.*")
            self._tts = package.InflectTTS(str(model_dir), device=device)
        self.model_dir = model_dir
        self.device = device
        self.sample_rate: int = self._tts.sample_rate
        self._backend = None

    # -- internal helpers -------------------------------------------------

    def _phonemise(self, chunk: str) -> str:
        """English text -> IPA, with stress marks and punctuation kept."""
        if self._backend is None:
            _configure_espeak()
            from phonemizer.backend import EspeakBackend

            self._backend = EspeakBackend(
                language=EN_LANGUAGE,
                preserve_punctuation=True,
                with_stress=True,
                language_switch="remove-flags",
            )
        return self._backend.phonemize([chunk])[0]

    def _speak_chunk(self, chunk: str, **options) -> np.ndarray:
        """Speak a chunk, playing a bleep wherever `censor_az` left one."""
        pieces = [
            bleep(self.sample_rate)
            if segment == BLEEP
            else self._tts.synthesize(
                segment, phonemes=self._phonemise(segment), **options
            )[1]
            for segment in (bleep_segments(chunk) if BLEEP in chunk else (chunk,))
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    # -- public API -------------------------------------------------------

    def prepare(
        self,
        text: str,
        *,
        max_words: int = DEFAULT_MAX_WORDS,
        censor: bool = True,
    ) -> tuple[str, ...]:
        """The chunks that would reach the model, without synthesising.

        There is no normalisation step: `normalize_az` rewrites numbers into
        Azerbaijani words and would be wrong here. eSpeak reads English digits
        on its own. Obscenities are a `BLEEP` unless `censor=False`.
        """
        prepared = " ".join((censor_az(text) if censor else text).split())
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
        max_words: int = DEFAULT_MAX_WORDS,
        censor: bool = True,
    ) -> np.ndarray:
        """Synthesise English into a float32 mono array ([-1, 1], 24 kHz)."""
        chunks = self.prepare(text, max_words=max_words, censor=censor)
        if not chunks:
            raise ValueError("The text is empty.")
        pause = self._package.boundary_pause_seconds
        pieces: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index:
                silence = round(self.sample_rate * pause(chunks[index - 1]))
                pieces.append(np.zeros(silence, dtype=np.float32))
            pieces.append(
                self._speak_chunk(
                    chunk, speed=speed, variation=variation, seed=seed + index
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
