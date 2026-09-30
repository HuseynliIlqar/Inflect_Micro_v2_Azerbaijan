"""Everything the interface does, minus the widgets.

Keeping it here rather than in `app.py` has one concrete payoff: the settings,
the CLI equivalent and the synthesis flow can all be tested against a stand-in
engine, so `tests/test_webui_runner.py` never loads the 37 MB model and the
suite stays at three seconds.

    >>> settings = Settings(speed=0.9, seed=42)
    >>> cli_command("Salam.", settings)
    'python say.py --speed 0.9 --seed 42 "Salam."'
"""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np

from aztts import (
    DEFAULT_EN_MODEL_DIR,
    DEFAULT_MAX_WORDS,
    DEFAULT_MODEL_DIR,
    AzTTS,
    EnVoice,
)

from aztts.en_voice import checkpoint_status

from .cleanup import autoclean

__all__ = [
    "DEVICES",
    "EmptyTextError",
    "PROSODY_LEVELS",
    "VOICES",
    "english_model_present",
    "english_model_status",
    "Result",
    "Settings",
    "cli_command",
    "load_engine",
    "random_seed",
    "synthesise",
]

PROSODY_LEVELS: tuple[str, ...] = ("off", "safe", "wide")
# "az" is this project's model. "en" is the English base model it was adapted
# from: another author's voice, downloaded separately, and none of the
# Azerbaijani text layers apply to it.
VOICES: tuple[str, ...] = ("az", "en")
DEVICES: tuple[str, ...] = ("cpu", "cuda")
SEED_LIMIT = 2**31 - 1

# The CLI's own defaults, so the two front ends cannot drift apart.
DEFAULT_SPEED = 1.0
DEFAULT_VARIATION = 0.667
DEFAULT_SEED = 7


class EmptyTextError(ValueError):
    """There is nothing left to speak once the text has been prepared."""


class Engine(Protocol):
    """The part of `AzTTS` the interface uses."""

    sample_rate: int

    def prepare(
        self,
        text: str,
        *,
        normalize: bool = ...,
        max_words: int = ...,
        censor: bool = ...,
    ) -> tuple[str, ...]: ...

    def synthesize(self, text: str, **options: object) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class Settings:
    """One synthesis request's knobs. Frozen: `replace()` gives a new one."""

    speed: float = DEFAULT_SPEED
    variation: float = DEFAULT_VARIATION
    seed: int = DEFAULT_SEED
    normalize: bool = True
    max_words: int = DEFAULT_MAX_WORDS
    prosody: str = "off"
    prosody_drop: bool = False
    cleanup: bool = False
    device: str = "cpu"
    voice: str = "az"
    # Not a widget: the interface bleeps unless `app.py --allow-profanity`
    # was launched on purpose, so a hosted Space can never turn it off.
    censor: bool = True

    def __post_init__(self) -> None:
        # The interface is not the only caller, so the bounds are checked here
        # rather than trusted from the sliders.
        if not 0.1 <= self.speed <= 3.0:
            raise ValueError(f"speed out of range: {self.speed}")
        if not 0.0 <= self.variation <= 1.0:
            raise ValueError(f"variation out of range: {self.variation}")
        if self.max_words < 0:
            raise ValueError(f"max_words cannot be negative: {self.max_words}")
        if self.prosody not in PROSODY_LEVELS:
            raise ValueError(f"unknown prosody level: {self.prosody}")
        if self.device not in DEVICES:
            raise ValueError(f"unknown device: {self.device}")
        if self.voice not in VOICES:
            raise ValueError(f"unknown voice: {self.voice}")

    @property
    def synthesis_options(self) -> dict[str, object]:
        """The keyword arguments the chosen engine takes.

        `EnVoice` has no normalisation or stress layer: `normalize_az` rewrites
        numbers into Azerbaijani words and `restress` is tuned to Azerbaijani
        stress, so neither is offered for English.
        """
        options: dict[str, object] = {
            "speed": self.speed,
            "variation": self.variation,
            "seed": self.seed,
            "max_words": self.max_words,
            # Both voices: Azerbaijani obscenities can be typed into either.
            "censor": self.censor,
        }
        if self.voice == "az":
            options.update(
                normalize=self.normalize,
                prosody=self.prosody,
                prosody_drop=self.prosody_drop,
            )
        return options


@dataclass(frozen=True, slots=True)
class Result:
    """A finished clip and what it cost to make."""

    path: Path
    chunks: tuple[str, ...]
    seconds: float
    elapsed: float
    sample_rate: int
    notches: tuple[float, ...] | None = None  # None when cleanup was off

    @property
    def realtime(self) -> float:
        """How many times faster than real time the synthesis ran."""
        return self.seconds / self.elapsed if self.elapsed > 0 else 0.0


def random_seed(rng: np.random.Generator | None = None) -> int:
    """A fresh seed for the "random seed" button."""
    generator = rng if rng is not None else np.random.default_rng()
    return int(generator.integers(0, SEED_LIMIT))


def _quote(text: str) -> str:
    """Double quotes work in bash, cmd and PowerShell alike."""
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"') + '"'


def cli_command(text: str, settings: Settings) -> str:
    """The `say.py` invocation matching these settings.

    Only what differs from the CLI's defaults is printed, so the line stays
    short enough to read and to paste.
    """
    parts = ["python", "say.py"]
    if settings.voice != "az":
        parts += ["--voice", settings.voice]
    if settings.speed != DEFAULT_SPEED:
        parts += ["--speed", f"{settings.speed:g}"]
    if settings.variation != DEFAULT_VARIATION:
        parts += ["--variation", f"{settings.variation:g}"]
    if settings.seed != DEFAULT_SEED:
        parts += ["--seed", str(settings.seed)]
    if not settings.normalize and settings.voice == "az":
        parts.append("--raw")
    if settings.max_words != DEFAULT_MAX_WORDS:
        parts += ["--max-words", str(settings.max_words)]
    if settings.prosody != "off" and settings.voice == "az":
        parts += ["--prosody", settings.prosody]
        if settings.prosody_drop:
            parts.append("--prosody-drop")
    if settings.device != "cpu":
        parts += ["--device", settings.device]
    if not settings.censor:
        parts.append("--allow-profanity")
    parts.append(_quote(" ".join(text.split())))
    line = " ".join(parts)
    if settings.cleanup:
        # There is no CLI flag for it; the batch tool is the equivalent.
        line += "\npython tools/audio_postprocess.py analyze out"
    return line


@lru_cache(maxsize=2)
def load_engine(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str = "cpu") -> AzTTS:
    """Build an engine once per (directory, device) and keep it resident.

    Loading costs a couple of seconds and the weights stay in memory, so the
    interface must not do it per request.
    """
    return AzTTS(model_dir, device=device)


@lru_cache(maxsize=2)
def load_en_engine(device: str = "cpu") -> EnVoice:
    """The English base model, cached the same way."""
    return EnVoice(device=device)


def english_model_status(model_dir: Path = DEFAULT_EN_MODEL_DIR) -> str:
    """"ok", "missing" or "pointer" -- see `aztts.en_voice.checkpoint_status`.

    The English weights ship with the repository, so "ok" is the normal answer.
    "pointer" means the clone was made without Git LFS, which is the one way a
    fresh clone can still end up without a usable English voice.
    """
    return checkpoint_status(model_dir)


def english_model_present(model_dir: Path = DEFAULT_EN_MODEL_DIR) -> bool:
    """Whether the English voice can actually be spoken."""
    return english_model_status(model_dir) == "ok"


@lru_cache(maxsize=1)
def _output_dir() -> Path:
    """A per-process scratch directory for the rendered clips."""
    return Path(tempfile.mkdtemp(prefix="aztts-ui-"))


def synthesise(
    text: str,
    settings: Settings,
    *,
    engine: Engine | None = None,
    out_dir: Path | None = None,
) -> Result:
    """Render the text to a WAV file and report what happened.

    `engine` is injectable so the tests can run without the model; leave it out
    and the cached real engine is used.
    """
    import soundfile as sf

    if not text.strip():
        raise EmptyTextError("the text is empty")
    if engine is not None:
        worker = engine
    elif settings.voice == "en":
        worker = load_en_engine(settings.device)
    else:
        worker = load_engine(DEFAULT_MODEL_DIR, settings.device)

    if settings.voice == "en":
        chunks = worker.prepare(
            text, max_words=settings.max_words, censor=settings.censor
        )
    else:
        chunks = worker.prepare(
            text,
            normalize=settings.normalize,
            max_words=settings.max_words,
            censor=settings.censor,
        )
    if not chunks:
        raise EmptyTextError("the text is empty once prepared")

    began = time.perf_counter()
    waveform = worker.synthesize(text, **settings.synthesis_options)
    elapsed = time.perf_counter() - began

    notches: tuple[float, ...] | None = None
    if settings.cleanup:
        waveform, notches = autoclean(waveform, worker.sample_rate)

    directory = out_dir if out_dir is not None else _output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex[:12]}.wav"
    sf.write(destination, waveform, worker.sample_rate)

    return Result(
        path=destination,
        chunks=tuple(chunks),
        seconds=waveform.size / worker.sample_rate,
        elapsed=max(elapsed, 1e-9),
        sample_rate=worker.sample_rate,
        notches=notches,
    )


def with_seed(settings: Settings, seed: int) -> Settings:
    """A copy carrying a different seed -- nothing is mutated in place."""
    return replace(settings, seed=seed)
