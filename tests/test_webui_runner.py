"""The interface's logic, driven by a stand-in engine.

No test here loads the model: `FakeEngine` returns a sine wave, which is enough
to exercise every path the widgets can reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from aztts import DEFAULT_MAX_WORDS
from webui.runner import (
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_VARIATION,
    EmptyTextError,
    Result,
    Settings,
    cli_command,
    english_model_present,
    english_model_status,
    random_seed,
    synthesise,
    with_seed,
)

RATE = 24_000


class FakeEngine:
    """Records the options it was called with and returns a second of tone."""

    sample_rate = RATE

    def __init__(self, chunks: tuple[str, ...] = ("Salam.",)) -> None:
        self._chunks = chunks
        self.options: dict[str, object] = {}
        self.prepared: dict[str, object] = {}

    def prepare(self, text, *, normalize=True, max_words=DEFAULT_MAX_WORDS):
        self.prepared = {"text": text, "normalize": normalize, "max_words": max_words}
        return self._chunks

    def synthesize(self, text, **options):
        self.options = options
        time = np.arange(RATE, dtype=np.float32) / RATE
        return (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)


# -- settings ---------------------------------------------------------------


def test_the_defaults_match_the_cli() -> None:
    settings = Settings()
    assert (settings.speed, settings.variation, settings.seed) == (
        DEFAULT_SPEED,
        DEFAULT_VARIATION,
        DEFAULT_SEED,
    )
    assert settings.max_words == DEFAULT_MAX_WORDS
    assert settings.normalize is True
    assert settings.prosody == "off"


@pytest.mark.parametrize(
    "kwargs",
    (
        {"speed": 0.0},
        {"speed": 5.0},
        {"variation": -0.1},
        {"variation": 1.5},
        {"max_words": -1},
        {"prosody": "loud"},
        {"device": "tpu"},
    ),
)
def test_bad_settings_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(Exception):
        settings.speed = 2.0  # type: ignore[misc]


def test_with_seed_returns_a_new_object() -> None:
    original = Settings(seed=7)
    changed = with_seed(original, 99)
    assert original.seed == 7
    assert changed.seed == 99
    assert changed is not original


def test_synthesis_options_cover_every_engine_keyword() -> None:
    options = Settings().synthesis_options
    assert set(options) == {
        "speed", "variation", "seed", "normalize",
        "max_words", "prosody", "prosody_drop",
    }
    # `cleanup` and `device` are ours, not the engine's.
    assert "cleanup" not in options and "device" not in options


def test_random_seed_stays_in_range() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert 0 <= random_seed(rng) < 2**31


# -- the command line equivalent --------------------------------------------


def test_default_settings_produce_the_shortest_command() -> None:
    assert cli_command("Salam.", Settings()) == 'python say.py "Salam."'


def test_changed_settings_appear_as_flags() -> None:
    command = cli_command(
        "Salam.",
        Settings(speed=0.9, variation=0.5, seed=42, normalize=False, max_words=8),
    )
    assert "--speed 0.9" in command
    assert "--variation 0.5" in command
    assert "--seed 42" in command
    assert "--raw" in command
    assert "--max-words 8" in command


def test_prosody_drop_is_only_printed_with_prosody() -> None:
    assert "--prosody-drop" not in cli_command("A.", Settings(prosody_drop=True))
    with_prosody = cli_command("A.", Settings(prosody="safe", prosody_drop=True))
    assert "--prosody safe" in with_prosody and "--prosody-drop" in with_prosody


def test_the_text_is_quoted_and_collapsed() -> None:
    command = cli_command("  Salam,\n  necəsiniz?  ", Settings())
    assert command.endswith('"Salam, necəsiniz?"')


def test_quotes_inside_the_text_are_escaped() -> None:
    command = cli_command('O dedi: "salam"', Settings())
    assert command.count('\\"') == 2


def test_cleanup_adds_the_batch_tool_line() -> None:
    assert "audio_postprocess" in cli_command("A.", Settings(cleanup=True))


# -- synthesis ---------------------------------------------------------------


def test_empty_text_is_refused(tmp_path) -> None:
    with pytest.raises(EmptyTextError):
        synthesise("   ", Settings(), engine=FakeEngine(), out_dir=tmp_path)


def test_text_that_prepares_to_nothing_is_refused(tmp_path) -> None:
    with pytest.raises(EmptyTextError):
        synthesise("...", Settings(), engine=FakeEngine(chunks=()), out_dir=tmp_path)


def test_a_wav_file_is_written(tmp_path) -> None:
    result = synthesise("Salam.", Settings(), engine=FakeEngine(), out_dir=tmp_path)
    assert isinstance(result, Result)
    assert result.path.exists() and result.path.suffix == ".wav"
    assert result.path.parent == tmp_path
    assert result.sample_rate == RATE
    assert result.seconds == pytest.approx(1.0, abs=0.01)
    assert result.realtime > 0


def test_every_call_writes_its_own_file(tmp_path) -> None:
    first = synthesise("Salam.", Settings(), engine=FakeEngine(), out_dir=tmp_path)
    second = synthesise("Salam.", Settings(), engine=FakeEngine(), out_dir=tmp_path)
    assert first.path != second.path


def test_the_settings_reach_the_engine(tmp_path) -> None:
    engine = FakeEngine()
    settings = Settings(speed=1.2, variation=0.4, seed=3, prosody="wide", max_words=9)
    synthesise("Salam.", settings, engine=engine, out_dir=tmp_path)
    assert engine.options == settings.synthesis_options
    assert engine.prepared["normalize"] is True
    assert engine.prepared["max_words"] == 9


def test_the_chunks_are_reported(tmp_path) -> None:
    engine = FakeEngine(chunks=("Bir.", "İki."))
    result = synthesise("Bir. İki.", Settings(), engine=engine, out_dir=tmp_path)
    assert result.chunks == ("Bir.", "İki.")


def test_cleanup_off_reports_no_notches(tmp_path) -> None:
    result = synthesise("Salam.", Settings(), engine=FakeEngine(), out_dir=tmp_path)
    assert result.notches is None


def test_cleanup_on_reports_a_tuple(tmp_path) -> None:
    result = synthesise(
        "Salam.", Settings(cleanup=True), engine=FakeEngine(), out_dir=tmp_path
    )
    assert isinstance(result.notches, tuple)


# -- the English voice -------------------------------------------------------


class FakeEnEngine(FakeEngine):
    """`EnVoice` has no `normalize` keyword -- its absence is the point."""

    def prepare(self, text, *, max_words=DEFAULT_MAX_WORDS):
        self.prepared = {"text": text, "max_words": max_words}
        return self._chunks


def test_the_default_voice_is_azerbaijani() -> None:
    assert Settings().voice == "az"


def test_an_unknown_voice_is_refused() -> None:
    with pytest.raises(ValueError):
        Settings(voice="de")


def test_english_drops_the_azerbaijani_only_options() -> None:
    options = Settings(voice="en", normalize=False, prosody="wide").synthesis_options
    assert set(options) == {"speed", "variation", "seed", "max_words"}


def test_azerbaijani_keeps_them() -> None:
    assert "normalize" in Settings(voice="az").synthesis_options


def test_the_english_engine_is_called_without_normalise(tmp_path) -> None:
    engine = FakeEnEngine()
    result = synthesise(
        "Hello there.", Settings(voice="en", max_words=9),
        engine=engine, out_dir=tmp_path,
    )
    assert engine.prepared == {"text": "Hello there.", "max_words": 9}
    assert "normalize" not in engine.options
    assert result.path.exists()


def test_the_english_command_names_the_voice() -> None:
    assert cli_command("Hello.", Settings(voice="en")) == (
        'python say.py --voice en "Hello."'
    )


def test_the_english_command_omits_azerbaijani_flags() -> None:
    command = cli_command(
        "Hello.", Settings(voice="en", normalize=False, prosody="safe", speed=1.2)
    )
    assert "--raw" not in command
    assert "--prosody" not in command
    assert "--speed 1.2" in command


def test_english_availability_is_reported(tmp_path) -> None:
    assert english_model_present(tmp_path) is False
    assert english_model_status(tmp_path) == "missing"
    (tmp_path / "model.pth").write_bytes(b"x" * 8192)
    assert english_model_present(tmp_path) is True
    assert english_model_status(tmp_path) == "ok"


def test_a_clone_without_git_lfs_is_not_called_present(tmp_path) -> None:
    (tmp_path / "model.pth").write_bytes(
        b"version https://git-lfs.github.com/spec/v1\nsize 37529995\n"
    )
    assert english_model_status(tmp_path) == "pointer"
    assert english_model_present(tmp_path) is False
