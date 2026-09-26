"""The browser interface -- Gradio wiring and nothing else.

    pip install -e ".[app]"      # or: pip install -r requirements-app.txt
    python app.py                # http://127.0.0.1:7860

Everything with logic in it lives in `webui/`, which is why this file has no
tests of its own: there is nothing here but widgets and the calls between them.

The model is loaded on the first request, not at import, so the page comes up
immediately and a machine without `model/` still shows a readable error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from aztts import DEFAULT_MAX_WORDS, ModelNotFoundError  # noqa: E402
from aztts.console import use_utf8  # noqa: E402
from say import DEMO, EN_DEMO  # noqa: E402  -- one source of truth for examples
from webui import i18n  # noqa: E402
from webui.runner import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_SPEED,
    DEFAULT_VARIATION,
    DEVICES,
    PROSODY_LEVELS,
    EmptyTextError,
    Settings,
    cli_command,
    english_model_present,
    english_model_status,
    random_seed,
    synthesise,
)

use_utf8()

EXAMPLES: list[list[str]] = [[text] for _, text in DEMO]

# The English weights ship with the repository, so this is normally true. It
# is false in a clone made without Git LFS, which is worth saying in the voice
# selector rather than at the moment someone presses Speak.
EN_AVAILABLE = english_model_present()

# The English examples mirror the Azerbaijani ones. They are shown only when
# the English voice is selected; `say.py` keeps the same pair.
EN_EXAMPLES: list[list[str]] = [[text] for _, text in EN_DEMO]


def _prosody_choices(language: str) -> list[tuple[str, str]]:
    """Localised labels over the stable `off`/`safe`/`wide` values."""
    return [(i18n.label(language, f"prosody_{level}"), level) for level in PROSODY_LEVELS]


def _heading(language: str) -> str:
    return (
        f"# {i18n.label(language, 'title')}\n\n"
        f"{i18n.label(language, 'subtitle')}"
    )


def _stats_line(language: str, result) -> str:
    """The performance readout, plus what the cleanup actually did."""
    line = i18n.label(
        language,
        "stats_template",
        seconds=result.seconds,
        elapsed=result.elapsed,
        realtime=result.realtime,
        chunks=i18n.chunk_count(language, len(result.chunks)),
        rate=result.sample_rate,
    )
    if result.notches is None:
        return line
    if not result.notches:
        return line + i18n.label(language, "stats_cleaned_none")
    notches = ", ".join(f"{hertz:.0f} Hz" for hertz in result.notches)
    return line + i18n.label(language, "stats_cleaned", notches=notches)


def speak(
    text: str,
    language: str,
    voice: str,
    speed: float,
    variation: float,
    seed: float,
    normalise: bool,
    max_words: float,
    prosody: str,
    prosody_drop: bool,
    cleanup: bool,
    device: str,
):
    """Run one synthesis and fill the four output widgets."""
    try:
        settings = Settings(
            speed=float(speed),
            variation=float(variation),
            seed=int(seed),
            normalize=bool(normalise),
            max_words=int(max_words),
            prosody=prosody,
            prosody_drop=bool(prosody_drop),
            cleanup=bool(cleanup),
            device=device,
            voice=voice,
        )
    except ValueError as error:
        return None, "", i18n.label(language, "error_failed", error=error), "", None

    command = cli_command(text, settings)
    try:
        result = synthesise(text, settings)
    except EmptyTextError:
        return None, "", i18n.label(language, "error_empty"), command, None
    except ModelNotFoundError:
        if settings.voice != "en":
            key = "error_model"
        else:
            # A clone without Git LFS has the file but not the weights; saying
            # "missing" there sends the reader looking for the wrong problem.
            key = (
                "voice_en_pointer"
                if english_model_status() == "pointer"
                else "voice_en_missing"
            )
        return None, "", i18n.label(language, key), command, None
    except (OSError, RuntimeError, ValueError) as error:
        return (
            None, "", i18n.label(language, "error_failed", error=error), command, None
        )

    numbered = "\n".join(
        f"[{index}] {chunk}" for index, chunk in enumerate(result.chunks, start=1)
    )
    # The result is kept so the statistics can be reprinted in the other
    # language without synthesising again.
    return str(result.path), numbered, _stats_line(language, result), command, result


def build() -> gr.Blocks:
    """Assemble the page. The language selector relabels it in place."""
    start = i18n.DEFAULT_LANGUAGE

    with gr.Blocks(title=i18n.label(start, "title"), fill_width=False) as demo:
        with gr.Row():
            heading = gr.Markdown(_heading(start))
            language = gr.Radio(
                choices=[(name, code) for code, name in i18n.LANGUAGE_NAMES.items()],
                value=start,
                label=i18n.label(start, "language_label"),
                scale=0,
                min_width=200,
            )

        with gr.Row():
            with gr.Column(scale=3):
                voice = gr.Radio(
                    choices=i18n.voice_choices(start, EN_AVAILABLE),
                    value="az",
                    label=i18n.label(start, "voice_label"),
                    info=i18n.label(start, "voice_info"),
                )
                text = gr.Textbox(
                    label=i18n.label(start, "text_label"),
                    placeholder=i18n.label(start, "text_placeholder"),
                    lines=4,
                    autofocus=True,
                )
                speak_button = gr.Button(
                    i18n.label(start, "speak"), variant="primary"
                )
                # `gr.Examples` builds its own label at construction time and
                # cannot be relabelled later, so the heading is ours.
                examples_heading = gr.Markdown(
                    f"**{i18n.label(start, 'examples_label')}**"
                )
                examples = gr.Examples(examples=EXAMPLES, inputs=[text], label="")
                audio = gr.Audio(
                    label=i18n.label(start, "audio_label"),
                    type="filepath",
                    autoplay=False,
                    show_download_button=True,
                )
                stats = gr.Markdown("")

            with gr.Column(scale=2):
                with gr.Group():
                    parameters_heading = gr.Markdown(
                        f"### {i18n.label(start, 'parameters')}"
                    )
                    speed = gr.Slider(
                        0.5, 2.0, value=DEFAULT_SPEED, step=0.05,
                        label=i18n.label(start, "speed_label"),
                        info=i18n.label(start, "speed_info"),
                    )
                    variation = gr.Slider(
                        0.0, 1.0, value=DEFAULT_VARIATION, step=0.001,
                        label=i18n.label(start, "variation_label"),
                        info=i18n.label(start, "variation_info"),
                    )
                    seed = gr.Number(
                        value=DEFAULT_SEED, precision=0,
                        label=i18n.label(start, "seed_label"),
                        info=i18n.label(start, "seed_info"),
                    )
                    shuffle = gr.Button(
                        i18n.label(start, "random_seed"), size="sm"
                    )
                    normalise = gr.Checkbox(
                        value=True,
                        label=i18n.label(start, "normalise_label"),
                        info=i18n.label(start, "normalise_info"),
                    )
                    cleanup = gr.Checkbox(
                        value=False,
                        label=i18n.label(start, "cleanup_label"),
                        info=i18n.label(start, "cleanup_info"),
                    )

                with gr.Accordion(i18n.label(start, "advanced"), open=False) as advanced:
                    max_words = gr.Slider(
                        0, 40, value=DEFAULT_MAX_WORDS, step=1,
                        label=i18n.label(start, "max_words_label"),
                        info=i18n.label(start, "max_words_info"),
                    )
                    prosody = gr.Radio(
                        choices=_prosody_choices(start),
                        value="off",
                        label=i18n.label(start, "prosody_label"),
                        info=i18n.label(start, "prosody_info"),
                    )
                    prosody_drop = gr.Checkbox(
                        value=False,
                        label=i18n.label(start, "prosody_drop_label"),
                        info=i18n.label(start, "prosody_drop_info"),
                    )
                    device = gr.Radio(
                        choices=list(DEVICES),
                        value="cpu",
                        label=i18n.label(start, "device_label"),
                    )

                chunks = gr.Textbox(
                    label=i18n.label(start, "chunks_label"),
                    info=i18n.label(start, "chunks_info"),
                    lines=4,
                    interactive=False,
                )
                command = gr.Code(
                    label=i18n.label(start, "cli_label"), language="shell"
                )

        footer = gr.Markdown(f"_{i18n.label(start, 'footer')}_")

        last_result = gr.State(None)

        inputs = [
            text, language, voice, speed, variation, seed, normalise,
            max_words, prosody, prosody_drop, cleanup, device,
        ]
        outputs = [audio, chunks, stats, command, last_result]
        speak_button.click(speak, inputs=inputs, outputs=outputs)
        text.submit(speak, inputs=inputs, outputs=outputs)
        shuffle.click(lambda: random_seed(), outputs=[seed])

        # -- voice switching ------------------------------------------------
        # The Azerbaijani text layers do not apply to the English checkpoint,
        # so their controls are greyed out rather than quietly ignored.
        def switch_voice(selected: str, selected_language: str):
            azerbaijani = selected == "az"
            suffix = "" if azerbaijani else (
                " " + i18n.label(selected_language, "az_only")
            )
            return [
                gr.update(
                    interactive=azerbaijani,
                    info=i18n.label(selected_language, "normalise_info") + suffix,
                ),
                gr.update(
                    interactive=azerbaijani,
                    info=i18n.label(selected_language, "prosody_info") + suffix,
                ),
                gr.update(interactive=azerbaijani),
                gr.update(samples=EXAMPLES if azerbaijani else EN_EXAMPLES),
            ]

        voice.change(
            switch_voice,
            inputs=[voice, language],
            outputs=[normalise, prosody, prosody_drop, examples.dataset],
        )

        # -- language switching ---------------------------------------------
        # Every widget that carries text is relabelled; the values stay put, so
        # switching language mid-session loses nothing the user typed.
        relabelled = [
            heading, language, voice, text, speak_button, examples_heading, audio,
            parameters_heading, speed, variation, seed, shuffle, normalise,
            cleanup, advanced, max_words, prosody, prosody_drop, device,
            chunks, command, footer, stats,
        ]

        def switch(selected: str, result):
            get = lambda key, **kw: i18n.label(selected, key, **kw)  # noqa: E731
            return [
                gr.update(value=_heading(selected)),
                gr.update(label=get("language_label")),
                gr.update(
                    choices=i18n.voice_choices(selected, EN_AVAILABLE),
                    label=get("voice_label"),
                    info=get("voice_info"),
                ),
                gr.update(label=get("text_label"), placeholder=get("text_placeholder")),
                gr.update(value=get("speak")),
                gr.update(value=f"**{get('examples_label')}**"),
                gr.update(label=get("audio_label")),
                gr.update(value=f"### {get('parameters')}"),
                gr.update(label=get("speed_label"), info=get("speed_info")),
                gr.update(label=get("variation_label"), info=get("variation_info")),
                gr.update(label=get("seed_label"), info=get("seed_info")),
                gr.update(value=get("random_seed")),
                gr.update(label=get("normalise_label"), info=get("normalise_info")),
                gr.update(label=get("cleanup_label"), info=get("cleanup_info")),
                gr.update(label=get("advanced")),
                gr.update(label=get("max_words_label"), info=get("max_words_info")),
                gr.update(
                    choices=_prosody_choices(selected),
                    label=get("prosody_label"),
                    info=get("prosody_info"),
                ),
                gr.update(
                    label=get("prosody_drop_label"), info=get("prosody_drop_info")
                ),
                gr.update(label=get("device_label")),
                gr.update(label=get("chunks_label"), info=get("chunks_info")),
                gr.update(label=get("cli_label")),
                gr.update(value=f"_{get('footer')}_"),
                # A finished clip's statistics follow the new language too.
                gr.update(value=_stats_line(selected, result))
                if result is not None
                else gr.update(),
            ]

        language.change(switch, inputs=[language, last_result], outputs=relabelled)

    return demo


if __name__ == "__main__":
    build().launch()
