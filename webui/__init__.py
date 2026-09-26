"""The browser interface around `aztts`.

    pip install -e ".[app]"
    python app.py

`app.py` is the Gradio wiring and nothing else. Everything it needs that can be
tested without loading the model lives here:

- `i18n`    -- the Azerbaijani and English label tables;
- `runner`  -- settings, the CLI equivalent of a set of settings, synthesis;
- `cleanup` -- the optional resonance filter applied to a finished clip.
"""

from __future__ import annotations

from .cleanup import autoclean, find_resonances
from .i18n import DEFAULT_LANGUAGE, LANGUAGES, label
from .runner import Result, Settings, cli_command, load_engine, synthesise

__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "Result",
    "Settings",
    "autoclean",
    "cli_command",
    "find_resonances",
    "label",
    "load_engine",
    "synthesise",
]
