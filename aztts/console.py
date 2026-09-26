"""Small helper that lets the console print Azerbaijani letters.

On Windows, Python's standard output defaults to `cp1252`, and printing a
letter such as `ə`, `ş` or `ğ` crashes the program with `UnicodeEncodeError`.
Every entry point calls `use_utf8()` as it starts up.
"""

from __future__ import annotations

import sys

__all__ = ["use_utf8"]


def use_utf8() -> None:
    """Switch stdout and stderr to UTF-8; carry on quietly if that fails."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            # Redirected or otherwise unusual stream -- printing still works.
            pass
