"""Import the v2 network code from a model package's runtime/ folder, read-only.

The runtime uses VITS' historical top-level module names (models, commons, modules, ...), which
clash with other packages. They are imported with the folder first on sys.path and removed from
sys.modules afterwards, so the caller keeps private module objects and nothing leaks.
"""
from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterator

NAMES = ("attentions", "commons", "inflect_alias_free", "models", "modules", "monotonic_align", "transforms", "utils")


@dataclass(frozen=True)
class Runtime:
    models: ModuleType
    commons: ModuleType
    modules: ModuleType
    root: Path


@contextmanager
def _isolated(root: Path) -> Iterator[None]:
    previous = {name: sys.modules.pop(name, None) for name in NAMES}
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in NAMES:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def load_runtime(root: Path) -> Runtime:
    root = Path(root).resolve()
    if not (root / "models.py").is_file():
        raise FileNotFoundError(f"{root} has no models.py; pass a model package's runtime/ folder")
    with _isolated(root):
        return Runtime(models=importlib.import_module("models"), commons=importlib.import_module("commons"),
                       modules=importlib.import_module("modules"), root=root)
