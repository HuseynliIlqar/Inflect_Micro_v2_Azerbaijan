from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from text.symbols import symbols


class DeploymentFrontendError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrontendOutput:
    raw_text: str
    normalized_text: str
    phoneme_text: str


def _load_contract() -> dict[str, Any]:
    path = PACKAGE_ROOT / "frontend.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentFrontendError(f"Could not load frontend contract: {exc}") from exc
    if contract.get("format") != "inflect_deployment_frontend_v1":
        raise DeploymentFrontendError("Unsupported or missing deployment frontend contract.")
    if contract.get("mode") not in {"espeak", "prephonemized", "custom"}:
        raise DeploymentFrontendError("Deployment frontend mode is invalid.")
    if not isinstance(contract.get("language"), str) or not contract["language"].strip():
        raise DeploymentFrontendError("Deployment frontend language is missing.")
    return contract


CONTRACT = _load_contract()
SYMBOLS = frozenset(symbols)
_ESPEAK_BACKEND: Any = None
_CUSTOM_FRONTEND: Any = None


def _normalize_generic(text: str) -> str:
    if not isinstance(text, str):
        raise DeploymentFrontendError("Text input must be a Unicode string.")
    if "\x00" in text:
        raise DeploymentFrontendError("Text input contains a null byte.")
    value = unicodedata.normalize("NFKC", text)
    value = "".join(" " if char in "\r\n\t" else char for char in value)
    value = "".join(
        char for char in value if not unicodedata.category(char).startswith("C")
    )
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise DeploymentFrontendError("Text input is empty after normalization.")
    return value


def _clean_phonemes(value: str) -> str:
    if not isinstance(value, str):
        raise DeploymentFrontendError("Phoneme input must be a Unicode string.")
    value = unicodedata.normalize("NFC", value)
    if "\x00" in value or any(char in "\r\n" for char in value):
        raise DeploymentFrontendError("Phoneme input contains unsupported controls.")
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise DeploymentFrontendError("Phoneme input is empty.")
    unknown = sorted(set(value).difference(SYMBOLS))
    if unknown:
        rendered = ", ".join(repr(char) for char in unknown[:16])
        raise DeploymentFrontendError(
            "Phoneme input contains symbols absent from this checkpoint: " + rendered
        )
    return value


def _configure_espeak() -> None:
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1"),
        Path("/usr/lib64/libespeak-ng.so.1"),
    )
    system = next((path for path in candidates if path.is_file()), None)
    if system is not None:
        os.environ.setdefault("PHONEMIZER_ESPEAK_LIBRARY", str(system))
        return
    try:
        import espeakng_loader

        os.environ.setdefault(
            "PHONEMIZER_ESPEAK_LIBRARY", espeakng_loader.get_library_path()
        )
        os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
        espeakng_loader.make_library_available()
        espeakng_loader.load_library()
    except (ImportError, OSError, RuntimeError) as exc:
        raise DeploymentFrontendError(
            "Could not initialize eSpeak NG. Install espeakng-loader or a system "
            "eSpeak NG library."
        ) from exc


def _espeak() -> Any:
    global _ESPEAK_BACKEND
    if _ESPEAK_BACKEND is not None:
        return _ESPEAK_BACKEND
    _configure_espeak()
    try:
        from phonemizer.backend import EspeakBackend

        _ESPEAK_BACKEND = EspeakBackend(
            language=CONTRACT["language"],
            preserve_punctuation=bool(CONTRACT["preserve_punctuation"]),
            with_stress=bool(CONTRACT["with_stress"]),
            language_switch="remove-flags",
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise DeploymentFrontendError(
            f"Could not create eSpeak frontend for {CONTRACT['language']!r}: {exc}"
        ) from exc
    return _ESPEAK_BACKEND


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _custom() -> Any:
    global _CUSTOM_FRONTEND
    if _CUSTOM_FRONTEND is not None:
        return _CUSTOM_FRONTEND
    record = CONTRACT.get("custom_hook")
    if not isinstance(record, dict):
        raise DeploymentFrontendError("Custom frontend package metadata is missing.")
    hook_path = (PACKAGE_ROOT / str(record.get("path", ""))).resolve()
    try:
        hook_path.relative_to(PACKAGE_ROOT)
    except ValueError as exc:
        raise DeploymentFrontendError("Custom hook path escapes the package.") from exc
    if hook_path.suffix.lower() != ".py" or not hook_path.is_file():
        raise DeploymentFrontendError(f"Packaged custom hook is missing: {hook_path.name}")
    digest = hashlib.sha256(hook_path.read_bytes()).hexdigest()
    if digest != record.get("source_sha256"):
        raise DeploymentFrontendError("Packaged custom hook hash verification failed.")
    spec = importlib.util.spec_from_file_location("_inflect_package_frontend", hook_path)
    if spec is None or spec.loader is None:
        raise DeploymentFrontendError("Could not load packaged custom frontend.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, str(record.get("factory", "")), None)
    if not callable(factory):
        raise DeploymentFrontendError("Packaged custom frontend factory is unavailable.")
    try:
        signature = inspect.signature(factory)
        try:
            signature.bind(language=CONTRACT["language"])
            implementation = factory(language=CONTRACT["language"])
        except TypeError:
            signature.bind()
            implementation = factory()
    except Exception as exc:
        raise DeploymentFrontendError(f"Custom frontend factory failed: {exc}") from exc
    for method in ("normalize", "phonemize", "metadata"):
        if not callable(getattr(implementation, method, None)):
            raise DeploymentFrontendError(
                f"Custom frontend does not provide callable {method}()."
            )
    metadata = implementation.metadata()
    if _canonical_hash(metadata) != record.get("metadata_sha256"):
        raise DeploymentFrontendError("Custom frontend metadata hash verification failed.")
    _CUSTOM_FRONTEND = implementation
    return implementation


def process_input(
    text: str | None = None,
    *,
    phonemes: str | None = None,
) -> FrontendOutput:
    raw_text = text or ""
    if phonemes is not None:
        normalized = _normalize_generic(text) if text and text.strip() else ""
        return FrontendOutput(raw_text, normalized, _clean_phonemes(phonemes))
    mode = CONTRACT["mode"]
    if mode == "prephonemized":
        raise DeploymentFrontendError(
            "This checkpoint uses a prephonemized frontend. Supply phonemes=... "
            "or use inference.py --phonemes."
        )
    if text is None:
        raise DeploymentFrontendError("Text input is required.")
    if mode == "custom":
        implementation = _custom()
        try:
            normalized = _normalize_generic(implementation.normalize(text))
            phoneme_text = implementation.phonemize(normalized)
        except Exception as exc:
            raise DeploymentFrontendError(f"Custom frontend failed: {exc}") from exc
    else:
        normalized = _normalize_generic(text)
        try:
            from phonemizer.separator import Separator

            phoneme_text = _espeak().phonemize(
                [normalized],
                separator=Separator(phone="", word=" ", syllable=""),
                strip=True,
                njobs=1,
            )[0]
        except (RuntimeError, ValueError, OSError) as exc:
            raise DeploymentFrontendError(
                f"eSpeak failed for language {CONTRACT['language']!r}: {exc}"
            ) from exc
    return FrontendOutput(raw_text, normalized, _clean_phonemes(phoneme_text))
