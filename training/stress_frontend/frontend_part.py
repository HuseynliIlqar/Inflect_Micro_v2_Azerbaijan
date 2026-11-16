

# ---------------------------------------------------------------------------------------------
# Frontend hook for the Inflect toolkit: normalize(), phonemize(), symbols(), metadata().
# The toolkit copies this one file into the model package, so the lexicon is embedded below.
# ---------------------------------------------------------------------------------------------
import base64 as _base64
import hashlib as _hashlib
import json as _json
import os as _os
import unicodedata as _unicodedata
import zlib as _zlib

FRONTEND_NAME = "az-espeak-stress"
FRONTEND_VERSION = "1.0"


def _embedded_lexicon() -> Lexicon:
    data = _json.loads(_zlib.decompress(_base64.b85decode(_LEXICON_BLOB)).decode("utf-8"))
    return Lexicon(verbs=frozenset(data["verbs"]), lemmas=frozenset(data["lemmas"]), forms=frozenset(data["forms"]))


def _toolkit_normalize(text: str) -> str:
    """Unicode and whitespace cleanup, the same as the toolkit applies to every transcript."""
    normalized = _unicodedata.normalize("NFKC", text)
    normalized = "".join(" " if character in "\r\n\t" else character for character in normalized)
    normalized = "".join(character for character in normalized if not _unicodedata.category(character).startswith("C"))
    return re.sub(r"\s+", " ", normalized).strip()


_PAUSE_MARKS = str.maketrans({"(": ",", ")": ","})


def _declared_only(phonemes: str) -> str:
    """Brackets become commas (a short pause); any other symbol the model does not know is dropped."""
    allowed = set(_BASE_SYMBOLS)
    kept = "".join(character for character in phonemes.translate(_PAUSE_MARKS) if character in allowed)
    kept = re.sub(r",(?:\s*,)+", ",", kept)
    return re.sub(r"\s+", " ", kept).strip()


class AzStressFrontend:
    """eSpeak NG (language az, through espeakng-loader) followed by the stress corrections above."""

    def __init__(self) -> None:
        import espeakng_loader

        # The bundled library, never a system one, so training and inference use the same eSpeak.
        _os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeakng_loader.get_library_path()
        _os.environ["ESPEAK_DATA_PATH"] = espeakng_loader.get_data_path()
        espeakng_loader.make_library_available()
        espeakng_loader.load_library()
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        self._backend = EspeakBackend("az", preserve_punctuation=True, with_stress=True, language_switch="remove-flags")
        self._separator = Separator(phone="", word=" ", syllable="")
        self._lexicon = _embedded_lexicon()

    def normalize(self, text: str) -> str:
        return _toolkit_normalize(text)

    def phonemize(self, normalized_text: str) -> str:
        raw = self._backend.phonemize([normalized_text], separator=self._separator, strip=True, njobs=1)[0]
        raw = re.sub(r"\s+", " ", _unicodedata.normalize("NFC", raw)).strip()
        return _declared_only(restress_sentence(normalized_text, raw, self._lexicon))

    def symbols(self) -> list[str]:
        # The base model's list has the apostrophe twice; the toolkit needs each symbol once.
        # Base indices are unaffected: the toolkit only appends symbols the base list lacks.
        return list(dict.fromkeys(_BASE_SYMBOLS))

    def metadata(self) -> dict:
        return {
            "name": FRONTEND_NAME,
            "version": FRONTEND_VERSION,
            "language": "az",
            "configuration": {
                "espeak_language": "az",
                # Kept static: the package re-hashes this at inference, so nothing machine-specific.
                "espeak_library": "espeakng-loader==0.2.4",
                "with_stress": True,
                "preserve_punctuation": True,
                "lexicon_sha256": _hashlib.sha256(_LEXICON_BLOB.encode("ascii")).hexdigest(),
            },
        }


def create_frontend(language: str = "az") -> AzStressFrontend:
    if language != "az":
        raise ValueError(f"This frontend is for Azerbaijani (az), not {language!r}.")
    return AzStressFrontend()
