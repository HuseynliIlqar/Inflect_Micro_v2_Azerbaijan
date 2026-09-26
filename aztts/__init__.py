"""Offline Azerbaijani text-to-speech.

    from aztts import AzTTS

    tts = AzTTS()
    tts.save("Salam, necəsiniz?", "out/salam.wav")

The text normalisation and chunking helpers can also be used on their own:

    from aztts import normalize_az, chunk_text
"""

from .az_chunk import DEFAULT_MAX_WORDS, chunk_text
from .az_prosody import restress
from .az_text import normalize_az
from .engine import DEFAULT_MODEL_DIR, AzTTS, ModelNotFoundError

__all__ = [
    "AzTTS",
    "DEFAULT_MAX_WORDS",
    "DEFAULT_MODEL_DIR",
    "ModelNotFoundError",
    "chunk_text",
    "normalize_az",
    "restress",
]

__version__ = "1.0.0"
