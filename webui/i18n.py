"""Interface labels in Azerbaijani and English.

This is the one module where Azerbaijani prose is allowed: the strings below are
*language data*, the same category as `LETTER_NAMES` in `aztts/az_text.py`, not
documentation. Code, comments and docstrings stay English. See CLAUDE.md.

Both tables must hold exactly the same keys -- `tests/test_i18n.py` enforces it,
so a label added to one language cannot silently go missing in the other.

    >>> label("az", "speak")
    'Səsləndir'
    >>> label("en", "speak")
    'Speak'
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

__all__ = [
    "DEFAULT_LANGUAGE",
    "LANGUAGES",
    "LANGUAGE_NAMES",
    "chunk_count",
    "label",
    "table",
    "voice_choices",
]

DEFAULT_LANGUAGE = "az"

_AZ: Mapping[str, str] = {
    # -- page ---------------------------------------------------------------
    "title": "Azərbaycan dilində mətn-nitq sintezi",
    "subtitle": (
        "9.36M parametrli VITS modeli, 24 kHz, tamamilə oflayn. "
        "Mətni yazın, parametrləri tənzimləyin və dinləyin."
    ),
    "language_label": "İnterfeys dili",
    "voice_label": "Səs",
    "voice_info": (
        "«Azərbaycanca» bu layihənin modelidir. «İngiliscə» isə onun uyğunlaş"
        "dırıldığı baza modeldir — başqa müəllifin başqa səsi, ayrıca yüklənir."
    ),
    "voice_az": "Azərbaycanca",
    "voice_en": "İngiliscə (baza model)",
    "voice_en_missing": (
        "İngiliscə baza model yüklənməyib (38 MB). Terminalda işlədin:\n"
        "    python training/scripts/download_model.py"
    ),
    "voice_en_pointer": (
        "İngiliscə çəkilər Git LFS göstəricisidir, faylın özü deyil. "
        "Klon Git LFS olmadan edilib. Düzəlişi:\n"
        "    git lfs install && git lfs pull"
    ),
    "az_only": "Yalnız azərbaycanca səs üçün.",
    # -- input --------------------------------------------------------------
    "text_label": "Mətn",
    "text_placeholder": "Salam, bu model tamamilə yerli maşında işləyir.",
    "speak": "Səsləndir",
    "examples_label": "Nümunə cümlələr",
    # -- parameters ---------------------------------------------------------
    "parameters": "Parametrlər",
    "speed_label": "Sürət",
    "speed_info": "Aşağı dəyər daha yavaş oxuyur. Standart: 1.0",
    "variation_label": "Dəyişkənlik",
    "variation_info": (
        "Aşağı dəyər daha sabit, yuxarı dəyər daha canlı oxunuş verir. "
        "Standart: 0.667"
    ),
    "seed_label": "Seed",
    "seed_info": "Eyni seed hər dəfə eyni oxunuşu verir.",
    "random_seed": "Təsadüfi seed",
    "normalise_label": "Mətni normallaşdır",
    "normalise_info": (
        "Rəqəmləri, tarixləri, faizləri və ixtisarları oxunacaq sözlərə çevirir. "
        "Söndürsəniz, mətn olduğu kimi modelə gedir."
    ),
    "max_words_label": "Hissə uzunluğu (söz)",
    "max_words_info": (
        "Cümlələr bu qədər sözlük hissələrə bölünür; 0 bölgünü modelin öz "
        "280 simvolluq qaydasına buraxır. Standart: 15"
    ),
    "prosody_label": "Vurğu (prosodiya)",
    "prosody_info": (
        "eSpeak demək olar ki, hər sözü vurğulayır və oxunuşu yastılaşdıran da "
        "budur. «Ehtiyatlı» yalnız təlim datasının vurğusuz göstərdiyi sözləri "
        "boşaldır, «geniş» daha da irəli gedir (təcrübi)."
    ),
    "prosody_off": "söndürülüb",
    "prosody_safe": "ehtiyatlı",
    "prosody_wide": "geniş",
    "prosody_drop_label": "Vurğunu tamamilə sil",
    "prosody_drop_info": "Vurğunu zəiflətmək əvəzinə büsbütün çıxarır.",
    "cleanup_label": "Səsi təmizlə",
    "cleanup_info": (
        "Vokoderin metal rezonansını süzür və səviyyəni bərabərləşdirir. "
        "Nitqin özünə də toxuna bilər -- müqayisə edin."
    ),
    "device_label": "Cihaz",
    "advanced": "Əlavə parametrlər",
    # -- output -------------------------------------------------------------
    "audio_label": "Nəticə",
    "chunks_label": "Modelə gedən mətn",
    "chunks_info": (
        "Normallaşdırmadan sonra modelin həqiqətən oxuduğu mətn və hissə "
        "sərhədləri."
    ),
    "stats_label": "Göstəricilər",
    "cli_label": "Bu ayarların terminal qarşılığı",
    # -- messages -----------------------------------------------------------
    "stats_template": (
        "{seconds:.2f} s səs, {elapsed:.2f} s-də hazırlandı "
        "({realtime:.1f}x real vaxt) · {chunks} · {rate} Hz"
    ),
    # Azerbaijani takes no plural after a numeral: "2 hissə", not "2 hissələr".
    "chunk_one": "{count} hissə",
    "chunk_many": "{count} hissə",
    "stats_cleaned": " · təmizləndi: {notches}",
    "stats_cleaned_none": " · təmizləndi (rezonans tapılmadı)",
    "error_empty": "Mətn boşdur.",
    "error_model": (
        "Model tapılmadı. `model/` qovluğunun tam kopyalandığını yoxlayın "
        "(təxminən 37 MB olmalıdır)."
    ),
    "error_failed": "Sintez alınmadı: {error}",
    "loading": "Model yüklənir, bir neçə saniyə çəkə bilər...",
    # -- footer -------------------------------------------------------------
    "footer": (
        "Bir səs, emosiya idarəsi və səs klonlama yoxdur. 9.36M parametrli "
        "modeldə nitq aydındır, lakin tam təbii deyil. Bu səsdən real bir "
        "insanı təqlid etmək və ya aldadıcı məzmun hazırlamaq üçün istifadə "
        "etməyin."
    ),
}

_EN: Mapping[str, str] = {
    # -- page ---------------------------------------------------------------
    "title": "Azerbaijani text to speech",
    "subtitle": (
        "A 9.36M-parameter VITS model, 24 kHz, fully offline. "
        "Type the text, adjust the parameters and listen."
    ),
    "language_label": "Interface language",
    "voice_label": "Voice",
    "voice_info": (
        "\"Azerbaijani\" is this project's model. \"English\" is the base model it "
        "was adapted from -- another author's voice, downloaded separately."
    ),
    "voice_az": "Azerbaijani",
    "voice_en": "English (base model)",
    "voice_en_missing": (
        "The English base model has not been downloaded (38 MB). Run:\n"
        "    python training/scripts/download_model.py"
    ),
    "voice_en_pointer": (
        "The English weights are a Git LFS pointer, not the file itself. "
        "The clone was made without Git LFS. Fix it with:\n"
        "    git lfs install && git lfs pull"
    ),
    "az_only": "Azerbaijani voice only.",
    # -- input --------------------------------------------------------------
    "text_label": "Text",
    "text_placeholder": "Salam, bu model tamamilə yerli maşında işləyir.",
    "speak": "Speak",
    "examples_label": "Example sentences",
    # -- parameters ---------------------------------------------------------
    "parameters": "Parameters",
    "speed_label": "Speed",
    "speed_info": "Lower is slower. Default: 1.0",
    "variation_label": "Variation",
    "variation_info": "Lower is steadier, higher is livelier. Default: 0.667",
    "seed_label": "Seed",
    "seed_info": "The same seed gives the same reading every time.",
    "random_seed": "Random seed",
    "normalise_label": "Normalise the text",
    "normalise_info": (
        "Rewrites digits, dates, percentages and abbreviations as the words a "
        "reader would say. Turn it off to pass the text through as written."
    ),
    "max_words_label": "Chunk length (words)",
    "max_words_info": (
        "Sentences are cut into chunks of this many words; 0 leaves splitting "
        "to the package's own 280-character rule. Default: 15"
    ),
    "prosody_label": "Stress (prosody)",
    "prosody_info": (
        "eSpeak stresses nearly every word, which is what flattens the "
        "reading. \"Safe\" only unstresses words the training data already "
        "shows without an accent; \"wide\" goes further (experimental)."
    ),
    "prosody_off": "off",
    "prosody_safe": "safe",
    "prosody_wide": "wide",
    "prosody_drop_label": "Remove the accent outright",
    "prosody_drop_info": "Removes the accent instead of demoting it.",
    "cleanup_label": "Clean up the audio",
    "cleanup_info": (
        "Filters out the vocoder's metallic resonance and evens out the level. "
        "It can touch the speech as well -- compare before deciding."
    ),
    "device_label": "Device",
    "advanced": "Advanced",
    # -- output -------------------------------------------------------------
    "audio_label": "Result",
    "chunks_label": "What reaches the model",
    "chunks_info": (
        "The text the model actually reads after normalisation, with the chunk "
        "boundaries."
    ),
    "stats_label": "Statistics",
    "cli_label": "The same settings on the command line",
    # -- messages -----------------------------------------------------------
    "stats_template": (
        "{seconds:.2f} s of audio in {elapsed:.2f} s "
        "({realtime:.1f}x real time) · {chunks} · {rate} Hz"
    ),
    "chunk_one": "{count} chunk",
    "chunk_many": "{count} chunks",
    "stats_cleaned": " · cleaned: {notches}",
    "stats_cleaned_none": " · cleaned (no resonance found)",
    "error_empty": "The text is empty.",
    "error_model": (
        "The model was not found. Check that `model/` was copied in full "
        "(it should be about 37 MB)."
    ),
    "error_failed": "Synthesis failed: {error}",
    "loading": "Loading the model, this can take a few seconds...",
    # -- footer -------------------------------------------------------------
    "footer": (
        "One voice, no emotion control, no voice cloning. At 9.36M parameters "
        "the speech is clear but not fully natural. Do not use this voice to "
        "impersonate a real person or to produce deceptive content."
    ),
}

LANGUAGES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {"az": MappingProxyType(dict(_AZ)), "en": MappingProxyType(dict(_EN))}
)

# What the language selector shows. Each name is written in its own language.
LANGUAGE_NAMES: Mapping[str, str] = MappingProxyType(
    {"az": "Azərbaycanca", "en": "English"}
)


def table(language: str) -> Mapping[str, str]:
    """The label table for a language, falling back to the default one."""
    return LANGUAGES.get(language, LANGUAGES[DEFAULT_LANGUAGE])


def label(language: str, key: str, **format_args: object) -> str:
    """One label, formatted. Unknown keys return the key itself, not a crash.

    A missing label should leave the interface usable and the mistake visible,
    which is why the key is echoed rather than raising.
    """
    text = table(language).get(key, key)
    if not format_args:
        return text
    try:
        return text.format(**format_args)
    except (IndexError, KeyError):
        return text


def voice_choices(language: str, english_available: bool) -> list[tuple[str, str]]:
    """Labels for the voice selector, over the stable `az`/`en` values.

    A missing English checkpoint is said out loud rather than hidden: the user
    can download it and reload, and a silently absent option looks like a bug.
    """
    english = label(language, "voice_en")
    if not english_available:
        english += " — " + label(language, "voice_en_missing").split("\n")[0]
    return [(label(language, "voice_az"), "az"), (english, "en")]


def chunk_count(language: str, count: int) -> str:
    """"3 chunks" / "3 hissə" -- English pluralises, Azerbaijani does not."""
    key = "chunk_one" if count == 1 else "chunk_many"
    return label(language, key, count=count)
