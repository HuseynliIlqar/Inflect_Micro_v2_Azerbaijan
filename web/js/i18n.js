/**
 * Interface labels, Azerbaijani and English.
 *
 * Generated from `webui/i18n.py` so the page and the Gradio interface say the
 * same things; the additions below are the ones only a browser page needs.
 * Azerbaijani here is label data, not documentation -- see CLAUDE.md.
 */

export const DEFAULT_LANGUAGE = "az";

export const LANGUAGE_NAMES = { az: "Azərbaycanca", en: "English" };

export const LABELS = {
  "az": {
    "title": "Azərbaycan dilində mətn-nitq sintezi",
    "subtitle": "9.36M parametrli VITS modeli, 24 kHz, tamamilə oflayn. Mətni yazın, parametrləri tənzimləyin və dinləyin.",
    "language_label": "İnterfeys dili",
    "text_label": "Mətn",
    "text_placeholder": "Salam, bu model tamamilə yerli maşında işləyir.",
    "speak": "Səsləndir",
    "examples_label": "Nümunə cümlələr",
    "parameters": "Parametrlər",
    "speed_label": "Sürət",
    "speed_info": "Aşağı dəyər daha yavaş oxuyur. Standart: 1.0",
    "variation_label": "Dəyişkənlik",
    "variation_info": "Aşağı dəyər daha sabit, yuxarı dəyər daha canlı oxunuş verir. Standart: 0.667",
    "seed_label": "Seed",
    "seed_info": "Eyni seed hər dəfə eyni oxunuşu verir.",
    "random_seed": "Təsadüfi seed",
    "normalise_label": "Mətni normallaşdır",
    "normalise_info": "Rəqəmləri, tarixləri, faizləri və ixtisarları oxunacaq sözlərə çevirir. Söndürsəniz, mətn olduğu kimi modelə gedir.",
    "max_words_label": "Hissə uzunluğu (söz)",
    "max_words_info": "Cümlələr bu qədər sözlük hissələrə bölünür; 0 bölgünü modelin öz 280 simvolluq qaydasına buraxır. Standart: 15",
    "advanced": "Əlavə parametrlər",
    "audio_label": "Nəticə",
    "chunks_label": "Modelə gedən mətn",
    "chunks_info": "Normallaşdırmadan sonra modelin həqiqətən oxuduğu mətn və hissə sərhədləri.",
    "stats_label": "Göstəricilər",
    "error_empty": "Mətn boşdur.",
    "footer": "Bir səs, emosiya idarəsi və səs klonlama yoxdur. 9.36M parametrli modeldə nitq aydındır, lakin tam təbii deyil. Bu səsdən real bir insanı təqlid etmək və ya aldadıcı məzmun hazırlamaq üçün istifadə etməyin.",
    "chunk_one": "{count} hissə",
    "chunk_many": "{count} hissə",
    "loading": "Model yüklənir — ilk dəfə bir neçə saniyə çəkir, sonra brauzer onu yadda saxlayır.",
    "loading_phonemes": "Fonemləşdirici yüklənir…",
    "loading_model": "Model yüklənir…",
    "synthesising": "Səsləndirilir…",
    "download": "Endir",
    "offline_note": "Səhifə yükləndikdən sonra internet olmadan da işləyir — bütün hesablama sizin brauzerinizdə gedir.",
    "en_note": "Bu səhifə yalnız azərbaycanca danışır. İngilis baza modeli üçün repodakı CLI-dən istifadə edin.",
    "stats": "{seconds} s səs, {elapsed} s-də hazırlandı ({realtime}x real vaxt) · {chunks} · {rate} Hz"
  },
  "en": {
    "title": "Azerbaijani text to speech",
    "subtitle": "A 9.36M-parameter VITS model, 24 kHz, fully offline. Type the text, adjust the parameters and listen.",
    "language_label": "Interface language",
    "text_label": "Text",
    "text_placeholder": "Salam, bu model tamamilə yerli maşında işləyir.",
    "speak": "Speak",
    "examples_label": "Example sentences",
    "parameters": "Parameters",
    "speed_label": "Speed",
    "speed_info": "Lower is slower. Default: 1.0",
    "variation_label": "Variation",
    "variation_info": "Lower is steadier, higher is livelier. Default: 0.667",
    "seed_label": "Seed",
    "seed_info": "The same seed gives the same reading every time.",
    "random_seed": "Random seed",
    "normalise_label": "Normalise the text",
    "normalise_info": "Rewrites digits, dates, percentages and abbreviations as the words a reader would say. Turn it off to pass the text through as written.",
    "max_words_label": "Chunk length (words)",
    "max_words_info": "Sentences are cut into chunks of this many words; 0 leaves splitting to the package's own 280-character rule. Default: 15",
    "advanced": "Advanced",
    "audio_label": "Result",
    "chunks_label": "What reaches the model",
    "chunks_info": "The text the model actually reads after normalisation, with the chunk boundaries.",
    "stats_label": "Statistics",
    "error_empty": "The text is empty.",
    "footer": "One voice, no emotion control, no voice cloning. At 9.36M parameters the speech is clear but not fully natural. Do not use this voice to impersonate a real person or to produce deceptive content.",
    "chunk_one": "{count} chunk",
    "chunk_many": "{count} chunks",
    "loading": "Loading the model — a few seconds the first time, then the browser keeps it.",
    "loading_phonemes": "Loading the phonemiser…",
    "loading_model": "Loading the model…",
    "synthesising": "Synthesising…",
    "download": "Download",
    "offline_note": "Once the page has loaded it works without a network — everything runs in your browser.",
    "en_note": "This page speaks Azerbaijani only. For the English base model, use the CLI in the repository.",
    "stats": "{seconds} s of audio in {elapsed} s ({realtime}x real time) · {chunks} · {rate} Hz"
  }
};

export function label(language, key, values) {
  const table = LABELS[language] ?? LABELS[DEFAULT_LANGUAGE];
  let text = table[key] ?? key;
  if (values) {
    for (const [name, value] of Object.entries(values)) {
      text = text.split(`{${name}}`).join(String(value));
    }
  }
  return text;
}

export function chunkCount(language, count) {
  return label(language, count === 1 ? "chunk_one" : "chunk_many", { count });
}
