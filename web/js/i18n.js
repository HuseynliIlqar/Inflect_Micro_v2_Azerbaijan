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

  "voice_label": "Səs",
  "voice_info": "«Azərbaycanca» bu layihənin modelidir. «İngiliscə» isə onun uyğunlaşdırıldığı baza modeldir — başqa müəllifin başqa səsi, ayrıca yüklənir.",
  "voice_az": "Azərbaycanca",
  "voice_en": "İngiliscə (baza model)",
  "az_only": "Yalnız azərbaycanca səs üçün.",
  "en_loading": "İngilis modeli yüklənir (38 MB, bir dəfə)…",
  "download_progress": "Model yüklənir… {loaded} / {total} MB",
  "compiling": "Model hazırlanır…",
  "progress_label": "Sintezin gedişi",
  "backend_webgpu": "WebGPU",
  "backend_wasm": "prosessor (WebAssembly)",
  "backend_wasm_threads": "prosessor (WebAssembly, {threads} axın)",
  "slow_note": "Bu brauzerdə WebGPU yoxdur, ona görə model prosessorda bir axında işləyir. Bir cümlə bir dəqiqəyə qədər çəkə bilər. Səhifə donmur, zolaq gedişi göstərir.",
  "slow_note_threads": "Bu brauzerdə WebGPU yoxdur, ona görə model prosessorda {threads} axınla işləyir. WebGPU-dan yavaşdır, amma bir axından təxminən iki dəfə sürətlidir.",
  "open_direct": "Hugging Face səhifəsinin içində çox axın mümkün deyil. Səhifəni birbaşa linkdə açsanız, təxminən iki dəfə sürətli işləyər:",
  "speed_note": "Sürət haqqında: model serverdə yox, sizin cihazınızda işləyir, ona görə sürət brauzerdən asılıdır. WebGPU olan brauzerdə sintez real vaxtdan sürətlidir. WebGPU olmayanda (bir çox telefon, Firefox) model prosessorda işləyir: birbaşa linkdə 4 axına qədər, Hugging Face səhifəsinin içində isə bir axında, çünki o səhifə çox axına icazə vermir. Bir axında bir cümlə bir dəqiqəyə qədər çəkə bilər. Bu, brauzerin məhdudiyyətidir, modelin deyil: kompüterdə komanda sətrindən (CLI) eyni model real vaxtdan 2–4 dəfə sürətli işləyir. İlk açılışda 37 MB yüklənir, sonra cihazda saxlanılır.",
  "error_worker": "Brauzer sintez üçün fon prosesini (Web Worker) başlada bilmədi. Brauzeri yeniləyin və ya başqa brauzer sınayın.",
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

  "voice_label": "Voice",
  "voice_info": "\"Azerbaijani\" is this project's model. \"English\" is the base model it was adapted from -- another author's voice, downloaded separately.",
  "voice_az": "Azerbaijani",
  "voice_en": "English (base model)",
  "az_only": "Azerbaijani voice only.",
  "en_loading": "Loading the English model (38 MB, once)…",
  "download_progress": "Loading the model… {loaded} / {total} MB",
  "compiling": "Preparing the model…",
  "progress_label": "Synthesis progress",
  "backend_webgpu": "WebGPU",
  "backend_wasm": "CPU (WebAssembly)",
  "backend_wasm_threads": "CPU (WebAssembly, {threads} threads)",
  "slow_note": "This browser has no WebGPU, so the model runs on the CPU on a single thread. A sentence can take up to a minute. The page does not freeze, and the bar shows how far it has got.",
  "slow_note_threads": "This browser has no WebGPU, so the model runs on the CPU on {threads} threads. Slower than WebGPU, but about twice as fast as a single thread.",
  "open_direct": "Inside the Hugging Face page the model cannot use more than one thread. Opened on its direct link, it runs about twice as fast:",
  "speed_note": "About speed: the model runs on your device, not on a server, so the speed depends on the browser. With WebGPU, synthesis is faster than real time. Without it (many phones, Firefox) the model runs on the CPU: on up to 4 threads on the direct link, but on a single thread inside the Hugging Face page, which does not allow more. On one thread a sentence can take up to a minute. That is a limit of the browser, not of the model: from the command line on a computer the same model runs at 2–4x real time. The first visit downloads 37 MB; after that it stays on the device.",
  "error_worker": "The browser could not start the background worker used for synthesis. Reload the page or try another browser.",
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
