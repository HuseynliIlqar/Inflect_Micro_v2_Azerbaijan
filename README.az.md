# Azərbaycanca TTS

Azərbaycan dilində danışan **tam oflayn** mətn→səs modeli. 9.36M parametr,
24 kHz mono, adi noutbukun prosessorunda **real vaxtdan 2–4 dəfə sürətli**
işləyir. Server, API açarı, internet — heç biri lazım deyil.

Bu model Owen Song-un
**[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
checkpoint-inin fine-tune-udur (Apache-2.0) — 25.07 saatlıq azərbaycanca nitq
üzərində 200.000 addım, 410 tensordan 409-u bit-bit köçürülüb.

> For the English documentation: **[README.md](README.md)**

### Üç istifadə yolu

| | |
| --- | --- |
| **İndi dinlə** | [Playground](https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan) — brauzerdə işləyir, heç nə quraşdırmadan |
| **Yerli işlət** | Reponu `git clone` et, `pip install`, `python say.py "Salam."` |
| **Çəkiləri götür** | Hub-da [`ilqarrrr/Inflect_Micro_v2_Azerbaijan`](https://huggingface.co/ilqarrrr/Inflect_Micro_v2_Azerbaijan) — PyTorch və ONNX |

---

## Məzmun

[Sürətli başlanğıc](#sürətli-başlanğıc) · [Dinlə](#dinlə) ·
[Komanda sətri](#komanda-sətri) · [Python-dan](#python-dan) ·
[Brauzer playground-u](#brauzer-playground-u) ·
[Yerli interfeys](#yerli-interfeys) · [Hər iki səs](#hər-iki-səs) ·
[Mətn qatı nə edir](#mətn-qatı-nə-edir) ·
[Kod üzərində işləmək](#kod-üzərində-işləmək) · [Model haqqında](#model-haqqında) ·
[Məhdudiyyətlər](#məhdudiyyətlər) ·
[Lisenziya və kommersiya istifadəsi](#lisenziya-və-kommersiya-istifadəsi) ·
[Təşəkkür](#təşəkkür)

---

## Sürətli başlanğıc

Python 3.11 və ya daha yeni. Çəkilər [Git LFS](https://git-lfs.com) ilə
saxlanılır, ona görə klonlamadan **əvvəl** onu quraşdırın:

```bash
git lfs install
git clone https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan
cd Inflect_Micro_v2_Azerbaijan
```

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
./.venv/bin/pip install -r requirements.txt                   # Linux / macOS
```

```bash
python say.py "Salam, necəsiniz?"
```

Səs `out/01.wav` faylına düşür. Vəssalam.

Quraşdırma zamanı heç nə yüklənmir: hər iki checkpoint (hərəsi 37 MB) repodadır.
Yalnız prosessorda işlədəcəksinizsə, PyTorch-un kiçik variantını götürün:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Dinlə

Hazır WAV faylları, hər səs üçün bir dəst — heç nə quraşdırmadan dinləyin.

**Azərbaycanca**, bu layihənin modeli:

| Fayl | Cümlə |
| --- | --- |
| [`01-salam.wav`](samples/01-salam.wav) | `Salam, bu model tamamilə yerli maşında işləyir.` |
| [`02-payiz.wav`](samples/02-payiz.wav) | `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` |
| [`03-sual.wav`](samples/03-sual.wav) | `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` |
| [`04-reqem.wav`](samples/04-reqem.wav) | `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` |
| [`05-uzun.wav`](samples/05-uzun.wav) | Uzun cümlə — intonasiyanın harada yastılaşdığını eşitmək üçün |

**İngiliscə**, baza model:

| Fayl | Cümlə |
| --- | --- |
| [`01-hello.wav`](samples/en/01-hello.wav) | `Hello, this model runs completely offline on your machine.` |
| [`02-autumn.wav`](samples/en/02-autumn.wav) | `Autumn had come, and the streets were covered with yellow leaves.` |
| [`03-question.wav`](samples/en/03-question.wav) | `Have you read this book? I found it very interesting.` |

Yalnız model həqiqətən dəyişəndə yenidən yaradın — hər iki dəst Git LFS-dən
keçir:

```bash
python say.py --out samples                 # azərbaycanca
python say.py --voice en --out samples/en   # ingiliscə
```

## Komanda sətri

```bash
python say.py "Bir cümlə."                        # -> out/01.wav
python say.py "Birinci." "İkinci."                # -> out/01.wav, out/02.wav
python say.py -f kitab.txt --one-file -o out/kitab  # bütün faylı bir WAV-a
python say.py --show-text "II Dünya, 25% artım"   # modelə nə gedir, göstər
python say.py --speed 0.85 --seed 42 "Daha yavaş."
python say.py --voice en "Hello there."           # ingilis baza modeli
```

| Bayraq | Standart | Nə edir |
| --- | ---: | --- |
| `--speed` | `1.0` | 0.5–2.0. Aşağı dəyər daha yavaş |
| `--variation` | `0.667` | 0.0–1.0. Aşağı daha sabit, yuxarı daha canlı |
| `--seed` | `7` | Eyni seed eyni səsi verir. Bəyənmədiyiniz cümlə üçün dəyişin |
| `--voice` | `az` | `en` ingilis baza modeli ilə danışır |
| `--device` | `cpu` | `cpu` və ya `cuda` |
| `--out` | `out/` | Çıxış qovluğu |
| `--raw` | — | Normallaşdırmanı söndürür |
| `--max-words` | `15` | Hissə uzunluğu. `0` bölgünü modelə buraxır |
| `--prosody` | `off` | `safe` / `wide` — vurğu işarələrini seyrəldir (təcrübi) |
| `--show-text` | — | Normallaşdırılmış mətni və hissə sərhədlərini çap edir |
| `--allow-profanity` | — | Söyüşləri bip əvəzinə olduğu kimi oxuyur |

## Python-dan

```python
from aztts import AzTTS

tts = AzTTS()                       # bir dəfə yaradın, təkrar işlədin
tts.save("Mətn burada.", "out/a.wav")

waveform = tts.synthesize("Xam massiv lazımdırsa.", speed=1.1, seed=3)
# → float32 numpy massiv, mono, [-1, 1], 24 000 Hz
```

Çəkilər bu repo ilə gəlir, ona görə `AzTTS()` heç nə yükləmir. Hub-dakı
nüsxədən istifadə etmək üçün:

```python
from huggingface_hub import snapshot_download

tts = AzTTS(snapshot_download("ilqarrrr/Inflect_Micro_v2_Azerbaijan"))
```

Mətn funksiyaları modeldən asılı deyil, ayrıca da işlədilə bilər:

```python
from aztts import normalize_az, chunk_text

normalize_az("II qrupda 25% artım oldu.")
# 'İkinci qrupda iyirmi beş faiz artım oldu.'

chunk_text("Uzun bir cümlə...", max_words=15)
# ('Uzun bir cümlə...',)
```

## Brauzer playground-u

**<https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan>**

Model **tamamilə brauzerdə** də işləyir, hər iki səslə, heç nə heç yerə
göndərilmir. Səhifə yükləndikdən sonra internetsiz də işləyir.

- ONNX Runtime Web, WebGPU üzərində — noutbukda təxminən 8x real vaxt
- Fonemləşdirmə üçün WebAssembly-yə kompilyasiya edilmiş eSpeak NG
- JavaScript-ə portlanmış azərbaycanca mətn qatı
- Sintez Web Worker-də gedir, ona görə səhifə donmur və irəliləyiş zolağı göstərilir

**Playground yavaş görünürsə, səbəb modeldə yox, brauzerdədir.** WebGPU
olmayanda (bir çox telefon, Firefox) model prosessorda, int8 decoder ilə işləyir:
təxminən 1.5 dəfə sürətlidir və yarı həcmdə yüklənir. Onun səsə təsiri modelin
iki oxunuşu arasındakı öz fərqindən kiçik ölçülüb; səhifə bu rejimin işlədiyini
bildirir (`?precision=fp32` onu söndürür).
[Birbaşa linkdə](https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space/index.html)
səhifə cross-origin izolyasiyalıdır və 4 axına qədər istifadə edir, bu da bir
axından təxminən iki dəfə sürətlidir. Hugging Face səhifəsinin içində izolyasiya
mümkün deyil və model bir axında işləyir; telefonda bir cümlə bir dəqiqəyə qədər
çəkə bilər. Səhifə hansı halın olduğunu bildirir və gedişi göstərməyə davam edir.
Eyni model komanda sətrindən real vaxtdan 2–4 dəfə sürətli işləyir. İlk açılışda
37 MB yüklənir, sonra model cihazda saxlanılır.

Port sözə görə qəbul edilmir: Python orijinalından yaradılmış qızıl fayllarla
yoxlanılır — 24 022 rəqəm müqayisəsi, 550 normallaşdırma cümləsi, 550 bölgü — və
ONNX yolu Python-dakı onnxruntime ilə tutuşdurulur (eyni sample sayı, yeddi
onluq dəqiqliklə eyni RMS). Təfərrüat: [web/README.md](web/README.md).

## Yerli interfeys

Eyni şey, Hub olmadan, və CLI-nin bütün bayraqları ilə:

```bash
pip install -r requirements-app.txt     # və ya: pip install -e ".[app]"
python app.py                           # http://127.0.0.1:7860
```

Gradio bilərəkdən **əsas asılılıq deyil** — `aztts`-i kitabxana kimi işlədən
adam veb yığını üçün ödəməməlidir.

Səhifə komanda sətrinin gizlətdiyi iki şeyi göstərir: modelə həqiqətən çatan
normallaşdırılmış mətni və seçdiyiniz ayarlara uyğun `say.py` əmrini — yəni həm
də CLI-ni öyrənmək üçün işləyir. `Səsi təmizlə` vokoderin metal rezonansını
hazır kliplə süzür, ffmpeg tələb etmir və hansı tezlikləri sildiyini bildirir.

## Hər iki səs

| | Azərbaycanca | İngiliscə |
| --- | --- | --- |
| Model | bu layihənin | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2), dəyişdirilməmiş |
| Yeri | `model/` | `model-en/` |
| Əmr | `python say.py "Salam."` | `python say.py --voice en "Hello."` |
| Mətn qatı | normallaşdırma, bölgü, vurğu | yalnız bölgü |

Hər ikisi repo ilə gəlir, ona görə klon heç nə yükləmədən hər iki dildə danışır.
Azərbaycanca mətn qatları ingiliscəyə aid deyil: `normalize_az` rəqəmləri
azərbaycanca sözlərə çevirərdi, ona görə o da, vurğu qatı da orada söndürülür və
interfeysdə passivləşir. İngilis mətni eSpeak-in `en-us` səsi ilə fonemlərə
çevrilib runtime-a verilir.

**Bu layihənin öz modeli ingiliscə danışmır.** O, 200.000 addım boyu tək bir
azərbaycanca səslə uyğunlaşdırılıb; ingilis mətni ona verilsə, çıxan şey həmin
səsin heç vaxt eşitmədiyi fonemləri oxumasıdır. İngilis çəkilərinin haradan
gəldiyi və necə istinad ediləcəyi `model-en/README.md`-dədir.

## Mətn qatı nə edir

Model adi nəsr üzərində öyrədilib, ona görə rəqəmlər və ixtisarlar xam halda
çatsa, səhv oxunur — `II` «ı ı» kimi səslənir, `25%` faiz işarəsini tamamilə
itirir. İki addım bunun qarşısını alır.

**1. Normallaşdırma** (`aztts/az_text.py`) — rəqəmlər, Roma rəqəmləri, tarixlər,
faizlər, ölçü vahidləri və ixtisarlar oxunan sözlərə çevrilir:

| Giriş | Modelə çatan |
| --- | --- |
| `II Dünya müharibəsi` | `İkinci Dünya müharibəsi` |
| `01/09/1939` | `bir sentyabr min doqquz yüz otuz doqquzuncu il` |
| `25%` | `iyirmi beş faiz` |
| `5 kq` | `beş kiloqram` |
| `19,99 AZN` | `on doqquz manat doxsan doqquz qəpik` |
| `3,14` | `üç tam yüzdə on dörd` |

**2. Hissələrə bölmə** (`aztts/az_chunk.py`) — cümlələr təxminən on beş sözlük
hissələrə bölünür, çünki uzun cümlənin sonuna doğru bu modelin intonasiyası
yastılaşır və söz sonluqları kəsilir. Hissələr arasına durğu işarəsinə uyğun
pauza qoyulur ki, nəticə kəsik-kəsik səslənməsin.

`--show-text` nə baş verdiyini göstərir, `--raw` isə bunu söndürür.

**Hər ikisindən əvvəl, senzura** (`aztts/az_profanity.py`) — söyüşlər
televiziyadakı kimi 1 kHz bip səsi ilə əvəz olunur: `Sən qəhbəsən, bildin?`
"Sən *(bip)*, bildin?" kimi səslənir. Şəkilçili formalar, Azərbaycan hərfləri
olmadan yazılış (`qehbe`) və hərf yerinə rəqəm (`s1kdir`) tutulur; eyni
hərflərlə başlayan təmiz sözlər (`şikayət`, `sikkə`, `götürmək`) tutulmur.
`axmaq`, `eşşək` kimi yüngül təhqirlər olduğu kimi oxunur.

Brauzer playground-unda və host edilən interfeysdə senzura həmişə açıqdır və
onu söndürmək üçün heç bir düymə yoxdur. Clone edilmiş repoda da default
açıqdır və yalnız qəsdən söndürülür: `python say.py --allow-profanity`,
`python app.py --allow-profanity` və ya
`AzTTS().synthesize(text, censor=False)`. `--raw` onu söndürmür; İngilis səsinə
də aiddir, çünki hər iki səsə Azərbaycanca yazmaq olar. Bu, adi mətn üçün
filtrdir, zəmanət deyil: hərf-hərf ayrı yazılmış söz (`s i k`) keçir.

## Kod üzərində işləmək

```bash
python -m pytest        # 303 test, üç saniyə, modeli heç vaxt yükləmir
```

<details>
<summary>Testlər nəyi əhatə edir</summary>

| | |
| --- | ---: |
| Mətn normallaşdırma, hissələrə bölmə, vurğu qatı | 116 |
| Sürətli monotonic alignment (train optimizasiyası) | 25 |
| İnterfeys etiketləri, parametrlər, rezonans filtri | 86 |

Brauzer portunun öz testləri var, Node ilə işləyir:

```bash
node web/tests/test_num_az.mjs      # num2words ilə 24 022 yoxlama
node web/tests/test_az_text.mjs     # normalize_az ilə 550 cümlə
node web/tests/test_az_chunk.mjs    # chunk_text ilə eyni 550 cümlə
node web/tests/test_progress.mjs    # irəliləyiş zolağı və keşlənən model yükləməsi
```

</details>

<details>
<summary>Repo strukturu</summary>

```
say.py            Komanda sətri — əsas giriş nöqtəsi
app.py            Brauzer interfeysi (Gradio) — könüllü
aztts/
  engine.py       AzTTS — model yüklənməsi, normallaşdırma, bölmə, sintez
  az_text.py      normalize_az: Roma rəqəmi, qısaltma, abreviatura
  az_dates.py     Tarix, saat, telefon nömrəsi → söz
  az_amounts.py   Pul, faiz, ölçü vahidi, kəsr, aralıq → söz
  az_words.py     Hərf registri, ahəng qanunu, rəqəm → söz
  az_tables.py    Söz cədvəlləri (aylar, vahidlər, valyutalar, ...)
  az_chunk.py     Cümlələri modelin bacardığı uzunluğa bölür
  az_prosody.py   Vurğu qatı (könüllü, --prosody)
  az_profanity.py Söyüşlər -> bip (--allow-profanity olmadıqca açıq)
  en_voice.py     İngilis baza modeli, eyni runtime ilə
  console.py      Windows konsolunu UTF-8-ə keçirir
webui/
  i18n.py         İnterfeys etiketləri, azərbaycanca və ingiliscə
  runner.py       Parametrlər, CLI qarşılığı, bir sintez
  cleanup.py      Tək-klip rezonans filtri (ffmpeg-siz)
web/              Brauzer playground-u (Hugging Face static Space)
model/            Azərbaycanca çəkilər + runtime (37 MB) — toxunmayın
model-en/         İngilis baza modelin çəkiləri (37 MB) — toxunmayın
tools/            seed_sweep, audio_postprocess — keyfiyyət alətləri
training/         Modelin necə hazırlandığı (baza model yüklənir)
packaging/        GitHub / Hugging Face / Kaggle yayımı
samples/          Nümunə səslər
tests/            303 test
out/              Yaratdığınız WAV faylları
```

`model/` upstream export paketidir və özü ilə birlikdə lazım olan hər şeyi
gətirir (çəkilər, fonem frontend-i, VITS runtime-ı). Bizim bütün kodumuz
`aztts/`, `webui/` və `web/` altındadır.

</details>

<details>
<summary>Keyfiyyət alətləri</summary>

```bash
# Bir cümləni 10 fərqli seed ilə səsləndir, ən yaxşısını seç
python tools/seed_sweep.py --seeds 10 "Xoş gəlmisiniz."

# Vokoderin metal rezonansını tap, sonra ffmpeg ilə təmizlə
python tools/audio_postprocess.py analyze samples
python tools/audio_postprocess.py clean out --lowpass 11000
```

Tətbiqin təkrar-təkrar işlətdiyi cümlələr üçün (salamlama, xəta mesajı) bir dəfə
yaxşı seed seçib sabitləmək dəyər — oxunuş nəzərəçarpacaq dərəcədə yaxşılaşır.

</details>

Nəyisə dəyişməzdən əvvəl [CLAUDE.md](CLAUDE.md) faylını oxuyun: qurulum, dəyişikliyi
yoxlayan əmrlər və mənbəyi oxumaqla görünməyən tələlər orada yazılıb. Cursor
eyni qaydaları `.cursor/rules/` qovluğundan götürür. Yayım addımları:
[packaging/PUBLISHING.md](packaging/PUBLISHING.md).

## Model haqqında

| | |
| --- | --- |
| Baza | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Parametr | 9.36M |
| Səs | 24 kHz mono, tək danışan |
| Təlim | 200.000 addım = 1.343 epoxa, 25.07 saat azərbaycanca audio |
| Sürət | Prosessorda 2–4x real vaxt |

Ölçmələr, dataset haqqında dəlillər və məhdudiyyətlər tam şəkildə
[docs/MODEL.md](docs/MODEL.md) faylındadır. [`training/`](training/README.md)
bütün prosesi saxlayır: rəsmi toolkit-ə tətbiq edilmiş yamaq, preset-lər,
nəticələr və loss tarixçəsi.

Onun bir hissəsini burada təkrarlamağa dəyər. Toolkit-in `maximum_path`
funksiyası hər addımın 8.7 saniyəsini yeyən təmiz Python dövrü idi; vektorlaşmış
torch versiyası bunu **1.395 saniyəyə** endirdi. İyirmi gün 3.2 günə, ~$213
~$34-a düşdü, və 25 test nəticənin orijinalla **bit-bit eyni** olduğunu sübut
edir.

## Məhdudiyyətlər

- **Bir səs.** Səs klonlama və çoxdanışanlı dəstək yoxdur.
- **Emosiya idarəsi yoxdur.** Oxunuş sakit və neytraldır.
- 9.36M parametrdə nitq aydın və anlaşıqlıdır, amma tam təbii deyil; vokoder
  bəzən metal rezonans buraxır.
- Nadir adlar, əcnəbi sözlər və qeyri-adi yazılışlar fonem frontend-inin
  mərhəmətindədir.
- Normallaşdırma `say.py` və `AzTTS` daxilində avtomatik işləyir. `model/`
  paketini birbaşa çağırırsınızsa, `normalize_az`-ı özünüz çağırmalısınız.

## Lisenziya və kommersiya istifadəsi

Bizim kod və hər iki checkpoint **Apache-2.0**-dır ([LICENSE](LICENSE)).
Kommersiya istifadəsinə icazə verilir. Yalnız bir şərt var və o da modeldən yox,
fonemləşdiricidən gəlir.

| Hissə | Lisenziya | Kommersiya istifadəsi |
| --- | --- | --- |
| Bizim kod (`aztts/`, `say.py`, `app.py`, `webui/`, `web/`) | Apache-2.0 | Bəli |
| Azərbaycanca çəkilər (`model/`) | Apache-2.0 | Bəli |
| İngiliscə çəkilər (`model-en/`) | Apache-2.0 | Bəli |
| Runtime hissələri (VITS, BigVGAN, alias-free-torch) | MIT / Apache-2.0 | Bəli |
| **`phonemizer` + eSpeak NG** | **GPL-3.0-or-later** | **İşlətmək: bəli. Yaymaq: aşağıya bax** |

### Məhsul buraxmazdan əvvəl yoxlanmalı tək şey

Fonemləşdirmə [`phonemizer`](https://github.com/bootphon/phonemizer) və
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) kitabxanalarını **eyni
prosesin içində** çağırır, hər ikisi isə GPL-3.0-or-later-dir. Şirkət daxilində
işlətmək heç nəyi dəyişmir. Birləşmiş məhsul yaymaq — masaüstü tətbiq, konteyner
image-i, müştəriyə verilən binar — həmin komponentlər üçün GPL-3.0 öhdəliklərini
gətirir, o cümlədən uyğun mənbə kodunu təklif etmək öhdəliyini.

Modelin özü eSpeak-ə möhtac deyil: `model/config.json` faylında
`accepts_prephonemized_input: true` yazılıb — yəni GPL-dən uzaq qalmalı olan
məhsul mətni öz aləti ilə fonemlərə çevirib modelə verə bilər.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) hər iki yolu açıqlayır və bütün
üçüncü tərəf komponentlərini sadalayır.

### Səslər haradan gəlir

**Azərbaycanca.**
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)
üzərində öyrədilib. Repozitoriyada lisenziya faylı yoxdur; Hugging Face icma
tabında birbaşa soruşulanda müəllif sərbəst istifadəyə icazə verdiyini bildirib
və ad çəkilməsini xahiş edib. **Şərt budur: istinad.** Audio çox güman ki
sintetikdir, şərtləri bilinməyən başqa bir TTS sistemi tərəfindən hazırlanıb —
[docs/MODEL.md](docs/MODEL.md).

**İngiliscə.** `owensong/Inflect-Micro-v2`, dəyişdirilmədən yenidən yayımlanır.
Model kartından kommersiya qərarı üçün vacib üç fakt:

- səs **sintetikdir**. Paket real danışanın səs korpusunu yaymır və səsi hər
  hansı real insanın kimliyi kimi təqdim etmir — yəni təmizlənməli şəxsiyyət və
  ya bənzərlik hüququ yoxdur;
- buraxılış **açıq-çəki**dir, açıq-data deyil: korpus konveyeri və filtrləmə
  infrastrukturu qapalıdır, ona görə təlim datası ictimai materiallardan
  yoxlanıla bilməz. Uyğunluq prosesiniz bu auditi tələb edirsə, müəllif e-poçt
  vasitəsilə deployment sorğularını qəbul edir;
- məsuliyyətli istifadə bəyanatı istəyir ki, kontekst yanlış təəssürat yarada
  biləcəksə sintetik nitq açıqlansın və səs heç kimi təqlid etmək üçün
  işlədilməsin.

### Hər iki səsdə qadağan olan

Real bir insanı təqlid etmək, aldadıcı məzmun hazırlamaq və ya sintetik nitqi
əsl səsyazma kimi təqdim etmək. Bu, həm bu layihənin, həm də upstream-in
mövqeyidir.

Bu, lisenziyaların oxunuşudur, hüquqi məsləhət deyil. Məhsulunuz buna
söykənirsə, ixtisaslı birinə təsdiqlətdirin.

## Təşəkkür

Bu model iki nəfər öz işini açıq paylaşdığı üçün mövcuddur.

**[`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**
— Owen Song, Apache-2.0. Buradakı hər çəki onunkindən başlayıb: 410 tensordan
409-u bit-bit köçürülüb və azərbaycanca fonemlər onsuz da onun simvol dəstində
var idi. O checkpoint olmasaydı, bu iş günlərlə yox, aylarla təlim tələb
edərdi.

**[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)**
— `ughurabbasov`. 25.07 saatlıq tək danışanlı azərbaycanca audio, müəllifin
icazəsi ilə, istinad şərti ilə istifadə olunub. **Bu modelin üzərində nəsə
qursanız, datasetə də istinad edin.**

Layihəyə və hər iki mənbəyə istinad edin — bax [CITATION.cff](CITATION.cff):

```bibtex
@software{azerbaijani_tts,
  title  = {Azerbaijani TTS: an offline 9.36M-parameter VITS model},
  author = {Huseynli, Ilqar},
  year   = {2026},
  url    = {https://github.com/HuseynliIlqar/Inflect_Micro_v2_Azerbaijan},
  note   = {Adapted from owensong/Inflect-Micro-v2 (Apache-2.0);
            trained on ughurabbasov/azerbaijani-tts-dataset,
            used with the author's permission}
}
```
