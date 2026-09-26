# Azərbaycanca TTS

Azərbaycan dilində danışan **tam oflayn** mətn→səs modeli. 9.36M parametr,
24 kHz mono, adi noutbukun prosessorunda **real vaxtdan 2–4 dəfə sürətli**
işləyir. Server, API açarı, internet — heç biri lazım deyil.

> For the English documentation see [README.md](README.md).

```bash
python say.py "Salam, necəsiniz?"
```

Səs `out/01.wav` faylına düşür. Vəssalam.

---

## Quraşdırma

Python 3.11 və ya daha yenisi lazımdır. Repo model çəkilərini
[Git LFS](https://git-lfs.com) ilə saxlayır, ona görə klonlamadan əvvəl onu
qurun:

```bash
git lfs install
git clone https://github.com/<istifadeci>/azerbaycan-tts
cd azerbaycan-tts
```

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe say.py "Salam dünya."
```

```bash
# Linux / macOS
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python say.py "Salam dünya."
```

PyTorch böyükdür. Yalnız prosessorda işlədəcəksinizsə CPU təkərini seçin —
xeyli kiçik olur:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Model çəkiləri (`model/`, 37 MB) qovluğun içindədir — ayrıca yükləmək lazım deyil.

## İstifadə

### Komanda sətri

```bash
python say.py "Bir cümlə."                       # → out/01.wav
python say.py "Birinci." "İkinci."               # → out/01.wav, out/02.wav
python say.py -f kitab.txt --one-file -o out/kitab   # bütün fayl bir WAV-a
python say.py --show-text "II Dünya, 25% artım"  # modelə nə getdiyini göstər
python say.py --speed 0.85 --seed 42 "Daha yavaş."
```

| Parametr | Default | Nə edir |
| --- | ---: | --- |
| `--speed` | `1.0` | 0.5–2.0. Aşağı = yavaş |
| `--variation` | `0.667` | 0.0–1.0. Aşağı = sabit, yuxarı = daha canlı |
| `--seed` | `7` | Eyni seed = eyni səs. Bəyənmədiyiniz cümlə üçün dəyişdirin |
| `--device` | `cpu` | `cpu` və ya `cuda` |
| `--out` | `out/` | Çıxış qovluğu |
| `--raw` | — | Normallaşdırmanı söndürür (aşağıya bax) |
| `--max-words` | `15` | Hissə uzunluğu. `0` = bölməni modelə burax |
| `--prosody` | `off` | `safe` / `wide` — vurğuları seyrəldir (təcrübi) |
| `--show-text` | — | Normallaşdırılmış mətni və hissə sərhədlərini çap edir |
| `--voice` | `az` | `en` ingiliscə baza modellə danışır (aşağıya bax) |

### Brauzer interfeysi

Modeli komanda sətri olmadan sınamaq üçün səhifə: cümləni yazın, parametrləri
tənzimləyin, dinləyin, endirin. Gradio tələb edir — əsas kitabxana bilərəkdən
ondan asılı deyil, ona görə yalnız səhifə lazımdırsa quraşdırın.

```bash
pip install -r requirements-app.txt     # və ya: pip install -e ".[app]"
python app.py                           # http://127.0.0.1:7860
```

İnterfeys azərbaycanca və ingiliscə arasında keçid edir, CLI-dəki bütün
parametrləri verir və komanda sətrinin gizlətdiyi iki şeyi göstərir: modelə
həqiqətən çatan normallaşdırılmış mətni və seçdiyiniz ayarlara uyğun `say.py`
əmrini — yəni səhifə həm də CLI-ni öyrənmək üçün işləyir.

`Səsi təmizlə` vokoderin metal rezonansını hazır kliplə süzür. Bu,
`tools/audio_postprocess.py`-ın tək-klip variantıdır, ffmpeg tələb etmir və
hansı tezlikləri sildiyini bildirir.

### İngilis səsi

Həm səhifə, həm də CLI `owensong/Inflect-Micro-v2` ilə — bu modelin
uyğunlaşdırıldığı ingilis checkpoint-i ilə — danışa bilir. O, `model-en/`
qovluğunda gəlir, yəni klon heç nə yükləmədən hər iki dildə danışır:

```bash
python say.py --voice en "Hello there."
```

Bu, başqa müəllifin başqa səsidir, Apache-2.0-dır və ona necə istinad
ediləcəyi `model-en/README.md`-də yazılıb. Çəkilər azərbaycanca model kimi Git
LFS ilə gəlir — `git lfs install` edilmədən klonlansa, fayl əvəzinə göstərici
düşür; həm CLI, həm də səhifə bunu açıq deyir, torch-un içində sınmır.

Azərbaycanca mətn qatları ona aid deyil: `normalize_az` rəqəmləri azərbaycanca
sözlərə çevirir, vurğu qatı isə azərbaycanca üçün köklənib — ona görə ingiliscə
üçün hər ikisi söndürülür və interfeysdə passivləşir. İngilis mətni eSpeak-in
`en-us` səsi ilə fonemlərə çevrilib runtime-a verilir.

Bu layihənin öz modeli ingiliscə danışmır. O, 200.000 addım boyu tək bir
azərbaycanca səslə uyğunlaşdırılıb; ingilis mətni ona verilsə, çıxan şey həmin
səsin tanımadığı fonemləri oxumasıdır.

### Python-dan

```python
from aztts import AzTTS

tts = AzTTS()                       # bir dəfə yaradın, təkrar işlədin
tts.save("Mətn burada.", "out/a.wav")

waveform = tts.synthesize("Xam massiv lazımdırsa.", speed=1.1, seed=3)
# → float32 numpy massiv, mono, [-1, 1], 24 000 Hz
```

Mətn funksiyaları modeldən asılı deyil, ayrıca da işlədilə bilər:

```python
from aztts import normalize_az, chunk_text

normalize_az("II qrupda 25% artım oldu.")
# 'İkinci qrupda iyirmi beş faiz artım oldu.'

chunk_text("Uzun bir cümlə...", max_words=15)
# ('Uzun bir cümlə...',)
```

## Mətn modelə çatmazdan əvvəl nə olur

Model adi mətn üzərində öyrədilib, ona görə rəqəm və qısaltmalar ona xam
şəkildə çatsa səhv oxunur — `II` "ı ı" kimi səslənir, `25%` faiz işarəsini
tamam itirir. İki addım bunun qarşısını alır:

**1. Normallaşdırma** (`aztts/az_text.py`) — rəqəmlər, Roma rəqəmləri, tarixlər,
faizlər, ölçü vahidləri və qısaltmalar danışılan sözə çevrilir:

| Giriş | Modelə gedən |
| --- | --- |
| `II Dünya müharibəsi` | `İkinci Dünya müharibəsi` |
| `01/09/1939` | `birinci sentyabr min doqquz yüz otuz doqquz` |
| `25%` | `iyirmi beş faiz` |
| `5 kq` | `beş kiloqram` |

**2. Hissələrə bölmə** (`aztts/az_chunk.py`) — cümlələr ~15 sözlük hissələrə
bölünür. Model kiçikdir və uzun cümlənin sonunda intonasiya düzləşir, söz
sonluqları kəsilir. Hissələr arasına durğu işarəsinin uzunluğuna uyğun pauza
qoyulur, ona görə qırıq səslənmir.

Nə baş verdiyini görmək üçün `--show-text` verin. Söndürmək üçün `--raw`.

## Nümunələr

Hazır WAV faylları, hər səs üçün bir dəst. Heç nə quraşdırmadan əvvəl dinləyin.

**Azərbaycanca** — bu layihənin modeli, `samples/`:

| Fayl | Cümlə |
| --- | --- |
| [`01-salam.wav`](samples/01-salam.wav) | `Salam, bu model tamamilə yerli maşında işləyir.` |
| [`02-payiz.wav`](samples/02-payiz.wav) | `Payız gəlmişdi və şəhərin küçələri saralmış yarpaqlarla örtülmüşdü.` |
| [`03-sual.wav`](samples/03-sual.wav) | `Sən bu kitabı oxumusan? Mənə çox maraqlı gəldi.` |
| [`04-reqem.wav`](samples/04-reqem.wav) | `II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.` |
| [`05-uzun.wav`](samples/05-uzun.wav) | Uzun cümlə — intonasiyanın harada yastılaşdığını eşitmək üçün |

**İngiliscə** — baza model, `samples/en/`:

| Fayl | Cümlə |
| --- | --- |
| [`01-hello.wav`](samples/en/01-hello.wav) | `Hello, this model runs completely offline on your machine.` |
| [`02-autumn.wav`](samples/en/02-autumn.wav) | `Autumn had come, and the streets were covered with yellow leaves.` |
| [`03-question.wav`](samples/en/03-question.wav) | `Have you read this book? I found it very interesting.` |

Bunlar fərqli müəlliflərin fərqli səsləridir. Azərbaycanca olan bu layihənindir;
ingiliscə olan isə `owensong/Inflect-Micro-v2`-dir, dəyişdirilməmiş.

Yenidən yaratmaq:

```bash
python say.py --out samples                 # azərbaycanca
python say.py --voice en --out samples/en   # ingiliscə
```

Hər iki dəst Git LFS-dən keçir, ona görə yalnız model həqiqətən dəyişəndə
yenidən yaradın.

## Keyfiyyət alətləri

```bash
# Bir cümləni 10 fərqli seed ilə səsləndir, ən yaxşısını seç
python tools/seed_sweep.py --seeds 10 "Xoş gəlmisiniz."

# Vokoderin metallik rezonansını tap, sonra ffmpeg ilə təmizlə
python tools/audio_postprocess.py analyze samples
python tools/audio_postprocess.py clean out --lowpass 11000
```

Tez-tez təkrarlanan cümlələr üçün (salamlama, xəta mesajı) seed-i bir dəfə
seçib koda sabitləmək oxunuşu nəzərəçarpacaq yaxşılaşdırır.

## Test

```bash
python -m pytest
```

227 test:

| | |
| --- | ---: |
| Mətn normallaşdırma, hissələrə bölmə, vurğu qatı | 116 |
| Sürətli monotonic alignment (train optimizasiyası) | 25 |
| İnterfeys etiketləri, parametrlər, rezonans filtri | 86 |

Testlər modeli yükləmir — 3 saniyədə bitir.

## Struktur

```
say.py            Komanda sətri — əsas giriş nöqtəsi
app.py            Brauzer interfeysi (Gradio) — könüllü
aztts/
  engine.py       AzTTS — model yüklənməsi, normallaşdırma, bölmə, sintez
  az_text.py      Rəqəm / Roma rəqəmi / qısaltma → söz
  az_chunk.py     Cümlələri modelin bacardığı uzunluğa bölür
  az_prosody.py   Vurğu qatı (könüllü, --prosody)
  en_voice.py     İngilis baza modeli, eyni runtime ilə
  console.py      Windows konsolunu UTF-8-ə keçirir
webui/
  i18n.py         İnterfeys etiketləri, azərbaycanca və ingiliscə
  runner.py       Parametrlər, CLI qarşılığı, bir sintez
  cleanup.py      Tək-klip rezonans filtri (ffmpeg-siz)
model/            Azərbaycanca çəkilər + runtime (37 MB) — toxunmayın
model-en/         İngilis baza modelin çəkiləri (37 MB) — toxunmayın
tools/            seed_sweep, audio_postprocess — keyfiyyət alətləri
training/         Modelin necə hazırlandığı (baza model yüklənir)
packaging/        GitHub / Hugging Face / Kaggle yayımı
samples/          Nümunə səslər
tests/            227 test
out/              Yaratdığınız WAV faylları
```

Hər üç platformaya yayım addımları:
[packaging/PUBLISHING.md](packaging/PUBLISHING.md). Kod üzərində işləyəcəksinizsə
— süni intellektlə və ya onsuz — [CLAUDE.md](CLAUDE.md) faylından başlayın:
orada qurulum, yoxlama əmrləri və mənbəyi oxumaqla görünməyən tələlər yazılıb.
Cursor eyni qaydaları `.cursor/rules/` qovluğundan avtomatik götürür.

`model/` upstream export paketidir və özü ilə birlikdə lazım olan hər şeyi
gətirir (çəkilər, fonem frontend-i, VITS runtime-ı). Bütün öz kodumuz
`aztts/` altındadır.

## Model haqqında

| | |
| --- | --- |
| Baza | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Parametr | 9.36M |
| Səs | 24 kHz mono, tək danışan |
| Train | 200,000 addım = 1,343 epoch, 25 saat azərbaycanca səs |
| Sürət | CPU-da 2–4× real vaxt |

Ətraflı — train prosesi, dataset, ölçmələr və məhdudiyyətlər:
[docs/MODEL.md](docs/MODEL.md).

## Model necə hazırlandı

[`training/`](training/README.md) qovluğu bütün prosesi saxlayır: rəsmi
toolkit-ə tətbiq olunan patch, presetlər, train nəticələri və loss tarixçəsi.

Ən vacib hissəsi — toolkit-in `maximum_path` funksiyası saf Python döngəsi idi
və addımın 8.7 saniyəsini yeyirdi. Vektorlaşdırılmış torch versiyası onu
**1.395 saniyəyə** endirdi: 20 gün → 3.2 gün, ~$213 → ~$34. Nəticənin orijinalla
**bit-eyni** olduğu 25 testlə sübut olunub və o testlər indi də işləyir.

İngiliscə baza model (adaptasiyanın başlanğıc nöqtəsi) bu repoda saxlanmır —
o, 38 MB-lıq upstream snapshot-dır. Lazım olanda yükləyin:

```bash
pip install huggingface_hub
python training/scripts/download_model.py
python say.py --model training/base-model --raw --max-words 0 -o out/base "Hello, this is the English base model."
```

## Məhdudiyyətlər

- **Tək səs.** Səs klonlama və ya çoxsəslilik yoxdur.
- **Emosiya idarəsi yoxdur.** Oxunuş sakit və neytraldır.
- Model kiçikdir (9.36M) — nəticə aydın və başa düşüləndir, amma tam təbii
  deyil. Vokoder bəzən metallik rezonans verir.
- Nadir adlar, xarici sözlər və yazılışı qeyri-adi olan sözlər fonem
  frontend-inə həssasdır.
- Normallaşdırma `say.py` və `AzTTS` daxilində avtomatik işləyir. `model/`
  paketini birbaşa çağırsanız, `normalize_az`-ı özünüz çağırmalısınız.

## Təşəkkür və istinad

Bu model iki nəfərin öz işini açıq paylaşdığına görə mövcuddur.

**Baza model — [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2)**,
müəllif Owen Song, Apache-2.0. Buradakı hər çəki onunkundan başlayıb: 410
tensorun 409-u bit-eyni köçürülüb və azərbaycan fonemləri onsuz da onun simvol
dəstində vardı. O checkpoint olmasaydı, bu iş günlərlə yox, aylarla train
tələb edərdi.

**Train datası — [`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)**,
müəllif `ughurabbasov`. 25.07 saat tək danışanlı azərbaycanca səs. Repoda
lisenziya faylı elan edilməyib; datasetin Hugging Face icma bölməsində birbaşa
soruşulduqda müəllif hər kəsin işlədə biləcəyini təsdiqləyib və istinad
edilməsini xahiş edib. **İstifadənin şərti istinaddır — bu model üzərində nəsə
qurursunuzsa, datasetə də istinad edin.**

Bu layihəni işlədirsinizsə, ona və hər iki mənbəyə istinad edin — bax
[CITATION.cff](CITATION.cff):

```bibtex
@software{azerbaijani_tts,
  title  = {Azerbaijani TTS: an offline 9.36M-parameter VITS model},
  author = {Huseynli, Ilqar},
  year   = {2026},
  url    = {https://github.com/<istifadeci>/azerbaycan-tts},
  note   = {Adapted from owensong/Inflect-Micro-v2 (Apache-2.0);
            trained on ughurabbasov/azerbaijani-tts-dataset,
            used with the author's permission}
}
```

## Kommersiya istifadəsi

Qısa cavab: **işlətmək üçün bəli, yaymaq üçün bir real şərt var.** Şərt modelə
yox, fonemləşdirmə hissəsinə aiddir.

| Hissə | Lisenziya | Kommersiya istifadəsi |
| --- | --- | --- |
| Bizim kod (`aztts/`, `say.py`, `app.py`, `webui/`) | Apache-2.0 | Bəli |
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

Modelin özü eSpeak-ə möhtac deyil. `model/config.json` faylında
`accepts_prephonemized_input: true` yazılıb — yəni GPL-dən uzaq qalmalı olan
məhsul mətni öz aləti ilə fonemlərə çevirib modelə verə bilər. Hər iki yol
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) faylında açıqlanıb.

### İki səs

**Azərbaycanca.**
[`ughurabbasov/azerbaijani-tts-dataset`](https://huggingface.co/datasets/ughurabbasov/azerbaijani-tts-dataset)
üzərində öyrədilib. Repozitoriyada lisenziya faylı yoxdur; HuggingFace icma
tabında birbaşa soruşulanda müəllif sərbəst istifadəyə icazə verdiyini bildirib
və ad çəkilməsini xahiş edib. Şərt budur: istinad. Audio çox güman ki
sintetikdir, şərtləri bilinməyən başqa bir TTS sistemi tərəfindən hazırlanıb —
[docs/MODEL.md](docs/MODEL.md).

**İngiliscə.** `owensong/Inflect-Micro-v2`, Apache-2.0, burada dəyişdirilmədən
yenidən yayımlanır. Model kartından kommersiya qərarı üçün vacib üç fakt:

- səs **sintetikdir**. Paket real danışanın səs korpusunu yaymır və səsi hər
  hansı real insanın kimliyi kimi təqdim etmir — yəni təmizlənməli şəxsiyyət və
  ya bənzərlik hüququ yoxdur;
- buraxılış **açıq-çəki**dir, açıq-data deyil: korpus yaradan konveyer və
  filtrləmə infrastrukturu qapalıdır, ona görə təlim datası ictimai
  materiallardan yoxlanıla bilməz. Uyğunluq prosesiniz bu auditi tələb edirsə,
  müəllif e-poçt vasitəsilə deployment sorğularını qəbul edir;
- məsuliyyətli istifadə bəyanatı istəyir ki, kontekst yanlış təəssürat yarada
  biləcəksə sintetik nitq açıqlansın və səs heç kimi təqlid etmək üçün
  işlədilməsin.

### Hər iki səsdə qadağan olan

Real bir insanı təqlid etmək, aldadıcı məzmun hazırlamaq və ya sintetik nitqi
əsl səsyazma kimi təqdim etmək. Bu, həm bu layihənin, həm də upstream-in
mövqeyidir.

Bu, lisenziyaların oxunuşudur, hüquqi məsləhət deyil. Məhsulunuz buna
söykənirsə, ixtisaslı birinə təsdiqlətdirin.

## Lisenziya

Öz kodumuz Apache-2.0-dır — [LICENSE](LICENSE).

**Runtime asılılıqları haqqında qeyd.** Fonemləşdirmə
[`phonemizer`](https://github.com/bootphon/phonemizer) (**GPL-3.0-or-later**) və
[eSpeak NG](https://github.com/espeak-ng/espeak-ng) (**GPL-3.0-or-later**)
üzərindən gedir; eSpeak-in paylaşılan kitabxanasını `espeakng-loader` təkərin
içinə yığır. Layihəni qurmaq və işlətmək buna görə dəyişmir, amma birləşmiş
məhsulu — paketlənmiş tətbiq, konteyner obrazı, binar fayl — **yenidən
paylayırsınızsa**, həmin komponentlərə GPL-3.0 şərtləri tətbiq olunur. Tam
mənzərə və bundan yayınma yolları:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Üçüncü tərəf komponentlər (VITS, BigVGAN, alias-free-torch) və onların
lisenziyaları da həmin fayldadır.

**Train datası.** Dataset və onun işlədilmə şərtləri üçün yuxarıdakı
[Təşəkkür və istinad](#təşəkkür-və-istinad) bölməsinə baxın. Səsi doğuran
sistemin şərtləri naməlum qalır — bax [docs/MODEL.md](docs/MODEL.md).

Bu səsi real bir şəxsi təqlid etmək və ya aldadıcı məzmun yaratmaq üçün
işlətməyin.
