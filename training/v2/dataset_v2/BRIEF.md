# Brief for writing dataset-v2 sentences (Azerbaijani)

## Why
Avaz is a small Azerbaijani TTS model. A large teacher TTS will read every sentence you write, and Avaz will then train on those recordings. A deep test (`D:\ktts\avaz-train\results\HESABAT.html`) measured these weaknesses:
- Yes/no questions end with falling pitch, while the teacher rises.
- Pitch range is narrow, and exclamations are rare in the data.
- Consonants get confused, mostly m→n, b→d, q→k, p and h. This happens most at the first and last word of a sentence and in short sentences. m→n also changes meaning: "Valideynlərim" → "Valideynlərin".
- Non-final stress is only half learned (dünən, bəli, gəlmə…).
- There is no paragraph-level prosody.

Your sentences are the cure. Each category below targets one weakness, so follow your category's focus closely.

## Language rules (all categories)
- Write natural, modern, standard Azerbaijani in Latin script, with correct spelling and all letters (ə ı ö ü ş ç ğ İ).
  - Write it the way a native speaker talks or a good modern book reads.
  - No translationese, no bureaucratic language, no archaic or Arabic-Persian classical style.
- Grammar must be correct: vowel harmony, case endings, personal endings.
  - If you are unsure about a form, choose a simpler sentence.
- Do not use these:
  - digits: write numbers as words;
  - abbreviations or all-caps acronyms;
  - e-mails or links;
  - words in a foreign Latin spelling: write foreign names the Azerbaijani way, e.g. Vaşinqton, Şekspir;
  - the letter w;
  - parentheses;
  - profanity;
  - claims about real living people.
- Allowed punctuation: , . ! ? : ; — « » " …
- A sentence starts with a capital letter and ends with . ! ? or …
- Vary everything:
  - topics: family, home, children, school, university, work, city, village, nature, weather, seasons, travel, food, health, sport, music, cinema, books, technology, history, science, crafts, animals, feelings, friendship, holidays;
  - names, both male and female;
  - sentence openings, verbs and tenses.
- Do not repeat a template with one word swapped. Every line must be clearly different.
- Do not copy sentences from anywhere. Write new ones.

## Output format
JSON Lines, one object per line: `{"cat": "<your code>", "text": "<sentence>"}`. Use `ensure_ascii=False`, which means the Azerbaijani letters are written as they are.
- Write into `D:\ktts\avaz-train\dataset_v2\out\<CODE>_01.jsonl`, `<CODE>_02.jsonl`, … with at most 150 lines per file.
- Use the Write tool for each new file. Never overwrite another code's files.
- After every 2–3 files, run the checker and fix every line it reports:
  `D:/ktts/avaz-train/.venv/Scripts/python.exe D:/ktts/avaz-train/dataset_v2/check_sentences.py "D:/ktts/avaz-train/dataset_v2/out/<CODE>_*.jsonl"`
  (in Bash, prefix with `PYTHONIOENCODING=utf-8`).
  - It flags format problems, word counts outside your range, and duplicates against the existing dataset, the test set and your own lines.
  - It also prints the "new-word share", which is the share of words the model never saw in training.
- When you have finished, run the checker one last time.
- Your final reply to me should be short:
  - how many valid lines you wrote, and in which files;
  - the checker's last summary line;
  - 5 example lines.

## Word-count limits used by the checker
| Category | Words |
|---|---|
| default | 2–25 |
| A3 | 8–25 |
| G1, G2 (paragraphs) | 25–60 |
| L | 16–35 |

## Categories
- **A1 — yes/no questions, everyday life** (home, family, food, shopping, neighbours, children, guests). Use the particle -mı/-mi/-mu/-mü in many forms:
  - on verbs in all tenses: gəldinmi, gedirsənmi, görmüsənmi, gələcəkmi, bilirsinizmi, ala bilərəmmi, getməliyəmmi;
  - on nouns and adjectives: evdədirmi, hazırdırmı, doğrudurmu;
  - varmı / yoxmu;
  - about 15% tag questions: "…, elə deyilmi?", "…, deyilmi?".
  - Length 2–12 words, with many short ones.
- **A2 — yes/no questions, the wider world** (work, school, city, transport, travel, health, sport, technology, weather, nature, culture). Same particle variety, plus about 25% questions **without** a particle, which are questions by intonation only: "Sən də gəlirsən?", "Qatar artıq gəlib?". Length 2–12 words.
- **A3 — long yes/no questions** (8–25 words): interview and discussion style, questions with subordinate clauses or lists, and the particle late in the sentence ("…yarışların gənclər arasında idmana marağı artıracağını düşünürsünüzmü?"). Mix polite (siz) and informal (sən).
- **B — other questions:**
  - about 1 000 question-word questions (kim, kimə, kimin, nə, nəyi, nəyə, nədən, harada, haraya, haradan, nə vaxt, niyə, nə üçün, necə, hansı, neçə, nə qədər), each word in many positions;
  - about 300 alternative questions with "…, yoxsa …?";
  - about 200 surprise or rhetorical questions ("Doğrudan?", "Bu nə deməkdir axı?", "Məgər sən bilmirdin?").
  - Length 2–15 words.
- **C — exclamations** (2–15 words). Spread the emotions evenly:
  - joy, surprise, admiration, warning, urgent command, congratulation and wishes, greeting, regret, anger and annoyance, relief, calling out to someone;
  - with interjections (Vay, Of, Ah, Ay aman, Ura, Afərin, Eh, Hey, Bəh-bəh) and without.
- **D1 — -m vs -n in verbs.** First- and second-person endings in all tenses, often both in one sentence ("Mən gəldim, sən gəlmədin."). Use positive and negative forms, and modals (-malıyam/-malısan, -a bilərəm/-a bilərsən).
  - At least half of the lines must END with a first-person word ending in -m ("…heç nə bilmirəm.", "…sabah zəng edərəm.").
  - Many short sentences.
- **D2 — -m vs -n in possessives and names.** Possessive -ım/-im/-um/-üm against -ın/-in/-un/-ün, including plural (-ımız/-ınız) and with case endings: evimdə / evində, kitabımı / kitabını, atamdan / atandan.
  - Use names ending in -m: Kərim, Rüstəm, Məryəm, Ədhəm, İbrahim, Səlim, Nəzim, Həkim, Xanım…
  - Use foreign names like Tom, Adam, Sem in Azerbaijani spelling.
  - Many lines should end with an -m word.
- **E — consonants at sentence edges, and short sentences.**
  - About 1 200 sentences of 2–6 words that begin and end with words starting with b, p, m, q, d, t, h, g, k ("Bağban budaqları budadı.", "Qonaqlar qapıda gözləyir.").
  - About 300 sentences with hard clusters and doubles: nitq, mətn, fikr, sirr, hiss, haqq, təbrik, təəssüf, camaat, saat, məəttəl, maaş.
  - About 300 lines of category **I**, with "cat": "I": numbers written as words, especially ordinals ("beşinci", "iyirmi dördüncü il", "üçüncü mərtəbə", "on doqquzuncu əsr"), dates, times ("saat yeddinin yarısında"), fractions ("üç tam onda beş") and percentages ("qırx faiz").
- **F — words with non-final stress.**
  - Each sentence (3–15 words) contains 1–3 words from this set:
    - adverbs and particles: dünən, indi, sonra, bəli, xeyr, bəlkə, əlbəttə, axı, lakin, amma, çünki, yalnız, yəqin, əslində, məgər, hələlik;
    - negative imperatives: gəlmə, getmə, yazma, alma, unutma, danışma, qorxma, tələsmə;
    - question words: nə vaxt, haraya;
    - vocatives: Ana!, Qızım!, Oğlum!;
    - place names like Ankara or Moskva.
  - Mix positions: start, middle, end.
- **G1 — narrative paragraphs** (fiction or audiobook style, 25–60 words, 2–4 sentences). Each is a small scene with a clear progression and some dialogue inside the narration, introduced with a colon, e.g. "Ana soruşdu: Niyə gecikdin?". Vary the mood: calm, tense, funny, sad, joyful.
- **G2 — informative paragraphs** (25–60 words, 2–4 sentences): history, geography, science, nature, culture, health, technology, how-to. The sentences of a paragraph must belong together, like a passage from a book or an audio guide.
- **L — long sentences** (16–35 words) with natural rhythm, like an audiobook or the news:
  - subordinate clauses, participial and adverbial phrases;
  - enumerations with commas;
  - asides set off with a dash "—";
  - "həm…, həm də…", "nəinki…, hətta…", "…-dıqca", "…-anda".
  - Mix narrative and informative.
- **J — new vocabulary** (6–15 words). Sentences rich in words the model has never seen.
  - Draw on specialised but modern areas: medicine, law, agriculture, crafts, cooking, botany, zoology, geology, astronomy, music theory, architecture, sports terms, finance, psychology, technology, sewing, carpentry, fishing, beekeeping.
  - Each sentence should hold at least two uncommon words.
  - Aim for a new-word share of at least 25% in the checker.
  - The sentences must still be natural, and every word must be real, correct Azerbaijani.
