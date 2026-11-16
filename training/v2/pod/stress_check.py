"""Does the model put the stress where the phonemes mark it?

For every validation sentence the model's own alignment (frames per input token) tells which
audio frames belong to which vowel. In each word with two or more vowels and one primary stress
mark, the stressed vowel is compared with the others on length, loudness and pitch, and on all
three together (sum of within-word z-scores). Chance = 1 / number of vowels.

Versions: the old Avaz with its own eSpeak phonemes, the new checkpoint with the stress
frontend's phonemes (at two sampling noise levels). For words where the two frontends put the
stress on different vowels, it also checks which of the two the new model actually stresses.

    python stress_check.py   (training venv with praat-parselmouth)
Writes /workspace/asr_eval/stress_check.json.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
import parselmouth
import torch

sys.path.insert(0, "/workspace/base")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference  # noqa: E402  the package runtime (symbols, commons, frontend)
from inflect_vits_frontend import run_vits_frontend  # noqa: E402
from listen import load_weights  # noqa: E402

OUT = Path("/workspace/asr_eval")
CHECKPOINT = Path("/workspace/runs/polish/exports/model-step-00030000.pth")
VOWELS = frozenset("aæeiɯoœuyɑəɪʊɛɔøʏɐ")
STRESS, LONG = "ˈ", "ː"
HOP = 256


def symbol_ids() -> dict[str, int]:
    return {symbol: index for index, symbol in enumerate(inference.symbols)}


def words_of(chars: list[str]) -> list[dict]:
    """Group kept phoneme characters into words: vowel positions and the primary-stressed vowel."""
    words, current, pending_stress = [], {"vowels": [], "stressed": None}, False
    for position, ch in enumerate(chars):
        if ch == " " or (not ch.isalpha() and ch not in (STRESS, LONG, "ˌ")):
            if current["vowels"]:
                words.append(current)
            current, pending_stress = {"vowels": [], "stressed": None}, False
            continue
        if ch == STRESS:
            pending_stress = True
        elif ch in VOWELS:
            if pending_stress and current["stressed"] is None:
                current["stressed"] = len(current["vowels"])
            pending_stress = False
            current["vowels"].append([position])
        elif ch == LONG and current["vowels"] and current["vowels"][-1][-1] == position - 1:
            current["vowels"][-1].append(position)
    if current["vowels"]:
        words.append(current)
    return words


@torch.inference_mode()
def synthesize_aligned(engine, phonemes: str, ids: dict[str, int], variation: float):
    chars = [ch for ch in phonemes if ch in ids]
    tokens, lengths = engine._tokens(None, phonemes=phonemes)
    if tokens.size(1) != 2 * len(chars) + 1:
        raise ValueError("token count does not match the kept phoneme characters")
    torch.manual_seed(0)
    audio, attn, *_ = engine.model.infer(tokens, lengths, noise_scale=variation, noise_scale_w=0.8,
                                         length_scale=1.0, max_len=4000)
    frames = attn[0, 0].sum(0).cpu().numpy()  # frames per token
    starts = np.concatenate([[0], np.cumsum(frames)[:-1]])
    return chars, audio[0, 0].float().cpu().numpy(), frames, starts


def vowel_features(begin_token: int, end_token: int, frames, starts, audio, pitch_times, pitch_st, rate) -> tuple:
    """Length (ms), loudness (dB) and mean pitch (semitones) of the frames from begin_token to end_token inclusive."""
    begin, end = starts[begin_token], starts[end_token] + frames[end_token]
    length = float(end - begin) * HOP / rate * 1000
    segment = audio[int(begin * HOP):int(end * HOP)]
    loudness = float(20 * np.log10(np.sqrt(np.mean(segment ** 2)) + 1e-9)) if len(segment) else -120.0
    t0, t1 = begin * HOP / rate, end * HOP / rate
    voiced = pitch_st[(pitch_times >= t0) & (pitch_times < t1)]
    voiced = voiced[np.isfinite(voiced)]
    return length, loudness, float(np.mean(voiced)) if len(voiced) else np.nan


def vowel_span(chars: list[str], vowel: list[int]) -> tuple[int, int]:
    """Tokens of a vowel nucleus, the same rule for every vowel: from the blank before it (or before a stress
    mark sitting right in front of it, since the model may spend the stressed vowel's extra length on the mark)
    to the blank after it (after the length mark for long vowels)."""
    first, last = vowel[0], vowel[-1]
    if first > 0 and chars[first - 1] in (STRESS, "ˌ"):
        first -= 1
    return 2 * first, 2 * last + 2


def word_scores(words: list[dict], chars: list[str], frames, starts, audio, rate) -> list[dict]:
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=rate)
    pitch = sound.to_pitch_ac(time_step=0.005, pitch_floor=70, pitch_ceiling=450)
    hz = pitch.selected_array["frequency"]
    pitch_st = np.where(hz > 0, 12 * np.log2(np.maximum(hz, 1e-6) / 100), np.nan)
    pitch_times = pitch.xs()
    scored = []
    for number, word in enumerate(words):
        feats = [vowel_features(*vowel_span(chars, vowel), frames, starts, audio, pitch_times, pitch_st, rate)
                 for vowel in word["vowels"]]
        scored.append({"features": feats, "stressed": word["stressed"], "final_word": number == len(words) - 1})
    return scored


def prominence(features: list[tuple], columns=(0, 1, 2)) -> list[float]:
    arr = np.array(features, dtype=float)
    total = np.zeros(len(arr))
    for column in columns:
        values = arr[:, column]
        if np.isnan(values).any():
            continue
        spread = values.std()
        if spread > 0:
            total += (values - values.mean()) / spread
    return list(total)


def summarize(scored: list[dict]) -> dict:
    """Share of words whose marked vowel is the most prominent, and how far it stands out on average."""
    hits = {"length": [], "loudness": [], "pitch": [], "combined": [], "loud_pitch": [], "chance": []}
    gaps = {"length_ms": [], "loudness_db": [], "pitch_st": []}
    by_place = {"final_syllable": [], "earlier_syllable": []}
    for word in scored:
        feats, stressed = word["features"], word["stressed"]
        if stressed is None or len(feats) < 2 or word["final_word"]:
            continue
        for column, key, gap in ((0, "length", "length_ms"), (1, "loudness", "loudness_db"), (2, "pitch", "pitch_st")):
            values = [f[column] for f in feats]
            if not any(np.isnan(values)):
                hits[key].append(int(np.argmax(values) == stressed))
                others = [v for i, v in enumerate(values) if i != stressed]
                gaps[gap].append(values[stressed] - float(np.mean(others)))
        combined = int(np.argmax(prominence(feats)) == stressed)
        hits["combined"].append(combined)
        hits["loud_pitch"].append(int(np.argmax(prominence(feats, (1, 2))) == stressed))
        hits["chance"].append(1 / len(feats))
        by_place["final_syllable" if stressed == len(feats) - 1 else "earlier_syllable"].append(combined)
    result = {key: round(statistics.mean(values), 3) for key, values in hits.items() if values}
    result.update({f"stressed_minus_others_{k}": round(statistics.median(v), 2) for k, v in gaps.items() if v})
    result["words"] = len(hits["combined"])
    result.update({f"combined_{k}": round(statistics.mean(v), 3) for k, v in by_place.items() if v})
    result.update({f"words_{k}": len(v) for k, v in by_place.items()})
    return result


def disagreement(new_scored: list[list[dict]], base_words: list[list[dict]], columns=(0, 1, 2)) -> dict:
    """Words where eSpeak and the stress frontend mark different vowels: which one does the new model stress?"""
    follows_new = follows_old = other = 0
    for sentence_new, sentence_base in zip(new_scored, base_words):
        if len(sentence_new) != len(sentence_base):
            continue
        for new, old in zip(sentence_new, sentence_base):
            if (len(new["features"]) != len(old["vowels"]) or new["stressed"] is None or old["stressed"] is None
                    or new["stressed"] == old["stressed"] or new["final_word"]):
                continue
            top = int(np.argmax(prominence(new["features"], columns)))
            follows_new += top == new["stressed"]
            follows_old += top == old["stressed"]
            other += top not in (new["stressed"], old["stressed"])
    return {"words": follows_new + follows_old + other, "follows_new_marks": follows_new,
            "follows_espeak_marks": follows_old, "elsewhere": other}


def main() -> None:
    rows = json.loads((OUT / "rows.json").read_text(encoding="utf-8"))
    ids = symbol_ids()
    engine = inference.InflectTTS("/workspace/base", device="cuda")
    rate = engine.sample_rate
    base_phonemes = [run_vits_frontend(row["text"]).phoneme_text for row in rows]

    def run(phoneme_list: list[str], variation: float) -> tuple[list, list]:
        all_scored, all_words = [], []
        for phonemes in phoneme_list:
            chars, audio, frames, starts = synthesize_aligned(engine, phonemes, ids, variation)
            words = words_of(chars)
            all_words.append(words)
            all_scored.append(word_scores(words, chars, frames, starts, audio, rate))
        return all_scored, all_words

    base_scored, base_words = run(base_phonemes, 0.667)
    load_weights(engine, CHECKPOINT)
    new_phonemes = [row["phonemes"] for row in rows]
    report = {"base (eSpeak marks)": summarize([w for s in base_scored for w in s])}
    for variation in (0.667, 0.333):
        new_scored, new_words = run(new_phonemes, variation)
        report[f"new 30k, variation {variation}"] = summarize([w for s in new_scored for w in s])
        report[f"new 30k, variation {variation}: eSpeak vs stress frontend disagreements"] = disagreement(new_scored, base_words)
        report[f"new 30k, variation {variation}: disagreements, loudness+pitch only"] = disagreement(new_scored, base_words, (1, 2))
    marks = [w for s in new_words for w in s if len(w["vowels"]) >= 2 and w["stressed"] is not None]
    report["stress frontend: share of multi-vowel words stressed before the last syllable"] = round(
        sum(w["stressed"] < len(w["vowels"]) - 1 for w in marks) / len(marks), 3)
    (OUT / "stress_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
