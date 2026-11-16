"""Fill the generated parts of results/HESABAT.html from the downloaded measurements:
the audio examples (<!--EXAMPLES-->) and the hard-text table (<!--HARD-TABLE-->).

    python build_report_sections.py D:/ktts/avaz-train/results
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

ERROR_EXAMPLES = [
    (319, "m→n mənanı dəyişir: \"Valideynlərim\" (mənim) → \"Valideynlərin\" (sənin)"),
    (27, "son samit m→n: \"taparam\" → \"taparan\""),
    (293, "\"Tom\" → \"Tol\" (bu ad train-də 1 170 dəfə var)"),
    (4, "söz başında b→d: \"bəzi\" → \"dəzi\""),
    (140, "söz başında q→t: \"qəbul\" → \"təbul\""),
    (15, "q→k: \"Qondu\" → \"Kondu\""),
]
QUESTION_EXAMPLES = [64, 66, 18]
VERSIONS = [("Müəllim", "asr_eval/examples/data_{i:04d}.wav", "dataset"),
            ("Köhnə Avaz", "asr_eval/base/{i:04d}.wav", "base"),
            ("Yeni (0.667)", "asr_eval/step-00030000/{i:04d}.wav", "step-00030000"),
            ("Yeni (0.333)", "asr_eval/exp-var-0.33/{i:04d}.wav", "exp-var-0.33")]


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def load(root: Path):
    rows = json.loads((root / "asr_eval/asr_rows.json").read_text(encoding="utf-8"))
    hyps033 = json.loads((root / "asr_eval/exp-var-0.33/hyps.json").read_text(encoding="utf-8"))
    voice = json.loads((root / "asr_eval/voice_rows.json").read_text(encoding="utf-8"))
    variants = json.loads((root / "asr_eval/variants/voice_rows_variants.json").read_text(encoding="utf-8"))
    finals = {(r["version"], r["index"]): r.get("final_st") for r in voice + variants}
    return rows, hyps033, finals


def players(i: int, heard: dict, extra: dict | None = None) -> str:
    cells = []
    for label, pattern, key in VERSIONS:
        note = f"<div class='heard'>Whisper: {esc(heard[key])}</div>" if heard.get(key) else ""
        if extra and extra.get(key) is not None:
            note += f"<div class='heard'>Cümlə sonu: {extra[key]:+.1f} yarımton</div>"
        cells.append(f"<div><div class='lab'>{label}</div><audio controls preload='none' src='{pattern.format(i=i)}'></audio>{note}</div>")
    return "<div class='ex ex4'>" + "".join(cells) + "</div>"


def examples(root: Path) -> str:
    rows, hyps033, finals = load(root)
    parts = ["<p>Hər nümunədə eyni cümlə dörd versiyada səslənir. Altında Whisper-in eşitdiyi mətn göstərilir. "
             "Qulaqla da yoxlamaq üçündür: Whisper-in səhv saydığı yerləri dinlə.</p>", "<h3>Tələffüz səhvləri</h3>"]
    for i, note in ERROR_EXAMPLES:
        heard = {"dataset": rows[i]["dataset"], "base": rows[i]["base"], "step-00030000": rows[i]["step-00030000"],
                 "exp-var-0.33": hyps033.get(str(i), "")}
        parts.append(f"<div class='card'><b>{esc(rows[i]['text'])}</b><div class='note'>{esc(note)}</div>{players(i, heard)}</div>")
    parts.append("<h3>Bəli/xeyr sualları: sonda ton qalxmalıdır</h3>")
    for i in QUESTION_EXAMPLES:
        extra = {"dataset": finals.get(("dataset", i)), "base": finals.get(("base", i)),
                 "step-00030000": finals.get(("step-00030000", i)), "exp-var-0.33": finals.get(("exp-var-0.33", i))}
        parts.append(f"<div class='card'><b>{esc(rows[i]['text'])}</b>{players(i, {}, extra)}</div>")
    return "\n".join(parts)


def hard_table(root: Path) -> str:
    results = json.loads((root / "hardtext/hardtext.json").read_text(encoding="utf-8"))
    head = ("<div class='scroll'><table><tr><th>#</th><th>Növ</th><th>Mətn</th><th>Yeni paket: normallaşdırma</th>"
            "<th>Yeni: Whisper eşitdi</th><th>Köhnə Avaz: Whisper eşitdi</th><th>Dinlə</th></tr>")
    body = []
    for n, row in enumerate(results):
        new, old = row.get("new", {}), row.get("base", {})
        norm = new.get("normalized", new.get("error", ""))
        norm_cell = esc(norm) if norm.strip() != row["text"].strip() else "<span class='note'>dəyişməyib</span>"
        audio = "".join(
            f"<audio controls preload='none' src='hardtext/{name}/{n:02d}.wav'></audio>"
            if (root / "hardtext" / name / f"{n:02d}.wav").exists()
            else f"<div class='note bad'>{label} bu cümləni oxuya bilmədi (çökdü)</div>"
            for name, label in (("new", "Yeni paket"), ("base", "Köhnə Avaz")))
        body.append(f"<tr><td class='n'>{n + 1}</td><td>{esc(row['cat'])}</td><td>{esc(row['text'])}</td><td>{norm_cell}</td>"
                    f"<td>{esc(new.get('hyp', ''))}</td><td>{esc(old.get('hyp', ''))}</td><td style='min-width:180px'>{audio}</td></tr>")
    return head + "".join(body) + "</table></div><p class='note'>Dinlə sütununda birinci pleyer yeni paket, ikincisi köhnə Avaz-dır.</p>"


def main() -> None:
    root = Path(sys.argv[1])
    page = root / "HESABAT.html"
    text = page.read_text(encoding="utf-8")
    text = text.replace("<!--EXAMPLES-->", examples(root))
    text = text.replace("<!--HARD-TABLE-->", hard_table(root))
    page.write_text(text, encoding="utf-8")
    print("filled", page)


if __name__ == "__main__":
    main()
