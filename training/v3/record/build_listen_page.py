"""Build a local listening page for a recording run: every take side by side with its scores,
the gate verdict and the chosen take, plus the per-category summary.

    python build_listen_page.py RUN_DIR [--per-category 8]

RUN_DIR holds takes/<take>/<id>.wav, scores/<take>.jsonl, gates.json and
<selection>/selection.jsonl + summary.json (pilot_selection or selection). Writes
RUN_DIR/listen.html with relative audio paths, so it opens straight from disk.
"""
from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
from common import RISING, Gates, rejection  # noqa: E402

STYLE = """
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1d1b;--muted:#66635c;--line:#e2ded5;--ok:#1f6b52;--bad:#a3361f;--pick:#24507a}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--card:#1f1f1c;--ink:#ecebe6;--muted:#a29f97;--line:#34332e;--ok:#62c49f;--bad:#ef8a72;--pick:#8fb8e3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;overflow-wrap:break-word}
main{max-width:1000px;margin:0 auto;padding:20px 16px 48px}h1{font-size:1.5rem;margin:0 0 6px}h2{font-size:1.15rem;margin:26px 0 8px}
.note{color:var(--muted);font-size:.88rem}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:10px 0}
.takes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:8px}.take{border:2px solid transparent;border-radius:8px;padding:6px}
.take.pick{border-color:var(--pick)}audio{width:100%}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.85rem}th,td{border:1px solid var(--line);padding:5px 7px;text-align:left}
@media (max-width:700px){.takes{grid-template-columns:1fr}}
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--selection", default="pilot_selection")
    args = parser.parse_args()
    run = args.run
    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    gates = Gates.from_dict(json.loads((run / "gates.json").read_text())["gates"])
    summary = json.loads((run / args.selection / "summary.json").read_text(encoding="utf-8"))
    takes = summary["takes"]
    question_takes = summary.get("question_takes", takes)
    scores = {t: {s["id"]: s for s in map(json.loads, (run / "scores" / f"{t}.jsonl").read_text(encoding="utf-8").splitlines())}
              for t in takes}
    selection = [json.loads(line) for line in (run / args.selection / "selection.jsonl").read_text(encoding="utf-8").splitlines()]
    by_cat = defaultdict(list)
    for item in selection:
        by_cat[item["cat"]].append(item)

    parts = [f"<!doctype html><html lang='az'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Dataset v2 dinləmə</title><style>{STYLE}</style></head><body><main>",
             "<h1>Dataset v2: versiyaların müqayisəsi</h1>",
             f"<p class='note'>Versiyalar: {', '.join(takes)}. Mavi çərçivə seçilən versiyadır. Hədlər: {esc(gates.__dict__)}</p>",
             "<div class='scroll'><table><tr><th>Kateqoriya</th><th>Cümlə</th><th>Saxlanıldı</th><th>Atılma səbəbləri</th></tr>"]
    for cat, info in summary["per_category"].items():
        parts.append(f"<tr><td>{esc(cat)}</td><td>{info['sentences']}</td><td>{info['kept_share']:.0%}</td>"
                     f"<td>{esc(info['drop_reasons'])}</td></tr>")
    parts.append("</table></div>")
    parts.append(f"<p class='note'>Versiyaların keçmə faizi: {esc(summary['take_pass_rate'])}; seçilənlər: {esc(summary['chosen_take_share'])}; "
                 f"meyl qoruması (median, seçilən / hamısı): {esc(summary['bias_guard'])}</p>")
    rng = random.Random(1)
    for cat in sorted(by_cat):
        items = by_cat[cat]
        shown = rng.sample(items, min(args.per_category, len(items)))
        parts.append(f"<h2>{esc(cat)}</h2>")
        for item in shown:
            text = rows[item["id"]]["text"]
            parts.append(f"<div class='card'><b>{esc(text)}</b> <span class='note'>{esc(item['id'])}</span><div class='takes'>")
            for take in takes:
                score = scores[take].get(item["id"])
                if score is None:
                    parts.append(f"<div class='take'><b>{esc(take)}</b><div class='note'>yoxdur</div></div>")
                    continue
                reason = rejection(cat, text, score, gates)
                if cat in RISING and take not in question_takes:
                    verdict = "<span class='note'>suallarda istifadə olunmur</span>"
                else:
                    verdict = "<span class='ok'>keçdi</span>" if reason is None else f"<span class='bad'>{esc(reason)}</span>"
                drop = score.get("end_drop_st", float("nan"))
                rise = f" · qalxıb-enmə {drop:.1f} st" if cat in RISING and drop == drop else ""
                pick = " pick" if item.get("take") == take else ""
                parts.append(
                    f"<div class='take{pick}'><b>{esc(take)}</b> {verdict}"
                    f"<audio controls preload='none' src='takes/{esc(take)}/{esc(item['id'])}.wav'></audio>"
                    f"<div class='note'>CER {score['cer']:.3f}{rise} · cızırtı {score['buzz']:.1f} · zil {score['ring']:.0%} · "
                    f"tembr {score['similarity']:.2f} · genişlik {score['range_st']:.1f} st · {score['seconds']:.1f} s</div>"
                    f"<div class='note'>Whisper: {esc(score['hyp'])}</div></div>")
            parts.append("</div></div>")
    parts.append("<script>document.addEventListener('play',e=>{document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause()})},true)</script>")
    parts.append("</main></body></html>")
    (run / "listen.html").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", run / "listen.html")


if __name__ == "__main__":
    main()
