"""A listening page to judge yes/no questions by ear: every take of a sample of A1-A3
sentences, with a "sounds like a proper question" / "does not" mark per take.

    python build_question_page.py RUN_DIR [--per-category 8]

RUN_DIR holds takes/<take>/<id>.wav and takes/question_contours.json. The page keeps the
marks in the browser and copies them as text ("id take +|-" lines) for the operator.
The pitch measures are hidden until the listener asks, so they do not steer the ear.
"""
from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
TAKES = ("A", "B", "C", "D", "E", "F")

STYLE = """
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d1d1b;--muted:#66635c;--line:#e2ded5;--ok:#1f6b52;--bad:#a3361f;--accent:#24507a}
@media (prefers-color-scheme:dark){:root{--bg:#151513;--card:#1f1f1c;--ink:#ecebe6;--muted:#a29f97;--line:#34332e;--ok:#62c49f;--bad:#ef8a72;--accent:#8fb8e3}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif;overflow-wrap:break-word}
main{max-width:1000px;margin:0 auto;padding:20px 16px 64px}h1{font-size:1.5rem;margin:0 0 6px}.note{color:var(--muted);font-size:.9rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
.takes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:8px}.take{border:1px solid var(--line);border-radius:8px;padding:8px}
audio{width:100%}.marks{display:flex;gap:6px;margin-top:6px}button{font:inherit;padding:6px 10px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}button[aria-pressed=true].yes{background:var(--ok);color:#fff;border-color:var(--ok)}
button[aria-pressed=true].no{background:var(--bad);color:#fff;border-color:var(--bad)}.metrics{display:none;font-size:.8rem;color:var(--muted);margin-top:4px}
body.show .metrics{display:block}.bar{position:sticky;bottom:0;background:var(--bg);padding:10px 0;display:flex;gap:10px;flex-wrap:wrap;align-items:center;border-top:1px solid var(--line)}
@media (max-width:700px){.takes{grid-template-columns:1fr 1fr}}@media (max-width:420px){.takes{grid-template-columns:1fr}}
"""

SCRIPT = """
const KEY = 'avaz-question-marks';
let marks = {};
try { marks = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { marks = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(marks)); } catch (e) {} count(); }
function count() { document.getElementById('count').textContent = Object.keys(marks).length + ' qiymət'; }
document.querySelectorAll('.marks').forEach(box => {
  const id = box.dataset.id;
  box.querySelectorAll('button').forEach(b => {
    b.setAttribute('aria-pressed', marks[id] === b.dataset.v ? 'true' : 'false');
    b.addEventListener('click', () => {
      marks[id] = marks[id] === b.dataset.v ? undefined : b.dataset.v;
      if (!marks[id]) delete marks[id];
      box.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', marks[id] === x.dataset.v ? 'true' : 'false'));
      save();
    });
  });
});
document.getElementById('show').addEventListener('click', e => {
  document.body.classList.toggle('show');
  e.target.textContent = document.body.classList.contains('show') ? 'Göstəriciləri gizlət' : 'Göstəriciləri göstər';
});
document.getElementById('copy').addEventListener('click', async () => {
  const text = Object.entries(marks).map(([k, v]) => k.replace('|', ' ') + ' ' + v).join('\\n');
  try { await navigator.clipboard.writeText(text); document.getElementById('status').textContent = 'Kopyalandı, çata yapışdır.'; }
  catch (e) { document.getElementById('out').value = text; document.getElementById('out').hidden = false; }
});
document.addEventListener('play', e => { document.querySelectorAll('audio').forEach(a => { if (a !== e.target) a.pause(); }); }, true);
count();
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--per-category", type=int, default=8)
    args = parser.parse_args()
    rows = {r["id"]: r for r in map(json.loads, (KIT / "sentences_v2.jsonl").read_text(encoding="utf-8").splitlines())}
    contours = json.loads((args.run / "takes" / "question_contours.json").read_text())
    by_id = defaultdict(dict)
    for c in contours:
        by_id[c["id"]][c["take"]] = c
    by_cat = defaultdict(list)
    for clip_id in by_id:
        by_cat[rows[clip_id]["cat"]].append(clip_id)
    rng = random.Random(3)
    parts = [f"<!doctype html><html lang='az'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Sual qiymətləndirməsi</title><style>{STYLE}</style></head><body><main>",
             "<h1>Bəli/xeyr sualları: hansı düzgün səslənir?</h1>",
             "<p class='note'>Hər sualın 6 versiyası var. Hər versiyanı dinlə və qiymət ver: <b>✓ düzgün sual</b>, yəni bir azərbaycanlı belə soruşar, "
             "və ya <b>✗ düz deyil</b>, yəni nəqli cümlə kimi səslənir və ya qəribədir. Bütün versiyaları qiymətləndirmək məcburi deyil, "
             "amma nə qədər çox olsa, süzgəc bir o qədər dəqiq olar. Qiymətlər brauzerdə saxlanılır. Sonda <b>Nəticəni kopyala</b> düyməsini bas və mətni çata yapışdır.</p>"]
    for cat in sorted(by_cat):
        for clip_id in sorted(rng.sample(by_cat[cat], min(args.per_category, len(by_cat[cat])))):
            parts.append(f"<div class='card'><b>{esc(rows[clip_id]['text'])}</b> <span class='note'>{esc(cat)} · {esc(clip_id)}</span><div class='takes'>")
            for take in TAKES:
                c = by_id[clip_id].get(take)
                if c is None:
                    continue
                key = f"{clip_id}|{take}"
                parts.append(
                    f"<div class='take'><b>{esc(take)}</b><audio controls preload='none' src='takes/{esc(take)}/{esc(clip_id)}.wav'></audio>"
                    f"<div class='marks' data-id='{esc(key)}'><button type='button' class='yes' data-v='+'>✓ düzgün sual</button>"
                    f"<button type='button' class='no' data-v='-'>✗ düz deyil</button></div>"
                    f"<div class='metrics'>son: {c['final_st']:+.1f} st · zirvə: {c['peak_rise_st']:+.1f} st · enmə: {c['end_drop_st']:.1f} st"
                    f"{' · oktava sıçrayışı' if c['octave_jump'] else ''}</div></div>")
            parts.append("</div></div>")
    parts.append("<div class='bar'><button type='button' id='copy'>Nəticəni kopyala</button><button type='button' id='show'>Göstəriciləri göstər</button>"
                 "<span id='count' class='note'></span><span id='status' class='note'></span></div>"
                 "<textarea id='out' hidden rows='8' style='width:100%' aria-label='Nəticə mətni'></textarea>")
    parts.append(f"<script>{SCRIPT}</script></main></body></html>")
    (args.run / "questions.html").write_text("\n".join(parts), encoding="utf-8")
    print("wrote", args.run / "questions.html")


if __name__ == "__main__":
    main()
