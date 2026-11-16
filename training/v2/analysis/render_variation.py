"""Read the same texts with the v2.1 package at several variation values, for a listening choice.

    <azerbaycan-tts venv python> render_variation.py

Writes D:/ktts/avaz-train/variation_test/<value>/{long_NN,short_NN}.wav, texts.json and
metrics.json (buzz and ring per clip, when buzz_measure imports). Same seeds for every value,
so only the variation differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PACKAGE = Path("D:/ktts/avaz-train/results/final/avaz-v2.1")
LONG_TEXTS = Path("D:/ktts/avaz-train/longtest/texts.json")
OUT = Path("D:/ktts/avaz-train/variation_test")
VALUES = (0.667, 0.3, 0.2)
PARAGRAPH_PAUSE = 0.55
SHORT = [
    "Valideynlərim nəvə istəyirlər.",
    "Yeməyi də bir şey taparam.",
    "Tom ağır xəstədir.",
    "Kitabım masadadır, kitabın isə çantadadır.",
    "Hazırda təqaüddə olsa da, bu sahədə əldə etdiyi bəzi bacarıqları görənləri təəccübləndirir.",
    "Babək xürrəmilərə rəhbərliyi qəbul etdikdən sonra onların mübarizəsi tamamilə Xilafətə qarşı çevrildi.",
    "Papaq pəncərənin pərdəsinə ilişdi.",
    "Bütün pulunu onlara verdinmi?",
    "Sən sabah işə gedəcəksənmi?",
    "Qırx qarğa qışda qaranlıq qayalıqda qaldı.",
]


def main() -> None:
    sys.path.insert(0, str(PACKAGE))
    from inference import InflectTTS

    engine = InflectTTS(PACKAGE, device="cpu")
    rate = engine.sample_rate
    long_texts = json.loads(LONG_TEXTS.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "texts.json").write_text(json.dumps({"long": long_texts, "short": SHORT}, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    for value in VALUES:
        folder = OUT / f"{value}"
        folder.mkdir(exist_ok=True)
        for n, text in enumerate(long_texts):
            pieces = []
            for p, paragraph in enumerate(text["paragraphs"]):
                if p:
                    pieces.append(np.zeros(round(rate * PARAGRAPH_PAUSE), np.float32))
                pieces.append(engine.synthesize(paragraph, variation=value, seed=p * 100)[1].astype(np.float32))
            sf.write(folder / f"long_{n:02d}.wav", np.concatenate(pieces), rate)
        for n, text in enumerate(SHORT):
            sf.write(folder / f"short_{n:02d}.wav", engine.synthesize(text, variation=value, seed=0)[1], rate)
        print(f"rendered variation {value}", flush=True)
    try:
        sys.path.insert(0, "D:/ktts/avaz-train/pod")
        import buzz_measure as bm
    except ImportError as error:
        print(f"buzz measurement skipped: {error}")
        return
    metrics = {}
    for value in VALUES:
        rows = [bm.measure(str(p)) for p in sorted((OUT / f"{value}").glob("*.wav"))]
        metrics[str(value)] = {"h2_4k": round(float(np.nanmedian([r["h2-4k"] for r in rows])), 2),
                               "ring_share": round(float(np.nanmedian([r["ring_share"] for r in rows])), 3)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
