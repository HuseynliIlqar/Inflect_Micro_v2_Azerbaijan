"""Measure buzz and high-band ring in TTS WAV files, the same way for every system.

Buzz is Paradee's measure: energy on the pitch harmonics against energy halfway between them, in
voiced frames, in dB, for 1-2, 2-4 and 4-8 kHz (STFT 4096 / window 1024 / hop 256 at 24 kHz, +-0.15 f0
around each line, a frame counts when the pitch is voiced two frames either side). Higher is cleaner.
Paradee read f0 from its own model; here f0 comes from the audio (YIN, refined on the spectrum with a
harmonic comb), so a system cannot be measured with information the others do not have.

Two supporting numbers: "ring" is the largest share of speech frames that show a spectral line at one
fixed frequency in 5-11.8 kHz (a vocoder resonance stays put, harmonics move with the pitch), and
"air" is the 10-12 kHz level against 1-4 kHz.

Usage: python -I buzz_measure.py OUT_JSON LABEL=GLOB [LABEL=GLOB ...]
"""
from __future__ import annotations

import glob
import json
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

SR, NF, WIN, HOP = 24000, 4096, 1024, 256
BANDS = ((1000, 2000), (2000, 4000), (4000, 8000))
FMIN, FMAX = 70.0, 450.0
YIN_W, YIN_THRESH, VOICED_MAX = 1024, 0.15, 0.25
SILENCE_DB = -45.0  # frames quieter than this, relative to the loudest frame, are skipped
COMB_SPAN, COMB_TOP = 0.02, 4000.0
RING_LO, RING_HI, RING_PROM_DB, RING_SMOOTH_HZ = 5000.0, 11800.0, 6.0, 300.0

FR = np.arange(NF // 2 + 1) * SR / NF
WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)  # periodic Hann, as torch.hann_window


def load(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def framed(x: np.ndarray, length: int) -> np.ndarray:
    """Frames of `length` centred on j * HOP, reflect-padded like torch.stft(center=True)."""
    half = length // 2
    xp = np.pad(x, (half, half + HOP), mode="reflect")
    n = 1 + len(x) // HOP
    return xp[np.arange(n)[:, None] * HOP + np.arange(length)[None, :]]


def yin(x: np.ndarray) -> np.ndarray:
    """f0 per frame in Hz, 0 where unvoiced."""
    tmax, tmin = int(np.ceil(SR / FMIN)), int(SR / FMAX)
    seg = framed(x, YIN_W + tmax)
    a = seg[:, :YIN_W]
    r = np.fft.irfft(np.conj(np.fft.rfft(a, NF)) * np.fft.rfft(seg, NF), NF)[:, : tmax + 1]
    cs = np.concatenate([np.zeros((len(seg), 1)), np.cumsum(seg**2, axis=1)], axis=1)
    lags = np.arange(tmax + 1)
    d = cs[:, [YIN_W]] + (cs[:, lags + YIN_W] - cs[:, lags]) - 2 * r
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1)
    dp = np.ones_like(d)
    dp[:, 1:] = d[:, 1:] * lags[1:] / np.maximum(cum, 1e-12)
    f0 = np.zeros(len(seg))
    for j in range(len(seg)):
        row = dp[j]
        below = np.nonzero(row[tmin:tmax] < YIN_THRESH)[0]
        t = tmin + below[0] if len(below) else tmin + int(np.argmin(row[tmin:tmax]))
        while t + 1 < tmax and row[t + 1] < row[t]:
            t += 1
        if row[t] > VOICED_MAX:
            continue
        den = row[t - 1] - 2 * row[t] + row[t + 1]
        shift = 0.5 * (row[t - 1] - row[t + 1]) / den if abs(den) > 1e-12 else 0.0
        f0[j] = SR / (t + float(np.clip(shift, -1, 1)))
    return f0


def refine(power: np.ndarray, f: float) -> float:
    """Nudge f0 within +-2 % to where the harmonics line up best on this frame's spectrum."""
    cands = f * (1 + np.linspace(-COMB_SPAN, COMB_SPAN, 81))
    ks = np.arange(1, int(COMB_TOP / f) + 1)
    logp = np.log(power + 1e-12)
    score = np.interp(np.outer(cands, ks), FR, logp).mean(axis=1)
    return float(cands[int(np.argmax(score))])


def band_ratio(p: np.ndarray, f: float, lo: float, hi: float) -> float:
    ks = np.arange(int(np.ceil(lo / f)), int(hi / f))
    on = sum(p[(FR > k * f - 0.15 * f) & (FR < k * f + 0.15 * f)].sum() for k in ks)
    off = sum(p[(FR > (k + 0.5) * f - 0.15 * f) & (FR < (k + 0.5) * f + 0.15 * f)].sum() for k in ks)
    return 10 * np.log10((on + 1e-12) / (off + 1e-12))


def ring_and_air(power: np.ndarray, speech: np.ndarray) -> tuple[float, float, float]:
    sel = power[speech]
    band = (FR >= RING_LO) & (FR <= RING_HI)
    db = 10 * np.log10(sel[:, band] + 1e-12)
    smooth = median_filter(db, size=(1, int(RING_SMOOTH_HZ / (SR / NF)) | 1), mode="nearest")
    prom = db - smooth
    peak = (prom >= RING_PROM_DB) & (db >= np.roll(db, 1, axis=1)) & (db >= np.roll(db, -1, axis=1))
    near = np.zeros_like(peak)
    for s in range(-2, 3):
        near |= np.roll(peak, s, axis=1)
    share = near.mean(axis=0)
    k = int(np.argmax(share))
    air = sel[:, (FR >= 10000) & (FR < 12000)].sum() / max(sel[:, (FR >= 1000) & (FR < 4000)].sum(), 1e-12)
    return float(share[k]), float(FR[band][k]), float(10 * np.log10(air + 1e-12))


def measure(path: str) -> dict:
    x = load(path)
    fr = framed(x, WIN) * WINDOW
    power = np.abs(np.fft.rfft(fr, NF)) ** 2
    rms_db = 10 * np.log10((fr**2).mean(axis=1) + 1e-12)
    speech = rms_db > rms_db.max() + SILENCE_DB
    f0 = yin(x) * speech
    voiced = f0 > 0
    if voiced.any():  # drop octave slips against the file's own median pitch
        med = np.median(f0[voiced])
        f0[voiced & ((f0 < 0.7 * med) | (f0 > 1.4 * med))] = 0
    ratios = {b: [] for b in BANDS}
    for j in range(2, len(f0) - 2):
        if min(f0[j - 2], f0[j], f0[j + 2]) < 80:
            continue
        f = refine(power[j], f0[j])
        for lo, hi in BANDS:
            ratios[(lo, hi)].append(band_ratio(power[j], f, lo, hi))
    ring, ring_hz, air = ring_and_air(power, speech)
    peak = float(np.abs(x).max())
    return {
        "file": path,
        "seconds": len(x) / SR,
        "voiced_frames": len(ratios[BANDS[0]]),
        "f0_median": float(np.median(f0[f0 > 0])) if (f0 > 0).any() else 0.0,
        **{f"h{lo // 1000}-{hi // 1000}k": float(np.mean(v)) if v else float("nan") for (lo, hi), v in ratios.items()},
        "ring_share": ring, "ring_hz": ring_hz, "air_db": air,
        "peak": peak, "clipped": int((np.abs(x) >= 0.999).sum()),
    }


def summary(rows: list[dict]) -> dict:
    out = {"files": len(rows), "seconds": sum(r["seconds"] for r in rows)}
    for key in ("h1-2k", "h2-4k", "h4-8k", "ring_share", "air_db", "f0_median"):
        v = np.array([r[key] for r in rows if np.isfinite(r[key])])
        out[key] = float(v.mean())
        out[key + "_ci"] = float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    out["peak_max"] = max(r["peak"] for r in rows)
    out["clipped"] = sum(r["clipped"] for r in rows)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out_json, specs = sys.argv[1], sys.argv[2:]
    result = {}
    for spec in specs:
        label, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"{label}: no files match {pattern}")
        rows = [measure(f) for f in files]
        result[label] = {"summary": summary(rows), "files": rows}
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    head = f"{'system':22} {'n':>3} {'sec':>6} {'f0':>5} {'1-2k':>6} {'2-4k (95% CI)':>15} {'4-8k':>6} {'ring':>5} {'air':>6} {'peak':>5} {'clip':>4}"
    print(head)
    for label, data in result.items():
        s = data["summary"]
        print(f"{label:22} {s['files']:3d} {s['seconds']:6.1f} {s['f0_median']:5.0f} {s['h1-2k']:6.2f} "
              f"{s['h2-4k']:7.2f} +-{s['h2-4k_ci']:4.2f}  {s['h4-8k']:6.2f} {s['ring_share']:5.2f} "
              f"{s['air_db']:6.1f} {s['peak_max']:5.2f} {s['clipped']:4d}")


if __name__ == "__main__":
    main()
