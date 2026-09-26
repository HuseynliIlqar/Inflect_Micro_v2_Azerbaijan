"""Find and remove the vocoder's resonant buzz, then even out loudness.

    python tools/audio_postprocess.py analyze samples
    python tools/audio_postprocess.py clean out --notch 3120 --out out/clean

A 9.36M HiFi-GAN decoder tends to leave a narrow resonance -- the metallic
"buzz" -- at one or two fixed frequencies.  Because it sits in the same place in
every clip, it shows up as a peak that stays above the averaged spectrum of the
whole set, and a narrow notch removes it without touching speech.

`analyze` reports those candidate frequencies.  `clean` applies the filter chain
through ffmpeg; nothing is guessed, the notches come from what analyze found.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aztts.console import use_utf8  # noqa: E402

use_utf8()

WINDOW = 4096
HOP = 2048
# A peak must stand this many dB above the smoothed spectrum to count as a
# resonance rather than ordinary speech energy.
PROMINENCE_DB = 4.0
# Speech formants live below this; buzz sits above it.
MIN_HZ = 1500.0
# Bands reported by `analyze`, so a broadband problem is visible even when no
# single peak stands out.
BANDS = ((0.0, 1500.0), (1500.0, 4000.0), (4000.0, 8000.0), (8000.0, 12000.0))


def average_spectrum(paths: tuple[Path, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Mean magnitude spectrum, in dB, across every frame of every file."""
    total: np.ndarray | None = None
    frames = 0
    sample_rate = 0
    window = np.hanning(WINDOW).astype(np.float32)
    for path in paths:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        sample_rate = sample_rate or rate
        if rate != sample_rate:
            raise ValueError(f"{path}: {rate} Hz, expected {sample_rate} Hz")
        for start in range(0, max(len(mono) - WINDOW, 0), HOP):
            spectrum = np.abs(np.fft.rfft(mono[start : start + WINDOW] * window))
            total = spectrum if total is None else total + spectrum
            frames += 1
    if total is None or not frames:
        raise ValueError("not enough audio to analyse")
    magnitude = 20.0 * np.log10(total / frames + 1e-12)
    return np.fft.rfftfreq(WINDOW, 1.0 / sample_rate), magnitude


def smooth(values: np.ndarray, width: int = 41) -> np.ndarray:
    """Moving average, used as the baseline a real peak has to beat."""
    kernel = np.ones(width, dtype=np.float64) / width
    padded = np.pad(values, width // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def find_resonances(
    frequencies: np.ndarray, magnitude: np.ndarray, limit: int = 5
) -> tuple[tuple[float, float], ...]:
    """Return (frequency, prominence_dB) for the strongest narrow peaks."""
    excess = magnitude - smooth(magnitude)
    peaks = [
        (float(frequencies[i]), float(excess[i]))
        for i in range(1, excess.size - 1)
        if frequencies[i] >= MIN_HZ
        and excess[i] >= PROMINENCE_DB
        and excess[i] > excess[i - 1]
        and excess[i] >= excess[i + 1]
    ]
    return tuple(sorted(peaks, key=lambda peak: peak[1], reverse=True)[:limit])


def band_report(
    frequencies: np.ndarray, magnitude: np.ndarray
) -> tuple[tuple[float, float, float, float], ...]:
    """Per-band (low, high, mean dB, largest deviation from the baseline)."""
    excess = magnitude - smooth(magnitude)
    rows = []
    for low, high in BANDS:
        selected = (frequencies >= low) & (frequencies < high)
        if not selected.any():
            continue
        rows.append(
            (low, high, float(magnitude[selected].mean()), float(excess[selected].max()))
        )
    return tuple(rows)


def collect_wavs(target: Path) -> tuple[Path, ...]:
    if target.is_file():
        return (target,)
    return tuple(sorted(target.rglob("*.wav")))


def build_filter(
    notches: tuple[float, ...],
    highpass: float,
    lowpass: float,
    loudness: float,
) -> str:
    stages = [f"highpass=f={highpass:g}"]
    stages += [f"equalizer=f={hz:g}:t=q:w=12:g=-9" for hz in notches]
    if lowpass > 0:
        stages.append(f"lowpass=f={lowpass:g}")
    stages.append(f"loudnorm=I={loudness:g}:TP=-1.5:LRA=11")
    return ",".join(stages)


def run_analyze(args: argparse.Namespace) -> int:
    paths = collect_wavs(args.target)
    if not paths:
        print(f"error: no wav files under {args.target}", file=sys.stderr)
        return 1
    frequencies, magnitude = average_spectrum(paths)
    resonances = find_resonances(frequencies, magnitude)
    print(f"files : {len(paths)}")
    print(f"root  : {args.target}")
    print()
    print("Mean energy per band:")
    for low, high, mean, deviation in band_report(frequencies, magnitude):
        print(f"  {low:6.0f}-{high:6.0f} Hz   mean {mean:7.1f} dB   "
              f"max deviation {deviation:5.2f} dB")
    print()
    if not resonances:
        print(f"No peak above {MIN_HZ:g} Hz clears {PROMINENCE_DB:g} dB of prominence.")
        print("If you still hear buzz, listen closely -- it may be broadband.")
        return 0
    print("Likely resonances (buzz candidates):")
    for hertz, prominence in resonances:
        print(f"  {hertz:8.0f} Hz   +{prominence:.1f} dB")
    print()
    flags = " ".join(f"--notch {hz:.0f}" for hz, _ in resonances[:2])
    print(f"To clean:\n  python {Path(__file__).name} clean <folder> {flags}")
    return 0


def run_clean(args: argparse.Namespace) -> int:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found (it must be on PATH)", file=sys.stderr)
        return 1
    paths = collect_wavs(args.target)
    if not paths:
        print(f"error: no wav files under {args.target}", file=sys.stderr)
        return 1
    chain = build_filter(
        tuple(args.notch), args.highpass, args.lowpass, args.loudness
    )
    print(f"filter: {chain}")
    print()
    base = args.target if args.target.is_dir() else args.target.parent
    for path in paths:
        destination = args.out / path.relative_to(base)
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-i", str(path), "-af", chain, str(destination)],
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            print(f"error: {path}: {completed.stderr.strip()}", file=sys.stderr)
            return 1
        print(f"  {destination}")
    print()
    print(f"done: {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Report candidate buzz frequencies.")
    analyze.add_argument("target", type=Path, help="A wav file or a folder of them.")
    analyze.set_defaults(handler=run_analyze)

    clean = commands.add_parser("clean", help="Apply the filter chain with ffmpeg.")
    clean.add_argument("target", type=Path, help="A wav file or a folder of them.")
    clean.add_argument("--out", type=Path, default=Path("out/clean"))
    clean.add_argument(
        "--notch",
        type=float,
        nargs="*",
        default=[],
        help="Frequencies to notch out, from `analyze`.",
    )
    clean.add_argument("--highpass", type=float, default=70.0)
    clean.add_argument(
        "--lowpass",
        type=float,
        default=0.0,
        help="Roll off above this frequency; 11000 tames the vocoder's band edge.",
    )
    clean.add_argument("--loudness", type=float, default=-16.0, help="LUFS target.")
    clean.set_defaults(handler=run_clean)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
