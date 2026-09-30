"""Build the int8 decoder the browser playground uses when there is no WebGPU.

    python tools/quantize_decoder.py                       # web/onnx/decode.onnx -> decode.int8.onnx
    python tools/quantize_decoder.py --memory-gib 3 --out /tmp/decode.int8.onnx

Static QDQ quantisation of the Conv/ConvTranspose layers, calibrated on real
Azerbaijani sentences. Kept in fp32, because quantising them costs quality or
speed:

- the flow (its layers are cheap; its error feeds everything after it)
- the vocoder's first and last convolutions
- the vocoder's last upsampling stage and its three resblocks: the largest
  tensors with the fewest channels, where int8 is both the least accurate and,
  with the quantise/dequantise overhead, not even faster

Measured against fp32 on the five demo sentences (same noise): a mean
log-spectral distance of 0.53, below the 0.73 between two fp32 seeds, and
about 1.5x faster in onnxruntime-web's wasm backend (1 and 4 threads). Dynamic
quantisation was slower than fp32 on this model and is not offered.

Needs `onnx` besides onnxruntime (`pip install onnx`); it is a publishing
tool, not a runtime dependency. Memory is the constraint, not CPU: the
calibrator keeps every activation of every sample (~2.3 MiB per frame, ~24 GiB
for this set), which froze a 16 GB machine once. So samples go through one at a
time, and the process is capped -- past the cap it fails with MemoryError
instead of dragging the machine into swap.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from aztts.console import use_utf8  # noqa: E402

use_utf8()

LAST_STAGE = 3          # the vocoder has four upsampling stages, 0..3
RESBLOCKS_PER_STAGE = 3

# Calibration text: varied length, questions, numbers, lists. Language data.
CALIBRATION = (
    "Bu gün hava çox isti olacaq, ona görə suyu yanınızda götürün.",
    "Qatar səhər saat yeddidə yola düşür və axşam Gəncəyə çatır.",
    "Nə üçün bu qədər gec gəldin? Səni bir saatdır gözləyirəm.",
    "Kitabxanada yeni kitablar var, amma oxucu azdır.",
    "Uşaqlar həyətdə top oynayır, analar isə skamyada söhbət edir.",
    "Bakı Xəzər dənizinin sahilində yerləşən böyük bir şəhərdir.",
    "Mən bu mahnını ilk dəfə nənəmdən eşitmişdim.",
    "Toplantı sabah saat on birdə başlayacaq, gecikməyin.",
    "Dağların başında hələ də qar var, yollar isə sürüşkəndir.",
    "Bu layihə üçün üç ay vaxtımız və beş nəfərlik komandamız var.",
    "Sən haqlısan, amma mən də öz fikrimdə qalıram.",
    "Bazarda meyvə bol idi: alma, armud, heyva və nar.",
    "Həkim dedi ki, hər gün ən azı yarım saat piyada gəzmək lazımdır.",
    "Maşın birdən dayandı və heç cür işə düşmədi.",
    "Qonaqlar gələndə çay süzüldü, masaya şirniyyat qoyuldu.",
    "Universitetə qəbul imtahanları iyul ayında keçiriləcək.",
    "Gəl, bir az oturaq, sonra yenə yola davam edərik.",
    "Onun səsi titrəyirdi, gözləri isə yaşla dolmuşdu.",
    "Şirkət keçən il gəlirini iyirmi faiz artırıb.",
    "Bu sualın cavabını heç kim bilmirdi, hamı susurdu.",
    "Gecə ulduzlar parlayırdı və dəniz sakit idi.",
    "Telefonu götür, anan sənə zəng edir!",
    "Muzeyin qapıları bazar ertəsi günləri bağlı olur.",
    "Kəndin kənarındakı çayın suyu yayda azalır.",
    "Mənə elə gəlir ki, bu məsələni tələsmədən həll etməliyik.",
    "Otağın pəncərəsindən bütün şəhər görünürdü.",
    "Yeni il gecəsi hamı meydana toplaşıb atəşfəşanlığa baxdı.",
    "Dərsdən sonra müəllim şagirdlərə ev tapşırığı verdi.",
    "Ağır yağış səbəbindən bir neçə uçuş təxirə salındı.",
    "Sabah səhər tezdən durub birlikdə dağa qalxacağıq.",
)


def nodes_kept_in_fp32(conv_names: Iterable[str]) -> list[str]:
    """The convolutions left unquantised, by node name (see the module docstring)."""
    last_blocks = tuple(
        f"/decoder/resblocks.{LAST_STAGE * RESBLOCKS_PER_STAGE + k}/" for k in range(RESBLOCKS_PER_STAGE)
    )
    return [
        name
        for name in conv_names
        if name.startswith("/flow/")
        or name.endswith(("/decoder/conv_pre/Conv", "/decoder/conv_post/Conv"))
        or name.startswith(f"/decoder/ups.{LAST_STAGE}/")
        or name.startswith(last_blocks)
    ]


# Windows caps committed memory; elsewhere RLIMIT_AS caps address space, which
# thread stacks, malloc arenas and onnxruntime's mappings also count against.
DEFAULT_MEMORY_GIB = 3.0 if sys.platform == "win32" else 8.0


def cap_memory(gib: float) -> None:
    """Fail with MemoryError past `gib` instead of paging the whole machine."""
    limit = int(gib * 2**30)
    if sys.platform != "win32":
        import resource

        try:
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError) as error:  # macOS does not enforce it
            print(f"warning: could not cap memory ({error}); watch it yourself", file=sys.stderr)
        return

    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    job = wintypes.HANDLE(kernel32.CreateJobObjectW(None, None))
    info = Extended()
    info.BasicLimitInformation.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = limit
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "could not set the memory limit")
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(kernel32.GetCurrentProcess())):
        raise OSError(ctypes.get_last_error(), "could not apply the memory limit")


def feed_one_sample_at_a_time() -> None:
    """Make onnxruntime's calibrators hold one sample's activations, not all of them.

    Both keep every sample until the end. Percentile ignores
    CalibMaxIntermediateOutputs, and MinMax in onnxruntime 1.30 drops a batch
    without computing its range when that limit is hit. Both merge their
    results across calls, so each sample gets a call of its own.
    """
    from onnxruntime.quantization import CalibrationDataReader
    from onnxruntime.quantization.calibrate import HistogramCalibrater, MinMaxCalibrater

    class One(CalibrationDataReader):
        def __init__(self, item: dict) -> None:
            self.item: dict | None = item

        def get_next(self) -> dict | None:
            item, self.item = self.item, None
            return item

    def one_at_a_time(collect_all):
        def collect(self, reader) -> None:
            samples = 0
            while (item := reader.get_next()) is not None:
                collect_all(self, One(item))
                samples += 1
            if not samples:
                raise ValueError("no calibration data: the frontend rejected every sentence")
        return collect

    HistogramCalibrater.collect_data = one_at_a_time(HistogramCalibrater.collect_data)
    MinMaxCalibrater.collect_data = one_at_a_time(MinMaxCalibrater.collect_data)


def calibration_inputs(duration_path: Path) -> Iterator[dict]:
    """Decoder inputs for the calibration sentences, through the real frontend."""
    import json

    import numpy as np
    import onnxruntime as ort
    from deployment_frontend import DeploymentFrontendError, process_input

    from aztts import normalize_az
    from aztts.az_chunk import chunk_text

    symbols = json.loads((PROJECT_ROOT / "model" / "symbols.json").read_text(encoding="utf-8"))
    symbols = symbols["symbols"] if isinstance(symbols, dict) else symbols
    ids = {symbol: index for index, symbol in enumerate(symbols)}
    duration = ort.InferenceSession(str(duration_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)

    for sentence in CALIBRATION:
        for chunk in chunk_text(normalize_az(sentence)):
            try:
                phonemes = process_input(chunk).phoneme_text
            except DeploymentFrontendError:
                continue
            raw = [ids[c] for c in phonemes if c in ids]
            spread = [0] * (2 * len(raw) + 1)
            spread[1::2] = raw
            tokens = np.array([spread], dtype=np.int64)
            mean, logs, mask = duration.run(["m_p_exp", "logs_p_exp", "y_mask"], {
                "tokens": tokens,
                "lengths": np.array([tokens.shape[1]], np.int64),
                "length_scale": np.array(1.0, np.float32),
            })
            yield {
                "m_p_exp": mean, "logs_p_exp": logs, "y_mask": mask,
                "zp_noise": rng.standard_normal(mean.shape).astype(np.float32),
                "noise_scale": np.array(0.667, np.float32),
            }


def quantize(decode: Path, duration: Path, out: Path) -> None:
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.items = calibration_inputs(duration)

        def get_next(self) -> dict | None:
            return next(self.items, None)

    feed_one_sample_at_a_time()
    with tempfile.TemporaryDirectory() as scratch:
        prepared = Path(scratch) / "decode.pre.onnx"
        # In a child process: shape inference keeps ~1 GiB it never returns,
        # which calibration then needs. The child inherits the memory cap.
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            pool.submit(quant_pre_process, str(decode), str(prepared), skip_symbolic_shape=False).result()
        convs = [n.name for n in onnx.load(str(prepared)).graph.node if n.op_type in ("Conv", "ConvTranspose")]
        keep = nodes_kept_in_fp32(convs)
        print(f"{len(convs)} convolutions, {len(keep)} kept in fp32")
        quantize_static(
            str(prepared), str(out), Reader(),
            quant_format=QuantFormat.QDQ, per_channel=True,
            activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
            op_types_to_quantize=["Conv", "ConvTranspose"],
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=keep,
        )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--decode", type=Path, default=PROJECT_ROOT / "web" / "onnx" / "decode.onnx")
    parser.add_argument("--duration", type=Path, default=PROJECT_ROOT / "web" / "onnx" / "duration.onnx")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "web" / "onnx" / "decode.int8.onnx")
    parser.add_argument("--memory-gib", type=float, default=DEFAULT_MEMORY_GIB,
                        help=f"fail past this much memory instead of swapping (default: {DEFAULT_MEMORY_GIB:g}; "
                             "committed memory on Windows, address space elsewhere)")
    args = parser.parse_args()

    for path in (args.decode, args.duration):
        if not path.is_file():
            print(f"missing {path}; see web/README.md for where the ONNX graphs come from", file=sys.stderr)
            return 1
    cap_memory(args.memory_gib)
    try:
        quantize(args.decode, args.duration, args.out)
    except (MemoryError, RuntimeError) as error:
        # onnxruntime's C++ allocations surface as RuntimeError, not MemoryError.
        text = str(error).lower()
        if isinstance(error, RuntimeError) and not any(k in text for k in ("bad_alloc", "allocat", "memory")):
            raise
        print(f"ran out of the {args.memory_gib:g} GiB allowed; raise --memory-gib", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
