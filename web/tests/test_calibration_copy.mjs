// What the device check says: every verdict readable, in both languages.
import { calibrationCopy, calibrationLine } from "../js/ui/calibration-copy.js";
import { CALIBRATION_LABELS } from "../js/ui/i18n-calibration.js";
import { label, LABELS } from "../js/ui/i18n.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

const GPU = { backend: "webgpu", precision: "fp32" };
const INT8 = { backend: "wasm", precision: "int8" };
const CPU = { backend: "wasm", precision: "fp32" };
const verdict = (extra) => ({
  chosen: INT8, rtf: 0.8, sentenceSeconds: 3.2, tier: "usable", gpu: "wrong-output",
  gpuMessage: "", int8Failed: false, threads: 4, framed: false,
  trials: [
    { ...INT8, ok: true, reason: null, rtf: 0.8 },
    { ...GPU, ok: false, reason: "length", rtf: 0.6 },
  ],
  ...extra,
});
const keys = (copy) => copy.reasons.map((r) => r.key);

{
  const copy = calibrationCopy(verdict(), { phone: true });
  expect("usable: amber, two bars", [copy.tone, copy.bars], ["warn", 2]);
  expect("usable: title", copy.title, "cal_title_usable");
  expect("the measured speed, in seconds", copy.summary, { key: "cal_summary", values: { rtf: "0.80", sentence: "3.2" } });
  expect("where it runs", copy.runsOn.key, "cal_runs_cpu_fast");
  expect("the reasons: the GPU, the threads, the advice", keys(copy), [
    "cal_gpu_wrong_output", "cal_threads_many", "cal_advice_phone",
  ]);
}
{
  const copy = calibrationCopy(verdict({ chosen: GPU, rtf: 0.1, sentenceSeconds: 0.4, tier: "fast", gpu: "chosen" }));
  expect("fast: green, three bars", [copy.tone, copy.bars, copy.title], ["good", 3, "cal_title_fast"]);
  expect("fast on the GPU: no thread line, no advice", keys(copy), ["cal_gpu_chosen"]);
  expect("runs on the GPU", copy.runsOn.key, "cal_runs_gpu");
}
{
  const copy = calibrationCopy(verdict({ tier: "slow", threads: 1, framed: true, gpu: "no-adapter" }));
  expect("slow and framed", keys(copy), ["cal_gpu_no_adapter", "cal_threads_framed", "cal_advice_browser"]);
  expect("slow: red", copy.tone, "bad");
}
{
  const copy = calibrationCopy(verdict({ chosen: CPU, int8Failed: true, tier: "slow", threads: 1, gpu: "unsupported" }), { appleMobile: true });
  expect("int8 broken, on an iPhone", keys(copy), ["cal_gpu_unsupported", "cal_threads_single", "cal_int8_failed", "cal_advice_ios"]);
  expect("full model on the CPU", copy.runsOn.key, "cal_runs_cpu_full");
}
{
  const copy = calibrationCopy(verdict({ chosen: null, rtf: null, sentenceSeconds: null, tier: "failed", gpu: "error", gpuMessage: "device lost" }));
  expect("failed: its own summary", [copy.title, copy.summary.key, copy.runsOn], ["cal_title_failed", "cal_summary_failed", null]);
  expect("the GPU's error is quoted", copy.reasons[0], { key: "cal_gpu_error", values: { message: "device lost" } });
}

expect(
  "the diagnostics line",
  calibrationLine(verdict()),
  "check=wasm/int8@0.80 gpu:wrong-output [wasm/int8:ok@0.80,webgpu/fp32:length@0.60]",
);
expect("no check", calibrationLine(null), "check=none");

// Both languages carry the same keys, and every one is used text.
const az = Object.keys(CALIBRATION_LABELS.az).sort();
expect("en has the same keys as az", Object.keys(CALIBRATION_LABELS.en).sort(), az);
for (const lang of Object.keys(LABELS)) {
  for (const key of az) {
    const text = label(lang, key);
    expect(`[${lang}] ${key} reads through label()`, text !== key && text.trim().length > 0, true);
  }
  // The verdict itself stays plain; only the reasons name WebGPU.
  for (const key of ["cal_title_fast", "cal_title_usable", "cal_title_slow", "cal_summary"]) {
    expect(`[${lang}] ${key} has no jargon`, /WebGPU|WASM|int8|fp32|backend/iu.test(label(lang, key)), false);
  }
  expect(`[${lang}] placeholders filled`, label(lang, "cal_summary", { rtf: "0.80", sentence: "3.2" }).includes("{"), false);
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  calibration-copy: ${checked} checks on what the device check says`);
