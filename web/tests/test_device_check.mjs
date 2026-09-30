// What the page tells a visitor about this device, before and after synthesis.
import { deviceVerdict, errorKind, adviceKey, isAppleMobile } from "../js/device-check.js";
import { deviceCopy } from "../js/device-view.js";
import { label } from "../js/i18n.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};
const pick = (v) => ({ level: v.level, bars: v.bars, tone: v.tone, dialog: v.dialog, action: v.action });

// -- the verdict ------------------------------------------------------------------

expect(
  "WebGPU: three bars, good, a toast not a dialog",
  pick(deviceVerdict({ gpu: true, threads: 4, framed: false })),
  { level: "gpu", bars: 3, tone: "good", dialog: false, action: null },
);
expect(
  "no WebGPU, 4 threads: two bars, fast mode, a dialog",
  pick(deviceVerdict({ gpu: false, threads: 4, framed: false })),
  { level: "threads", bars: 2, tone: "warn", dialog: true, action: null },
);
expect(
  "no WebGPU, one thread inside the Hub page: one bar, offer the direct link",
  pick(deviceVerdict({ gpu: false, threads: 1, framed: true })),
  { level: "single-framed", bars: 1, tone: "bad", dialog: true, action: "open-direct" },
);
expect(
  "no WebGPU, one thread, not framed: the browser is the limit",
  pick(deviceVerdict({ gpu: false, threads: 1, framed: false })),
  { level: "single", bars: 1, tone: "bad", dialog: true, action: null },
);
expect(
  "WebGPU failed at run time: same as no WebGPU, flagged",
  deviceVerdict({ gpu: true, gpuFailed: true, threads: 4, framed: false }).level,
  "threads",
);
expect(
  "the failure is remembered in the verdict",
  deviceVerdict({ gpu: true, gpuFailed: true, threads: 4 }).gpuFailed,
  true,
);
expect(
  "?backend=wasm: treated as no WebGPU, and says it was forced",
  [deviceVerdict({ gpu: true, preferWasm: true, threads: 4 }).level,
    deviceVerdict({ gpu: true, preferWasm: true, threads: 4 }).forced],
  ["threads", true],
);
expect("threads default to one", deviceVerdict({ gpu: false }).bars, 1);

// -- errors -> what to do next ----------------------------------------------------

expect("a download failure is the network", errorKind("could not download ./onnx/decode.onnx (HTTP 503)"), "network");
expect("a fetch TypeError is the network", errorKind("Failed to fetch"), "network");
expect("the worker", errorKind("The browser could not start the background worker"), "worker");
expect("nothing loads", errorKind("no backend could load the model (wasm/fp32: x)"), "backend");
expect("out of memory", errorKind("RangeError: Array buffer allocation failed"), "memory");
expect("anything else", errorKind("something odd"), "generic");
expect("empty input is not an error kind", errorKind(""), "generic");

// -- the words each verdict reads as ------------------------------------------------

const az = (key, values) => label("az", key, values);
const copyOf = (facts) => deviceCopy(deviceVerdict(facts), az);
expect("GPU title", copyOf({ gpu: true }).title, label("az", "device_title_gpu"));
expect("threads text carries the count", copyOf({ gpu: false, threads: 4 }).text.includes("4 axınla"), true);
expect("framed single thread points at the direct link", copyOf({ gpu: false, framed: true }).text,
  label("az", "device_text_single_framed"));
expect("a failed GPU says so in the title", copyOf({ gpu: true, gpuFailed: true, threads: 4 }).title,
  label("az", "device_gpu_failed_title"));
expect("forced CPU carries the test-mode note", copyOf({ gpu: true, preferWasm: true, threads: 4 }).note,
  label("az", "device_note_forced"));
expect("no note otherwise", copyOf({ gpu: false, threads: 4 }).note, "");
for (const level of ["device_title_gpu", "device_text_gpu", "device_title_threads", "device_text_threads",
  "device_title_single", "device_text_single", "device_text_single_framed", "device_gpu_failed_title",
  "error_next_network", "error_next_worker", "error_next_backend", "error_next_memory", "error_next_generic"]) {
  expect(`label exists in both languages: ${level}`,
    [label("az", level) !== level, label("en", level) !== level], [true, true]);
}

// -- phones: WebGPU is skipped on purpose, so the card must not promise it -----------

expect(
  "a phone with WebGPU and four threads: the phone verdict, not the GPU one",
  pick(deviceVerdict({ gpu: true, phone: true, threads: 4, framed: false })),
  { level: "phone", bars: 2, tone: "warn", dialog: true, action: null },
);
expect(
  "a phone without WebGPU reads the same",
  deviceVerdict({ gpu: false, phone: true, threads: 4, framed: false }).level,
  "phone",
);
expect(
  "one thread inside the Hub page is still the worse news on a phone",
  deviceVerdict({ gpu: true, phone: true, threads: 1, framed: true }).level,
  "single-framed",
);
expect(
  "?backend=webgpu on a phone is the GPU verdict",
  deviceVerdict({ gpu: true, phone: true, forceGpu: true, threads: 4, framed: false }).level,
  "gpu",
);
expect(
  "a desktop is unaffected",
  deviceVerdict({ gpu: true, phone: false, threads: 4, framed: false }).level,
  "gpu",
);
for (const lang of ["az", "en"]) {
  const t = (key, values) => label(lang, key, values);
  const copy = deviceCopy(deviceVerdict({ gpu: true, phone: true, threads: 4 }), t);
  expect(`[${lang}] the phone card has its own title`, copy.title, t("device_title_phone"));
  expect(`[${lang}] the phone text names the threads`, copy.text.includes("4"), true);
  expect(`[${lang}] the phone note says why WebGPU is off`, copy.note, t("device_note_phone"));
}

// -- iPhone and iPad: every browser there is WebKit, so "try Chrome" is no advice ------

const IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/140.0.0.0 Mobile/15E148 Safari/604.1";
const MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15";
expect("Chrome on an iPhone is Apple mobile", isAppleMobile({ userAgent: IPHONE }), true);
expect("an iPad in desktop mode: a Mac with a touch screen", isAppleMobile({ userAgent: MAC, maxTouchPoints: 5 }), true);
expect("a real Mac", isAppleMobile({ userAgent: MAC, maxTouchPoints: 0 }), false);
expect("Android is not Apple", isAppleMobile({ userAgent: "Mozilla/5.0 (Linux; Android 13) Mobile" }), false);
expect("a worker failure on iOS gets the iOS advice", adviceKey("worker", { appleMobile: true }), "error_next_worker_ios");
expect("so does a backend failure", adviceKey("backend", { appleMobile: true }), "error_next_backend_ios");
expect("a network failure needs no iOS wording", adviceKey("network", { appleMobile: true }), "error_next_network");
expect("elsewhere the usual advice", adviceKey("worker"), "error_next_worker");
expect("one thread on iOS keeps its level", deviceVerdict({ threads: 1, appleMobile: true }).level, "single");
for (const lang of ["az", "en"]) {
  const t = (key, values) => label(lang, key, values);
  for (const key of ["error_next_worker_ios", "error_next_backend_ios", "device_text_single_ios"]) {
    expect(`[${lang}] ${key} exists`, t(key) !== key, true);
    expect(`[${lang}] ${key} does not send an iPhone to Chrome`, /Chrome|Edge/u.test(t(key)), false);
  }
  expect(
    `[${lang}] one thread on iOS reads the iOS text`,
    deviceCopy(deviceVerdict({ threads: 1, appleMobile: true }), t).text,
    t("device_text_single_ios"),
  );
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  device-check: ${checked} checks on the device verdict and error advice`);
