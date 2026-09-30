// Which mode banner the page shows, and the diagnostics line a tester copies.
import { bannerState, badgeParts, diagnosticsLine } from "../js/mode-banner.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

const gpuLoadFailure = { backend: "webgpu", precision: "fp32", stage: "load", message: "no adapter features" };
const int8LoadFailure = { backend: "wasm", precision: "int8", stage: "load", message: "bad graph" };

// -- the banner -------------------------------------------------------------------

expect(
  "WebGPU: no banner",
  bannerState({ voice: "az", backend: "webgpu", precision: "fp32", fallbacks: [], fullQuality: false }),
  { kind: "none", action: null, gpuFailure: null },
);
expect(
  "int8 on the CPU: fast mode, offer full quality",
  bannerState({ voice: "az", backend: "wasm", precision: "int8", fallbacks: [], fullQuality: false }),
  { kind: "fast", action: "full", gpuFailure: null },
);
expect(
  "fp32 on the CPU by choice: full quality, offer fast mode",
  bannerState({ voice: "az", backend: "wasm", precision: "fp32", fallbacks: [], fullQuality: true }),
  { kind: "full", action: "fast", gpuFailure: null },
);
expect(
  "fp32 on the CPU because int8 failed: say so, no toggle",
  bannerState({ voice: "az", backend: "wasm", precision: "fp32", fallbacks: [int8LoadFailure], fullQuality: false }),
  { kind: "int8-failed", action: null, gpuFailure: null },
);
expect(
  "WebGPU failed, now int8: fast mode, with the GPU's reason",
  bannerState({ voice: "az", backend: "wasm", precision: "int8", fallbacks: [gpuLoadFailure], fullQuality: false }),
  { kind: "fast", action: "full", gpuFailure: "no adapter features" },
);
expect(
  "English on the CPU: no int8 exists, no banner",
  bannerState({ voice: "en", backend: "wasm", precision: "fp32", fallbacks: [], fullQuality: false }),
  { kind: "none", action: null, gpuFailure: null },
);
expect(
  "English after a GPU failure: still report the GPU",
  bannerState({ voice: "en", backend: "wasm", precision: "fp32", fallbacks: [gpuLoadFailure], fullQuality: false }),
  { kind: "gpu-failed", action: null, gpuFailure: "no adapter features" },
);
expect(
  "nothing known yet: no banner",
  bannerState({ voice: "az", backend: null, precision: null, fallbacks: [], fullQuality: false }),
  { kind: "none", action: null, gpuFailure: null },
);
expect(
  "missing fallbacks are treated as none",
  bannerState({ voice: "az", backend: "wasm", precision: "int8" }),
  { kind: "fast", action: "full", gpuFailure: null },
);

// -- the badge --------------------------------------------------------------------

expect("GPU badge", badgeParts({ backend: "webgpu", precision: "fp32", threads: 1 }), ["GPU", "fp32"]);
expect("CPU int8 badge, one thread", badgeParts({ backend: "wasm", precision: "int8", threads: 1 }), ["CPU", "int8"]);
expect(
  "CPU badge carries threads",
  badgeParts({ backend: "wasm", precision: "int8", threads: 4 }),
  ["CPU", "int8", { threads: 4 }],
);
expect("unknown backend: no badge", badgeParts({ backend: null, precision: null }), []);

// -- diagnostics ------------------------------------------------------------------

const line = diagnosticsLine({
  voice: "az",
  backend: "wasm",
  precision: "int8",
  threads: 4,
  isolated: true,
  framed: false,
  fullQuality: false,
  fallbacks: [gpuLoadFailure],
  seconds: 3.79,
  elapsed: 1.81,
  chunks: 1,
  userAgent: "Mozilla/5.0 (Linux; Android 14) Chrome/140",
});
expect("one line", line.includes("\n"), false);
expect("names the step that ran", line.includes("backend=wasm/int8"), true);
expect("threads and isolation", line.includes("threads=4") && line.includes("isolated=yes"), true);
expect("the failure and its reason", line.includes("webgpu/fp32:load(no adapter features)"), true);
expect("speed", line.includes("3.79 s in 1.81 s (2.1x)"), true);
expect("the browser", line.endsWith("Mozilla/5.0 (Linux; Android 14) Chrome/140"), true);
expect(
  "no failures says none",
  diagnosticsLine({ voice: "az", backend: "webgpu", precision: "fp32", fallbacks: [] }).includes("fallbacks=none"),
  true,
);
expect(
  "a reason cannot break the line",
  diagnosticsLine({
    voice: "az", backend: "wasm", precision: "fp32",
    fallbacks: [{ backend: "wasm", precision: "int8", stage: "run", message: "line one\nline two" }],
  }).includes("\n"),
  false,
);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  mode-banner: ${checked} checks on the banner, the badge and the diagnostics line`);
