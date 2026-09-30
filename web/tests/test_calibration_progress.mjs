// The device check's screen: the bar, the stage line and the list of steps.
import { advance, finish, startProgress } from "../js/ui/calibration-progress.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

const INT8 = { backend: "wasm", precision: "int8" };
const GPU = { backend: "webgpu", precision: "fp32" };
const rows = (state) => state.rows.map((row) => `${row.id}:${row.status}`);

const steps = [
  { stage: "probe" },
  { stage: "probed", gpu: "ok" },
  { stage: "candidates", steps: [INT8, GPU] },
  { stage: "download", ...INT8, loaded: 11e6, total: 22e6 },
  { stage: "compile", ...INT8 },
  { stage: "test", ...INT8, run: 0, runs: 3 },
  { stage: "test", ...INT8, run: 3, runs: 3 },
  { stage: "tested", ...INT8, ok: true, reason: null, rtf: 0.42 },
  { stage: "download", ...GPU, loaded: 0, total: 38e6 },
  { stage: "test", ...GPU, run: 1, runs: 3 },
  { stage: "tested", ...GPU, ok: false, reason: "length", rtf: 0.3 },
  { stage: "finalise" },
  { stage: "compile", ...INT8 },
];

const start = startProgress();
expect("starts on the GPU check", rows(start), ["probe:active"]);
expect("starts at zero", start.fraction, 0);

const seen = [start];
for (const message of steps) seen.push(advance(seen.at(-1), message));
const at = (stage, extra = {}) => seen[steps.findIndex((m) => m.stage === stage && Object.entries(extra).every(([k, v]) => m[k] === v)) + 1];

expect("the probe's answer", at("probed").rows[0], { id: "probe", status: "done", label: "cal_row_probe", detail: { key: "cal_probe_ok", values: {} } });
expect("a row per backend, then the result", rows(at("candidates")), [
  "probe:done", "wasm/int8:pending", "webgpu/fp32:pending", "finalise:pending",
]);
expect("downloading marks the row active", rows(at("download", { backend: "wasm" }))[1], "wasm/int8:active");
expect(
  "the stage line names the download",
  at("download", { backend: "wasm" }).stage,
  { key: "cal_stage_download", values: { loaded: "11.0", total: "22.0" }, backendKey: "cal_backend_cpu" },
);
expect(
  "the warm-up has its own words",
  at("test", { run: 0 }).stage,
  { key: "cal_stage_warmup", values: {}, backendKey: "cal_backend_cpu" },
);
expect(
  "a timed run says which",
  at("test", { run: 3 }).stage,
  { key: "cal_stage_test", values: { run: 3, runs: 3 }, backendKey: "cal_backend_cpu" },
);
expect(
  "a passed backend shows its speed",
  at("tested", { backend: "wasm" }).rows[1],
  { id: "wasm/int8", status: "done", label: "cal_row_cpu", detail: { key: "cal_detail_speed", values: { rtf: "0.42" } } },
);
expect(
  "a failed one says why",
  at("tested", { backend: "webgpu" }).rows[2].detail,
  { key: "cal_detail_wrong", values: {} },
);
expect("and is marked failed", at("tested", { backend: "webgpu" }).rows[2].status, "failed");
expect("finalising", rows(at("finalise")).at(-1), "finalise:active");
expect("loading the winner stays on the finalise line", seen.at(-1).stage.key, "cal_stage_finalise");

const fractions = seen.map((state) => state.fraction);
expect("the bar never goes back", fractions.every((f, i) => i === 0 || f >= fractions[i - 1]), true);
expect("and stays below full until the end", fractions.every((f) => f < 1), true);
expect("finish fills it", finish(seen.at(-1)).fraction, 1);
expect("finish marks the last row done", rows(finish(seen.at(-1))).at(-1), "finalise:done");
expect("advance does not change the state it was given", rows(seen[2]), ["probe:done"]);

// int8 fails: the fp32 CPU row appears before the result row.
{
  let state = startProgress();
  for (const message of [
    { stage: "probed", gpu: "unsupported" },
    { stage: "candidates", steps: [INT8] },
    { stage: "tested", ...INT8, ok: false, reason: "load", rtf: null },
    { stage: "download", backend: "wasm", precision: "fp32", loaded: 1, total: 2 },
  ]) state = advance(state, message);
  expect("no GPU: the probe row says so", state.rows[0].detail.key, "cal_probe_none");
  expect("no GPU: the probe row is not a failure", state.rows[0].status, "skipped");
  expect("a new row for the fallback", rows(state), [
    "probe:skipped", "wasm/int8:failed", "wasm/fp32:active", "finalise:pending",
  ]);
  expect("an error reads as an error", state.rows[1].detail.key, "cal_detail_error");
}

expect(
  "?backend=wasm: switched off, not missing",
  advance(startProgress(), { stage: "probed", gpu: "off" }).rows[0].detail.key,
  "cal_probe_off",
);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  calibration-progress: ${checked} checks on the device check's screen`);
