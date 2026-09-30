// The device check end to end -- against a stand-in onnxruntime, no model, no GPU.
import { Engine } from "../js/engine/engine.js";
import { calibrate } from "../js/engine/calibrate.js";
import { decide } from "../js/engine/calibration-verdict.js";
import { probeGpu, adapterKey } from "../js/engine/gpu-probe.js";
import { knownSteps, voicePlan } from "../js/engine/backend-plan.js";
import { REFERENCE } from "../js/engine/calibration-reference.js";

console.warn = () => {};

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

/** A waveform whose fingerprint is the reference's for `precision`. */
function referenceWave(precision) {
  const { samples, rms, envelope } = REFERENCE[precision];
  const { window } = REFERENCE;
  const out = new Float32Array(samples);
  let energy = 0;
  envelope.forEach((level, block) => {
    for (let i = 0; i < window; i += 1) out[block * window + i] = i % 2 ? level : -level;
    energy += level * level * window;
  });
  const tail = samples - envelope.length * window;
  const level = Math.sqrt(Math.max(0, rms * rms * samples - energy) / tail);
  for (let i = envelope.length * window; i < samples; i += 1) out[i] = i % 2 ? level : -level;
  return out;
}

/**
 * A fake onnxruntime whose clock moves by `cost[step]` ms per decode run.
 * `wrong[step]` adds samples to that step's output; `breakLoad` fails a load.
 */
function fakeOrt({ cost = {}, wrong = {}, breakLoad = [] } = {}) {
  const clock = { now: 0 };
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  }
  const InferenceSession = {
    async create(path, { executionProviders: [backend] }) {
      const precision = path.includes("int8") ? "int8" : "fp32";
      const step = `${backend}/${precision}`;
      if (path.includes("decode") && breakLoad.includes(step)) throw new Error(`cannot load ${step}`);
      return {
        async run(feeds) {
          if ("tokens" in feeds) {
            const zeros = new Float32Array(8);
            return {
              m_p_exp: new Tensor("float32", zeros, [1, 2, 4]),
              logs_p_exp: new Tensor("float32", zeros, [1, 2, 4]),
              y_mask: new Tensor("float32", new Float32Array(4).fill(1), [1, 1, 4]),
            };
          }
          clock.now += cost[step] ?? 100;
          const wave = referenceWave(precision);
          const extra = wrong[step] ?? 0;
          const data = extra ? Float32Array.from([...wave, ...new Float32Array(extra).fill(0.1)]) : wave;
          return { waveform: { data } };
        },
        async release() {},
      };
    },
  };
  return { Tensor, InferenceSession, clock };
}

const engineWith = (ort, extra = {}) =>
  new Engine(ort, {
    durationPath: "duration.onnx",
    decodePath: "decode.onnx",
    int8DecodePath: "decode.int8.onnx",
    phonemize: async () => "salam",
    ...extra,
  });
const stepOf = (s) => (s ? `${s.backend}/${s.precision}` : null);
const audioMs = (REFERENCE.fp32.samples / REFERENCE.sampleRate) * 1000;

async function check(ort, probe, extra = {}) {
  const engine = engineWith(ort, extra.engine);
  const events = [];
  const trials = await calibrate(engine, {
    reference: REFERENCE,
    probe,
    runs: 3,
    now: () => ort.clock.now,
    onProgress: (stage, detail) => events.push({ stage, ...detail }),
    ...extra.options,
  });
  const verdict = decide({ probe, trials, reference: REFERENCE, threads: 4, framed: false, measuredAt: 0 });
  return { engine, trials, verdict, events };
}

// A desktop: the GPU is measured after the CPU and wins.
{
  const ort = fakeOrt({ cost: { "webgpu/fp32": audioMs * 0.1, "wasm/int8": audioMs * 0.4 } });
  const { trials, verdict, events } = await check(ort, { status: "ok" });
  expect("desktop: CPU first, then the GPU", trials.map((t) => stepOf(t.step)), ["wasm/int8", "webgpu/fp32"]);
  expect("desktop: three timed runs each, warm-up not counted", trials.map((t) => t.runs.length), [3, 3]);
  expect("desktop: the GPU wins", stepOf(verdict.chosen), "webgpu/fp32");
  expect("desktop: its speed", verdict.rtf, 0.1);
  expect("the page is told the candidates first", events[0], {
    stage: "candidates",
    steps: [{ backend: "wasm", precision: "int8" }, { backend: "webgpu", precision: "fp32" }],
  });
  const tests = events.filter((e) => e.stage === "test").map((e) => `${e.backend}:${e.run}/${e.runs}`);
  expect("each run is announced, warm-up as 0", tests.slice(0, 4), ["wasm:0/3", "wasm:1/3", "wasm:2/3", "wasm:3/3"]);
  const tested = events.filter((e) => e.stage === "tested").map((e) => [e.backend, e.ok]);
  expect("each trial's result is announced", tested, [["wasm", true], ["webgpu", true]]);
}

// The Redmi Note 10 Pro: the GPU loads, runs, and gets the length wrong.
{
  const ort = fakeOrt({ wrong: { "webgpu/fp32": 5760 }, cost: { "wasm/int8": audioMs * 1.2 } });
  const { verdict } = await check(ort, { status: "ok" });
  expect("redmi: the CPU", stepOf(verdict.chosen), "wasm/int8");
  expect("redmi: the GPU's audio was wrong", verdict.gpu, "wrong-output");
}

// No adapter: the GPU is not even loaded.
{
  const ort = fakeOrt();
  const { trials, verdict } = await check(ort, { status: "no-adapter" });
  expect("no adapter: only the CPU is measured", trials.map((t) => stepOf(t.step)), ["wasm/int8"]);
  expect("no adapter: said so", verdict.gpu, "no-adapter");
}

// int8 will not load: fp32 on the CPU is measured in its place.
{
  const ort = fakeOrt({ breakLoad: ["wasm/int8"] });
  const { trials, verdict } = await check(ort, { status: "unsupported" });
  expect("int8 broken: fp32 measured next", trials.map((t) => stepOf(t.step)), ["wasm/int8", "wasm/fp32"]);
  expect("int8 broken: the load error is kept", trials[0].error.stage, "load");
  expect("int8 broken: fp32 chosen", stepOf(verdict.chosen), "wasm/fp32");
}

// Full quality asked for: the CPU step measured is fp32.
{
  const ort = fakeOrt();
  const { trials } = await check(ort, { status: "unsupported" }, { options: { fullQuality: true } });
  expect("full quality: fp32 on the CPU", trials.map((t) => stepOf(t.step)), ["wasm/fp32"]);
}

// The engine takes the verdict's plan.
{
  const ort = fakeOrt();
  const engine = engineWith(ort);
  await engine.setPlan([{ backend: "webgpu", precision: "fp32" }, { backend: "wasm", precision: "int8" }, { backend: "wasm", precision: "fp32" }]);
  await engine.load();
  expect("setPlan: the engine starts on its first step", stepOf(engine), "webgpu/fp32");
  await engine.setPlan([{ backend: "wasm", precision: "int8" }, { backend: "wasm", precision: "fp32" }]);
  expect("a new first step drops the loaded sessions", engine.decode, null);
  await engine.load();
  expect("and loads the new one", stepOf(engine), "wasm/int8");
  const decode = engine.decode;
  await engine.setPlan([{ backend: "wasm", precision: "int8" }, { backend: "wasm", precision: "fp32" }]);
  expect("the same plan keeps the sessions", engine.decode === decode, true);
}

// A plan whose first step is already loaded keeps the sessions but takes over the chain.
{
  const engine = engineWith(fakeOrt());
  await engine.load();
  const decode = engine.decode;
  const measured = [{ backend: "wasm", precision: "int8" }, { backend: "webgpu", precision: "fp32" }, { backend: "wasm", precision: "fp32" }];
  await engine.setPlan(measured);
  expect("kept sessions", engine.decode === decode, true);
  expect("the chain walked on a failure is the measured one", engine.plan.map(stepOf), measured.map(stepOf));
}

// A GPU that never finishes loading is a failed trial, not a hung check.
{
  const ort = fakeOrt();
  const engine = engineWith(ort);
  const create = ort.InferenceSession.create;
  ort.InferenceSession.create = (path, options) =>
    (options.executionProviders[0] === "webgpu" ? new Promise(() => {}) : create(path, options));
  const trials = await calibrate(engine, {
    reference: REFERENCE, probe: { status: "ok" }, runs: 1, now: () => ort.clock.now, loadTimeoutMs: 50,
  });
  expect("a hung GPU load times out", trials.map((t) => t.error?.stage ?? "ok"), ["ok", "timeout"]);
}

// voicePlan: the verdict applied to a voice.
{
  const GPU = { backend: "webgpu", precision: "fp32" };
  const INT8 = { backend: "wasm", precision: "int8" };
  const CPU = { backend: "wasm", precision: "fp32" };
  const plan = [GPU, INT8, CPU];
  expect("a voice with int8 keeps the plan", voicePlan(plan, { hasInt8: true }), plan);
  expect("a voice without int8 drops that step", voicePlan(plan, { hasInt8: false }), [GPU, CPU]);
  expect("full quality drops int8", voicePlan([INT8, GPU, CPU], { hasInt8: true, fullQuality: true }), [GPU, CPU]);
  expect("CPU fp32 is always last", voicePlan([CPU, INT8], { hasInt8: true }), [INT8, CPU]);
  expect("and always there", voicePlan([GPU], { hasInt8: false }), [GPU, CPU]);
}

// knownSteps: a plan from storage is not trusted.
{
  const GPU = { backend: "webgpu", precision: "fp32" };
  const INT8 = { backend: "wasm", precision: "int8" };
  expect("a good plan passes", knownSteps([GPU, INT8]), [GPU, INT8]);
  expect("unknown steps are dropped", knownSteps([{ backend: "webgl", precision: "fp16" }, INT8]), [INT8]);
  expect("extra fields are dropped", knownSteps([{ ...INT8, evil: 1 }]), [INT8]);
  expect("duplicates are dropped", knownSteps([INT8, INT8]), [INT8]);
  expect("not a list", knownSteps("webgpu"), null);
  expect("nothing known", knownSteps([{}]), null);
}

// setPlan(null) twice keeps the guessed chain's sessions.
{
  const engine = engineWith(fakeOrt());
  await engine.setPlan(null);
  await engine.load();
  const decode = engine.decode;
  await engine.setPlan(null);
  expect("no plan, and still none: the sessions stay", engine.decode === decode, true);
}

// probeGpu, against stand-in navigators.
{
  const adapter = (info = {}, device = async () => ({ destroy() {} })) => ({ info, requestDevice: device });
  const nav = (gpu) => ({ gpu });
  expect("no navigator.gpu", (await probeGpu({ nav: {} })).status, "unsupported");
  expect("?backend=wasm", (await probeGpu({ nav: nav({}), preferWasm: true })).status, "off");
  expect("no adapter", (await probeGpu({ nav: nav({ requestAdapter: async () => null }) })).status, "no-adapter");
  expect(
    "a software adapter",
    (await probeGpu({ nav: nav({ requestAdapter: async () => adapter({ isFallbackAdapter: true }) }) })).status,
    "software",
  );
  const refused = adapter({}, async () => {
    throw new Error("device refused");
  });
  const failed = await probeGpu({ nav: nav({ requestAdapter: async () => refused }) });
  expect("a device that will not start", [failed.status, failed.message], ["device-failed", "device refused"]);
  const ok = await probeGpu({
    nav: nav({ requestAdapter: async () => adapter({ vendor: "qualcomm", architecture: "adreno-600" }) }),
  });
  expect("a working adapter", ok, { status: "ok", vendor: "qualcomm", architecture: "adreno-600" });
  expect("the adapter's identity keys the stored verdict", adapterKey(ok), "ok:qualcomm:adreno-600");
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  calibrate: ${checked} checks on the device check against a stand-in runtime`);
