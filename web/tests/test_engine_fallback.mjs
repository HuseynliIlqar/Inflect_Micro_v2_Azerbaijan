// Engine walks the fallback chain -- against a stand-in onnxruntime, no model.
import { Engine } from "../js/tts.js";

// Engine warns on every step it abandons; here that is the point, not noise.
console.warn = () => {};

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

/**
 * A fake onnxruntime. `breakLoad` and `breakRun` name the steps that fail,
 * as "backend/precision", with precision read from the decoder's path.
 */
function fakeOrt({ breakLoad = [], breakRun = [], flakyRun = {} } = {}) {
  const created = [];
  // flakyRun: { "backend/precision": n } -- the first n runs of that step fail.
  const flaky = { ...flakyRun };
  class Tensor {
    constructor(type, data, dims) {
      Object.assign(this, { type, data, dims });
    }
  }
  const InferenceSession = {
    async create(path, { executionProviders: [backend] }) {
      const precision = path.includes("int8") ? "int8" : "fp32";
      const step = `${backend}/${precision}`;
      created.push(`${step}:${path}`);
      if (path.includes("decode") && breakLoad.includes(step)) throw new Error(`cannot load ${step}`);
      return {
        async run(feeds) {
          if ("tokens" in feeds) {
            const frames = 4;
            const zeros = new Float32Array(2 * frames);
            return {
              m_p_exp: new Tensor("float32", zeros, [1, 2, frames]),
              logs_p_exp: new Tensor("float32", zeros, [1, 2, frames]),
              y_mask: new Tensor("float32", new Float32Array(frames).fill(1), [1, 1, frames]),
            };
          }
          if (breakRun.includes(step)) throw new Error(`cannot run ${step}`);
          if (flaky[step] > 0) {
            flaky[step] -= 1;
            throw new Error(`hiccup on ${step}`);
          }
          return { waveform: { data: new Float32Array(1024).fill(precision === "int8" ? 0.25 : 0.5) } };
        },
        async release() {},
      };
    },
  };
  return { Tensor, InferenceSession, created };
}

const engineWith = (ort, extra = {}) =>
  new Engine(ort, {
    durationPath: "duration.onnx",
    decodePath: "decode.onnx",
    int8DecodePath: "decode.int8.onnx",
    phonemize: async () => "salam",
    ...extra,
  });

const stepOf = (engine) => `${engine.backend}/${engine.precision}`;
const stagesOf = (engine) => engine.failures.map((f) => `${f.backend}/${f.precision}:${f.stage}`);

// No WebGPU in Node, so every chain here starts on the CPU.

{
  const engine = engineWith(fakeOrt());
  await engine.load();
  expect("no GPU: starts on int8", stepOf(engine), "wasm/int8");
  expect("no failures on the happy path", engine.failures, []);
}

{
  const engine = engineWith(fakeOrt(), { int8DecodePath: null });
  await engine.load();
  expect("no int8 graph: fp32 on the CPU", stepOf(engine), "wasm/fp32");
}

{
  const engine = engineWith(fakeOrt(), { fullQuality: true });
  await engine.load();
  expect("full quality: skips int8", stepOf(engine), "wasm/fp32");
}

{
  const engine = engineWith(fakeOrt({ breakLoad: ["wasm/int8"] }));
  await engine.load();
  expect("int8 fails to load: falls to fp32", stepOf(engine), "wasm/fp32");
  expect("the load failure is recorded", stagesOf(engine), ["wasm/int8:load"]);
  expect("the reason is kept", engine.failures[0].message, "cannot load wasm/int8");
}

{
  const engine = engineWith(fakeOrt({ breakRun: ["wasm/int8"] }));
  const seen = [];
  const { waveform } = await engine.load().then(() =>
    engine.speak("Salam.", { onBackend: (state) => seen.push(`${state.backend}/${state.precision}`) }),
  );
  expect("int8 fails at run time: finishes on fp32", stepOf(engine), "wasm/fp32");
  expect("the page is told about the switch", seen, ["wasm/fp32"]);
  expect("the audio comes from the fp32 decoder", waveform.some((v) => v === 0.5), true);
  expect("the run failure is recorded", stagesOf(engine), ["wasm/int8:run"]);
}

{
  const engine = engineWith(fakeOrt({ breakLoad: ["wasm/int8", "wasm/fp32"] }));
  let message = "";
  try {
    await engine.load();
  } catch (error) {
    message = error.message;
  }
  expect("nothing loads: one clear error", message.startsWith("no backend could load the model"), true);
  expect("it names every step that failed", message.includes("wasm/int8") && message.includes("wasm/fp32"), true);
}

{
  const engine = engineWith(fakeOrt({ breakRun: ["wasm/int8", "wasm/fp32"] }));
  await engine.load();
  let message = "";
  try {
    await engine.speak("Salam.");
  } catch (error) {
    message = error.message;
  }
  expect("the last step failing at run time rethrows", message, "cannot run wasm/fp32");
  expect("both run failures are recorded", stagesOf(engine), ["wasm/int8:run", "wasm/fp32:run"]);
}

// With WebGPU: define an adapter for the length of these cases.
Object.defineProperty(globalThis.navigator, "gpu", {
  configurable: true,
  value: { requestAdapter: async () => ({}) },
});

{
  const engine = engineWith(fakeOrt());
  await engine.load();
  expect("with a GPU: starts on WebGPU", stepOf(engine), "webgpu/fp32");
}

{
  const engine = engineWith(fakeOrt({ breakLoad: ["webgpu/fp32"] }));
  await engine.load();
  expect("WebGPU fails to load: int8 on the CPU", stepOf(engine), "wasm/int8");
}

{
  const engine = engineWith(fakeOrt({ breakRun: ["webgpu/fp32"] }));
  await engine.load();
  await engine.speak("Salam.");
  expect("WebGPU fails at run time: int8 on the CPU", stepOf(engine), "wasm/int8");
}

{
  const engine = engineWith(fakeOrt(), { preferWasm: true });
  await engine.load();
  expect("?backend=wasm skips the GPU", stepOf(engine), "wasm/int8");
}

{
  const engine = engineWith(fakeOrt({ breakRun: ["webgpu/fp32"] }));
  await engine.load();
  await engine.speak("Salam.");
  await engine.setFullQuality(true);
  expect("leaving int8 releases the sessions; they reload on the next request", engine.decode, null);
  await engine.speak("Salam.");
  expect("full quality then speaks on the CPU in fp32, not on the broken GPU", stepOf(engine), "wasm/fp32");
  expect("the GPU failure is still reported", stagesOf(engine), ["webgpu/fp32:run"]);
}

{
  const ort = fakeOrt();
  const engine = engineWith(ort);
  await engine.load();
  const before = ort.created.length;
  await engine.setFullQuality(true);
  await engine.speak("Salam.");
  expect("full quality on WebGPU: nothing to reload", ort.created.length, before);
}

delete globalThis.navigator.gpu;

{
  const engine = engineWith(fakeOrt({ flakyRun: { "wasm/int8": 1 } }));
  await engine.load();
  await engine.speak("Salam.");
  expect("one hiccup: retried on the same step", stepOf(engine), "wasm/int8");
  expect("a hiccup that recovered is not a fallback", engine.failures, []);
}

{
  const engine = engineWith(fakeOrt({ flakyRun: { "wasm/fp32": 2 } }), { int8DecodePath: null });
  await engine.load();
  let first = "";
  try {
    await engine.speak("Salam.");
  } catch (error) {
    first = error.message;
  }
  await engine.speak("Salam.");
  expect("the last step failing twice rethrows", first, "hiccup on wasm/fp32");
  expect("a later success does not carry the old failure", engine.failures, []);
}

{
  const engine = engineWith(fakeOrt({ breakRun: ["wasm/int8"], breakLoad: ["wasm/fp32"] }));
  await engine.load();
  let message = "";
  try {
    await engine.speak("Salam.");
  } catch (error) {
    message = error.message;
  }
  expect("no step left to load: the run error is what the page sees", message, "cannot run wasm/int8");
  expect("and the broken sessions are released", engine.decode, null);
}

{
  const ort = fakeOrt();
  const engine = engineWith(ort, { int8DecodePath: null });
  await engine.load();
  const before = ort.created.length;
  await engine.setFullQuality(true);
  expect("no int8 graph (English): full quality changes nothing", ort.created.length, before);
  expect("and keeps the sessions", engine.decode !== null, true);
}

{
  const engine = engineWith(fakeOrt());
  await engine.load();
  await engine.setFullQuality(true);
  await engine.load();
  expect("full quality on int8: reloads on fp32", stepOf(engine), "wasm/fp32");
  await engine.setFullQuality(false);
  await engine.load();
  expect("and back to int8", stepOf(engine), "wasm/int8");
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  engine-fallback: ${checked} checks against a stand-in onnxruntime`);
