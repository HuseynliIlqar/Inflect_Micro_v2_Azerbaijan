// The device check's verdict: which backend passed, which is fastest, what to say.
import { decide, judgeTrial, speedTier } from "../js/engine/calibration-verdict.js";
import { REFERENCE } from "../js/engine/calibration-reference.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

const GPU = { backend: "webgpu", precision: "fp32" };
const INT8 = { backend: "wasm", precision: "int8" };
const CPU = { backend: "wasm", precision: "fp32" };
const AUDIO_MS = (REFERENCE.fp32.samples / REFERENCE.sampleRate) * 1000;

const good = (precision) => ({ ...REFERENCE[precision], finite: true });
// Three timed runs at `rtf` x the clip's length, each with a correct fingerprint.
const passing = (step, rtf) => ({
  step,
  runs: [1, 1, 1].map(() => ({ ms: rtf * AUDIO_MS, print: good(step.precision) })),
});
const OK = { status: "ok" };
const base = { threads: 4, framed: false, reference: REFERENCE, measuredAt: 1 };
const stepOf = (s) => (s ? `${s.backend}/${s.precision}` : null);
const planOf = (c) => c.plan.map(stepOf);

// judgeTrial
{
  const trial = {
    step: INT8,
    runs: [300, 100, 200].map((ms) => ({ ms, print: good("int8") })),
  };
  const judged = judgeTrial(trial, REFERENCE);
  expect("the median run is the speed", judged.rtf, Number((200 / AUDIO_MS).toFixed(3)));
  expect("a correct trial passes", [judged.ok, judged.reason], [true, null]);
}
{
  const runs = [0, 256 * 9].map((extra) => ({ ms: 100, print: { ...good("fp32"), samples: REFERENCE.fp32.samples + extra } }));
  expect("runs of different lengths are unstable", judgeTrial({ step: GPU, runs }, REFERENCE).reason, "unstable");
}
{
  const runs = [1, 2, 3].map(() => ({ ms: 100, print: { ...good("fp32"), samples: REFERENCE.fp32.samples + 2160 } }));
  expect("the same wrong length each time is a wrong length", judgeTrial({ step: GPU, runs }, REFERENCE).reason, "length");
}
expect(
  "a step that failed to load",
  judgeTrial({ step: GPU, error: { stage: "load", message: "no adapter" }, runs: [] }, REFERENCE),
  { backend: "webgpu", precision: "fp32", ok: false, reason: "load", rtf: null, message: "no adapter" },
);

// speedTier
expect("fast", speedTier(0.3), "fast");
expect("usable", speedTier(1.2), "usable");
expect("slow", speedTier(2.5), "slow");
expect("nothing measured", speedTier(null), "failed");

// decide: a desktop with a working GPU
{
  const c = decide({ ...base, probe: OK, trials: [passing(INT8, 0.4), passing(GPU, 0.1)] });
  expect("desktop: the GPU", stepOf(c.chosen), "webgpu/fp32");
  expect("desktop: fast", c.tier, "fast");
  expect("desktop: gpu chosen", c.gpu, "chosen");
  expect("desktop: plan, fastest first, CPU fp32 last", planOf(c), ["webgpu/fp32", "wasm/int8", "wasm/fp32"]);
  expect("a sentence is four seconds of speech", c.sentenceSeconds, 0.4);
  expect("carries the reference's version", c.version, REFERENCE.version);
}

// the Redmi Note 10 Pro: an adapter, and wrong durations
{
  const wrong = {
    step: GPU,
    runs: [1, 2, 3].map(() => ({ ms: 400, print: { ...good("fp32"), samples: REFERENCE.fp32.samples + 5760 } })),
  };
  const c = decide({ ...base, probe: OK, trials: [passing(INT8, 1.2), wrong] });
  expect("redmi: the CPU", stepOf(c.chosen), "wasm/int8");
  expect("redmi: gpu computed wrong audio", c.gpu, "wrong-output");
  expect("redmi: usable", c.tier, "usable");
  expect("redmi: the failed GPU is not in the plan", planOf(c), ["wasm/int8", "wasm/fp32"]);
}

// a GPU that works but is no faster
{
  const c = decide({ ...base, probe: OK, trials: [passing(INT8, 1.0), passing(GPU, 0.9)] });
  expect("barely faster GPU: the CPU is kept", stepOf(c.chosen), "wasm/int8");
  expect("barely faster GPU: slower", c.gpu, "slower");
  expect("a GPU that passed stays in the plan as a fallback", planOf(c), ["wasm/int8", "webgpu/fp32", "wasm/fp32"]);
  const forced = decide({ ...base, forceGpu: true, probe: OK, trials: [passing(INT8, 1.0), passing(GPU, 0.9)] });
  expect("?backend=webgpu takes a GPU that passed", stepOf(forced.chosen), "webgpu/fp32");
}

// no GPU at all
{
  const c = decide({ ...base, threads: 1, probe: { status: "no-adapter" }, trials: [passing(INT8, 2.5)] });
  expect("no adapter: said so", c.gpu, "no-adapter");
  expect("no adapter: slow", c.tier, "slow");
  expect("no adapter: plan", planOf(c), ["wasm/int8", "wasm/fp32"]);
  const off = decide({ ...base, probe: { status: "off" }, trials: [passing(INT8, 0.4)] });
  expect("?backend=wasm: off", off.gpu, "off");
}

// a GPU that throws
{
  const c = decide({
    ...base,
    probe: OK,
    trials: [passing(INT8, 1), { step: GPU, error: { stage: "run", message: "device lost" }, runs: [] }],
  });
  expect("a GPU that throws: error", c.gpu, "error");
  expect("the error message is kept", c.gpuMessage, "device lost");
}

// int8 wrong, fp32 on the CPU right
{
  const broken = { step: INT8, runs: [{ ms: 100, print: { ...good("int8"), finite: false } }] };
  const c = decide({ ...base, probe: { status: "unsupported" }, trials: [broken, passing(CPU, 2)] });
  expect("int8 broken: fp32 on the CPU", stepOf(c.chosen), "wasm/fp32");
  expect("int8 broken: said so", c.int8Failed, true);
  expect("int8 broken: plan", planOf(c), ["wasm/fp32"]);
}

// nothing works
{
  const broken = { step: INT8, error: { stage: "load", message: "out of memory" }, runs: [] };
  const c = decide({ ...base, probe: { status: "unsupported" }, trials: [broken] });
  expect("nothing passed: failed", [c.tier, c.chosen, c.rtf], ["failed", null, null]);
  expect("nothing passed: the CPU fp32 step is still tried", planOf(c), ["wasm/fp32"]);
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  calibration-verdict: ${checked} checks on the device check's verdict`);
