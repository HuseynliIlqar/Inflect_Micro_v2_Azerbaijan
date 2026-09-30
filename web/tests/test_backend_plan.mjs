// The fallback chain: which backend and precision to try, in what order.
import { fallbackPlan, stepsAfter, planOptionsFromQuery, sameStep } from "../js/backend-plan.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

const GPU = { backend: "webgpu", precision: "fp32" };
const INT8 = { backend: "wasm", precision: "int8" };
const CPU = { backend: "wasm", precision: "fp32" };

// -- the chain -------------------------------------------------------------------

expect("full chain on a WebGPU device", fallbackPlan({ gpu: true, hasInt8: true }), [GPU, INT8, CPU]);
expect("no WebGPU starts on int8", fallbackPlan({ gpu: false, hasInt8: true }), [INT8, CPU]);
expect("no int8 graph skips that step", fallbackPlan({ gpu: true, hasInt8: false }), [GPU, CPU]);
expect("nothing but the CPU", fallbackPlan({ gpu: false, hasInt8: false }), [CPU]);
expect("full quality skips int8", fallbackPlan({ gpu: false, hasInt8: true, fullQuality: true }), [CPU]);
expect(
  "full quality keeps WebGPU, which is already full quality",
  fallbackPlan({ gpu: true, hasInt8: true, fullQuality: true }),
  [GPU, CPU],
);
expect("?backend=wasm skips WebGPU", fallbackPlan({ gpu: true, preferWasm: true, hasInt8: true }), [INT8, CPU]);
expect("defaults are the safe minimum", fallbackPlan({}), [CPU]);
expect("the CPU fp32 step is always last", fallbackPlan({ gpu: true, hasInt8: true }).at(-1), CPU);

// -- moving down the chain --------------------------------------------------------

const full = fallbackPlan({ gpu: true, hasInt8: true });
expect("after WebGPU comes int8", stepsAfter(full, GPU), [INT8, CPU]);
expect("after int8 comes fp32", stepsAfter(full, INT8), [CPU]);
expect("nothing after the last step", stepsAfter(full, CPU), []);
expect("an unknown step leaves the whole plan", stepsAfter(full, { backend: "webgl", precision: "fp32" }), full);
expect("steps compare by value", sameStep({ ...GPU }, GPU), true);
expect("precision matters", sameStep(INT8, CPU), false);

// -- the page's URL switches ------------------------------------------------------

expect("no switches", planOptionsFromQuery(""), { preferWasm: false, fullQuality: false });
expect("?backend=wasm", planOptionsFromQuery("?backend=wasm"), { preferWasm: true, fullQuality: false });
expect("?precision=fp32", planOptionsFromQuery("?precision=fp32"), { preferWasm: false, fullQuality: true });
expect(
  "both, in any order",
  planOptionsFromQuery("?precision=fp32&backend=wasm"),
  { preferWasm: true, fullQuality: true },
);
expect("unknown values are ignored", planOptionsFromQuery("?backend=webgl&precision=int4"), {
  preferWasm: false,
  fullQuality: false,
});

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  backend-plan: ${checked} checks on the fallback chain`);
