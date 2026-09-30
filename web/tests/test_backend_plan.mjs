// The fallback chain: which backend and precision to try, in what order.
import { fallbackPlan, stepsAfter, planOptionsFromQuery, sameStep, isPhone } from "../js/backend-plan.js";

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

expect("no switches", planOptionsFromQuery(""), { preferWasm: false, fullQuality: false, forceGpu: false });
expect("?backend=wasm", planOptionsFromQuery("?backend=wasm"), { preferWasm: true, fullQuality: false, forceGpu: false });
expect("?precision=fp32", planOptionsFromQuery("?precision=fp32"), { preferWasm: false, fullQuality: true, forceGpu: false });
expect("?backend=webgpu", planOptionsFromQuery("?backend=webgpu"), { preferWasm: false, fullQuality: false, forceGpu: true });
expect(
  "both, in any order",
  planOptionsFromQuery("?precision=fp32&backend=wasm"),
  { preferWasm: true, fullQuality: true, forceGpu: false },
);
expect("unknown values are ignored", planOptionsFromQuery("?backend=webgl&precision=int4"), {
  preferWasm: false,
  fullQuality: false,
  forceGpu: false,
});

// -- phones -----------------------------------------------------------------------
// Measured on a Redmi Note 10 Pro (Adreno 618, Chrome 154): the GPU step was no
// faster than int8 on four threads, computed wrong durations, and lost the
// WebGPU instance after each use. Phones start on the CPU.

expect("a phone with WebGPU starts on int8", fallbackPlan({ gpu: true, phone: true, hasInt8: true }), [INT8, CPU]);
expect(
  "?backend=webgpu on a phone keeps the GPU step",
  fallbackPlan({ gpu: true, phone: true, forceGpu: true, hasInt8: true }),
  [GPU, INT8, CPU],
);
expect("forcing the GPU needs an adapter", fallbackPlan({ gpu: false, forceGpu: true, hasInt8: true }), [INT8, CPU]);
expect("?backend=wasm still wins over forcing", fallbackPlan({ gpu: true, preferWasm: true, forceGpu: true, hasInt8: true }), [INT8, CPU]);

const UA = {
  android: "Mozilla/5.0 (Linux; Android 13; 2209116AG) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Mobile Safari/537.36",
  androidReduced: "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Mobile Safari/537.36",
  androidTablet: "Mozilla/5.0 (Linux; Android 14; SM-X710) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
  iphone: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1",
  ipad: "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
  windows: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154.0.0.0 Safari/537.36",
  mac: "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15",
};
expect("an Android phone", isPhone({ userAgent: UA.android }), true);
expect("Chrome's reduced Android user agent", isPhone({ userAgent: UA.androidReduced }), true);
expect("an Android tablet", isPhone({ userAgent: UA.androidTablet }), true);
expect("an iPhone", isPhone({ userAgent: UA.iphone }), true);
expect("an iPad", isPhone({ userAgent: UA.ipad }), true);
expect("a Windows desktop", isPhone({ userAgent: UA.windows }), false);
expect("a Mac", isPhone({ userAgent: UA.mac }), false);
expect("an iPad in desktop mode reads as a Mac and keeps WebGPU", isPhone({ userAgent: UA.mac, mobile: undefined }), false);
expect("the client hint decides when it says mobile", isPhone({ userAgent: UA.windows, mobile: true }), true);
expect("nothing known is not a phone", isPhone({}), false);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  backend-plan: ${checked} checks on the fallback chain`);
