/**
 * The fallback chain: which backend and precision to try, best first.
 *
 *   1. WebGPU, fp32    -- the full model on the GPU, faster than real time
 *   2. CPU, int8       -- a quantised decoder, ~1.5x faster than fp32 on wasm
 *   3. CPU, fp32       -- always last: slow, but it runs everywhere
 *
 * The int8 decoder keeps the vocoder's last upsampling stage, the flow and the
 * vocoder's first and last convolutions in fp32; that is the variant whose
 * change to the spectrum measured smaller than the model's own difference
 * between two seeds. `tools/quantize_decoder.py` builds it.
 *
 * Phones skip step 1. On a Redmi Note 10 Pro (Adreno 618, Chrome 154) the GPU
 * step ran no faster than int8 on four threads, computed wrong durations (the
 * same sentence and seed came out 4.03 s and 4.12 s long; the CPU gives 3.79 s
 * every time) and lost the WebGPU instance after each use. `?backend=webgpu`
 * still tries it, for testing.
 *
 * Pure, so the order is tested without a browser, a GPU or a model.
 */

const GPU_FP32 = Object.freeze({ backend: "webgpu", precision: "fp32" });
const CPU_INT8 = Object.freeze({ backend: "wasm", precision: "int8" });
const CPU_FP32 = Object.freeze({ backend: "wasm", precision: "fp32" });

/**
 * @param {{ gpu?: boolean, preferWasm?: boolean, phone?: boolean, forceGpu?: boolean,
 *           hasInt8?: boolean, fullQuality?: boolean }} options
 *   gpu: a WebGPU adapter exists; preferWasm: `?backend=wasm`; phone: see
 *   `isPhone`; forceGpu: `?backend=webgpu`, which keeps the GPU on a phone;
 *   hasInt8: this voice ships an int8 decoder; fullQuality: skip the int8 step.
 * @returns {{ backend: string, precision: string }[]}
 */
export function fallbackPlan({
  gpu = false, preferWasm = false, phone = false, forceGpu = false, hasInt8 = false, fullQuality = false,
} = {}) {
  return [
    ...(gpu && !preferWasm && (!phone || forceGpu) ? [GPU_FP32] : []),
    ...(hasInt8 && !fullQuality ? [CPU_INT8] : []),
    CPU_FP32,
  ];
}

/** @returns {boolean} */
export function sameStep(a, b) {
  return a?.backend === b?.backend && a?.precision === b?.precision;
}

/** The steps still to try once `step` has failed; the whole plan for an unknown step. */
export function stepsAfter(plan, step) {
  const index = plan.findIndex((candidate) => sameStep(candidate, step));
  return index === -1 ? plan : plan.slice(index + 1);
}

/**
 * The page's URL switches for testers: `?backend=wasm` skips WebGPU,
 * `?backend=webgpu` uses it even on a phone, and `?precision=fp32` skips the
 * int8 step. Anything else is ignored.
 */
export function planOptionsFromQuery(search) {
  const params = new URLSearchParams(search);
  return {
    preferWasm: params.get("backend") === "wasm",
    fullQuality: params.get("precision") === "fp32",
    forceGpu: params.get("backend") === "webgpu",
  };
}

/**
 * Whether this is a phone or tablet. `mobile` is `navigator.userAgentData.mobile`,
 * which only Chromium has; Safari and Firefox are read from the user agent.
 * An iPad in its default desktop mode reports a Mac and is not caught -- it
 * keeps WebGPU, which is acceptable: nothing measured says Apple GPUs fail.
 *
 * @param {{ userAgent?: string, mobile?: boolean }} device
 */
export function isPhone({ userAgent = "", mobile } = {}) {
  if (mobile === true) return true;
  return /Android|iPhone|iPad|iPod|Mobile/u.test(userAgent);
}

/** `isPhone` for the page or worker it runs in. */
export function thisIsPhone(nav = globalThis.navigator) {
  return isPhone({ userAgent: nav?.userAgent ?? "", mobile: nav?.userAgentData?.mobile });
}
