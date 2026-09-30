/** The device check on the page: said on load, revised if the GPU falls through. */

import { hasWebGpu } from "../engine/engine.js";
import { deviceVerdict } from "../ui/device-check.js";
import { createDeviceView } from "../ui/device-view.js";
import { els } from "./dom.js";
import {
  APPLE_MOBILE, DIRECT_URL, EXPECTED_THREADS, IN_FRAME, PHONE, QUERY, state,
} from "./context.js";

export const device = createDeviceView(els, { directUrl: DIRECT_URL });

/** On page load, before any model is fetched: what will this device do? */
export async function checkDevice() {
  const { preferWasm, forceGpu } = QUERY;
  // A phone does not ask for an adapter it will not use.
  const gpu = PHONE && !forceGpu ? false : await hasWebGpu();
  device.announce(deviceVerdict({
    gpu, preferWasm, phone: PHONE, forceGpu, appleMobile: APPLE_MOBILE,
    threads: EXPECTED_THREADS, framed: IN_FRAME,
  }));
}

/** WebGPU was expected and the worker fell back: say so, once. */
export function reviseDevice() {
  if (device.verdict?.level !== "gpu" || state.backend !== "wasm") return;
  const reason = state.fallbacks.find((f) => f.backend === "webgpu")?.message ?? "";
  device.revise(
    deviceVerdict({
      gpu: true, gpuFailed: true, appleMobile: APPLE_MOBILE, threads: state.threads, framed: IN_FRAME,
    }),
    reason,
  );
}
