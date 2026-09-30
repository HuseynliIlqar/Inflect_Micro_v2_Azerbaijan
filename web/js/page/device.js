/**
 * The device card: the measured verdict (js/page/calibration.js) or, when the
 * check could not run, a guess from the user agent -- and revised if the
 * backend it promised falls through during synthesis.
 */

import { hasWebGpu } from "../engine/engine.js";
import { sameStep } from "../engine/backend-plan.js";
import { deviceVerdict } from "../ui/device-check.js";
import { createDeviceView } from "../ui/device-view.js";
import { browserStorage, forgetStored } from "../ui/calibration-store.js";
import { els } from "./dom.js";
import {
  APPLE_MOBILE, DIRECT_URL, EXPECTED_THREADS, IN_FRAME, PHONE, QUERY, state,
} from "./context.js";

export const device = createDeviceView(els, { directUrl: DIRECT_URL });

/** Without the device check: what will this device probably do? */
export async function guessDevice() {
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
  const chosen = device.verdict?.calibration?.chosen;
  // The measured backend failed for real: the stored verdict is wrong, so the
  // next visit measures again.
  if (chosen && state.fallbacks.some((failure) => sameStep(failure, chosen))) forgetStored(browserStorage());
  const expectedGpu = device.verdict?.level === "gpu" || chosen?.backend === "webgpu";
  if (!expectedGpu || state.backend !== "wasm") return;
  const reason = state.fallbacks.find((f) => f.backend === "webgpu")?.message ?? "";
  device.revise(
    deviceVerdict({
      gpu: true, gpuFailed: true, appleMobile: APPLE_MOBILE, threads: state.threads, framed: IN_FRAME,
    }),
    reason,
  );
}
