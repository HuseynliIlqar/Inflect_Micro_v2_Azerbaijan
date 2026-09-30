/**
 * The device check, start to finish, as the worker runs it: probe the GPU,
 * measure every usable backend, decide, and leave the Azerbaijani engine
 * loaded on the winner so the first Speak does not wait again.
 */

import { probeGpu } from "./gpu-probe.js";
import { calibrate } from "./calibrate.js";
import { decide } from "./calibration-verdict.js";
import { REFERENCE } from "./calibration-reference.js";

/**
 * @param {import("./engine.js").Engine} engine  the Azerbaijani voice's engine
 * @param {{ threads: number, framed: boolean, preferWasm: boolean,
 *           forceGpu: boolean, fullQuality: boolean,
 *           onProgress: (stage: string, detail?: object) => void }} context
 * @returns {Promise<object>} the verdict from js/engine/calibration-verdict.js
 */
export async function checkDevice(engine, { threads, framed, preferWasm, forceGpu, fullQuality, onProgress }) {
  onProgress("probe");
  const probe = await probeGpu({ preferWasm });
  onProgress("probed", { gpu: probe.status });
  const trials = await calibrate(engine, { reference: REFERENCE, probe, fullQuality, onProgress });
  const verdict = decide({
    probe, trials, reference: REFERENCE, threads, framed, forceGpu, measuredAt: Date.now(),
  });
  onProgress("finalise");
  await engine.setPlan(verdict.plan);
  await engine.load((stage, detail) => onProgress(stage, detail));
  return verdict;
}
