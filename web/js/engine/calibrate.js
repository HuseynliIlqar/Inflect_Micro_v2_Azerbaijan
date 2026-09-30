/**
 * The device check's measurement: the check's sentence, synthesised on every
 * backend this device can use, timed, and fingerprinted.
 *
 * The CPU goes first -- it is what most phones end up on, and it needs the
 * smaller download -- then WebGPU, if js/engine/gpu-probe.js found a working
 * adapter. Each backend gets one warm-up run (a GPU compiles its shaders on
 * the first) and `runs` timed ones. The noise is switched off, so every run
 * of a correct backend gives the same audio as the reference.
 *
 * What to make of the results is js/engine/calibration-verdict.js's job.
 */

import { fingerprint } from "./fingerprint.js";
import { judgeTrial } from "./calibration-verdict.js";

const GPU = Object.freeze({ backend: "webgpu", precision: "fp32" });
const CPU_INT8 = Object.freeze({ backend: "wasm", precision: "int8" });
const CPU_FP32 = Object.freeze({ backend: "wasm", precision: "fp32" });
// variation 0 switches the noise off: the output depends on the phonemes alone.
const SETTINGS = Object.freeze({ speed: 1, variation: 0, seed: 0 });
// One run of a 1.3 s sentence; a GPU that has not answered by then is stuck.
const RUN_TIMEOUT_MS = 120_000;
// Downloading and compiling on the GPU: a phone took 6-20 s to compile, and the
// download reports progress; a driver that hangs here says nothing at all.
const GPU_LOAD_TIMEOUT_MS = 240_000;

class TimeoutError extends Error {}

const messageOf = (error) => String(error?.message ?? error);

function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new TimeoutError(`no answer in ${ms / 1000} s`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** Load one step and time it. Never throws: a failure is part of the result. */
async function measure(engine, step, { reference, runs, timeoutMs, loadTimeoutMs, now, onProgress }) {
  const report = (stage, detail = {}) => onProgress?.(stage, { ...step, ...detail });
  try {
    const loading = engine.createSessions(step, (stage, detail) => report(stage, detail));
    await (step.backend === "webgpu" ? withTimeout(loading, loadTimeoutMs) : loading);
  } catch (error) {
    const stage = error instanceof TimeoutError ? "timeout" : "load";
    return { step, error: { stage, message: messageOf(error) }, runs: [] };
  }
  const timed = [];
  for (let run = 0; run <= runs; run += 1) {
    report("test", { run, runs });
    const began = now();
    try {
      const waveform = await withTimeout(engine.rawWaveform(reference.phonemes, SETTINGS), timeoutMs);
      if (run > 0) timed.push({ ms: now() - began, print: fingerprint(waveform, reference.window) });
    } catch (error) {
      const stage = error instanceof TimeoutError ? "timeout" : "run";
      return { step, error: { stage, message: messageOf(error) }, runs: timed };
    }
  }
  return { step, runs: timed };
}

/**
 * @param {import("./engine.js").Engine} engine  the Azerbaijani voice's engine
 * @param {{ reference: object, probe: { status: string }, runs?: number,
 *           fullQuality?: boolean, timeoutMs?: number, loadTimeoutMs?: number,
 *           now?: () => number,
 *           onProgress?: (stage: string, detail: object) => void }} options
 *   onProgress stages: "candidates" {steps}, then per step "download"
 *   {loaded, total}, "compile", "test" {run, runs} and "tested" {ok, reason, rtf}.
 * @returns {Promise<object[]>} one trial per step, for `decide`
 */
export async function calibrate(engine, {
  reference, probe, runs = 3, fullQuality = false, timeoutMs = RUN_TIMEOUT_MS,
  loadTimeoutMs = GPU_LOAD_TIMEOUT_MS, now = () => performance.now(), onProgress,
}) {
  const cpu = engine.int8DecodePath && !fullQuality ? CPU_INT8 : CPU_FP32;
  const steps = [cpu, ...(probe.status === "ok" ? [GPU] : [])];
  onProgress?.("candidates", { steps });
  const options = { reference, runs, timeoutMs, loadTimeoutMs, now, onProgress };
  const trials = [];
  for (const step of steps) {
    const { trial, ok } = await measureAndReport(engine, step, options);
    trials.push(trial);
    // A broken int8 decoder: the full one is what the CPU would fall back to.
    if (step === CPU_INT8 && !ok) trials.push((await measureAndReport(engine, CPU_FP32, options)).trial);
  }
  return trials;
}

async function measureAndReport(engine, step, options) {
  const trial = await measure(engine, step, options);
  const { backend, precision, ok, reason, rtf } = judgeTrial(trial, options.reference);
  options.onProgress?.("tested", { backend, precision, ok, reason, rtf });
  return { trial, ok };
}
