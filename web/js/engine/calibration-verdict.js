/**
 * The device check's verdict: from what each backend did with the check's
 * sentence, which one this device should use, how fast it will be, and why.
 *
 * A backend is only used when its audio matched the reference
 * (js/engine/fingerprint.js) on every run. Among those, the fastest wins --
 * except that WebGPU must be clearly faster than the CPU to be preferred: a
 * phone's GPU driver is the part most likely to fail later, and a CPU that is
 * nearly as fast is the safer choice.
 *
 * Pure, so every verdict is tested without a browser.
 */

import { matchReference } from "./fingerprint.js";

// WebGPU is chosen only when at least this much faster than the best CPU step.
const GPU_MARGIN = 1.2;
// Seconds of speech in a typical sentence of the playground's chunk size.
export const TYPICAL_SPEECH_SECONDS = 4;
// Seconds of computing per second of speech, at the edge of each tier.
const FAST = 0.5;
const USABLE = 1.5;

const CPU_FP32 = Object.freeze({ backend: "wasm", precision: "fp32" });
const stepOf = ({ backend, precision }) => ({ backend, precision });
const same = (a, b) => a?.backend === b?.backend && a?.precision === b?.precision;

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

/**
 * One backend's trial -> passed or not, why, and its speed as seconds of
 * computing per second of speech (`rtf`, lower is faster).
 *
 * @param {{ step: {backend: string, precision: string},
 *           error?: { stage: "load"|"run"|"timeout", message: string },
 *           runs: { ms: number, print: object }[] }} trial
 */
export function judgeTrial({ step, error, runs = [] }, reference) {
  const expected = reference[step.precision];
  const audioMs = (expected.samples / reference.sampleRate) * 1000;
  const rtf = runs.length ? Number((median(runs.map((run) => run.ms)) / audioMs).toFixed(3)) : null;
  const judged = (ok, reason, message = "") => ({ ...stepOf(step), ok, reason, rtf, message });
  if (error) return judged(false, error.stage, error.message);
  if (!runs.length) return judged(false, "run");
  const lengths = new Set(runs.map((run) => run.print.samples));
  const verdicts = runs.map((run) => matchReference(run.print, expected));
  // One length every time, or the durations are not being computed reliably.
  if (lengths.size > 1 && verdicts.every((v) => v.reason !== "not-finite")) return judged(false, "unstable");
  const failed = verdicts.find((v) => !v.ok);
  return failed ? judged(false, failed.reason) : judged(true, null);
}

/** "fast" | "usable" | "slow", or "failed" when nothing could be measured. */
export function speedTier(rtf) {
  if (rtf === null || rtf === undefined) return "failed";
  if (rtf <= FAST) return "fast";
  if (rtf <= USABLE) return "usable";
  return "slow";
}

function gpuStatus(probe, gpuTrial, chosen) {
  if (probe.status !== "ok") return probe.status;
  if (!gpuTrial) return "off";
  if (gpuTrial.ok) return same(chosen, gpuTrial) ? "chosen" : "slower";
  if (["load", "run", "timeout"].includes(gpuTrial.reason)) return "error";
  return gpuTrial.reason === "unstable" ? "unstable" : "wrong-output";
}

function pick(passed, forceGpu) {
  const gpu = passed.find((trial) => trial.backend === "webgpu");
  const cpu = passed.filter((trial) => trial.backend === "wasm").sort((a, b) => a.rtf - b.rtf)[0];
  if (gpu && (forceGpu || !cpu || gpu.rtf * GPU_MARGIN <= cpu.rtf)) return gpu;
  return cpu ?? null;
}

/**
 * @param {{ probe: { status: string, message?: string }, trials: object[],
 *           reference: object, threads: number, framed: boolean,
 *           measuredAt: number, forceGpu?: boolean }} facts
 *   probe.status: "ok" | "unsupported" | "no-adapter" | "software" |
 *   "device-failed" | "off" -- see js/engine/gpu-probe.js.
 */
export function decide({ probe, trials, reference, threads, framed, measuredAt, forceGpu = false }) {
  const judged = trials.map((trial) => judgeTrial(trial, reference));
  const passed = judged.filter((trial) => trial.ok);
  const winner = pick(passed, forceGpu);
  const chosen = winner ? stepOf(winner) : null;
  const rest = passed.filter((trial) => !same(trial, chosen)).sort((a, b) => a.rtf - b.rtf);
  const plan = [...(chosen ? [chosen] : []), ...rest.map(stepOf)];
  if (!plan.some((step) => same(step, CPU_FP32))) plan.push(CPU_FP32);

  const gpuTrial = judged.find((trial) => trial.backend === "webgpu");
  const int8Trial = judged.find((trial) => trial.precision === "int8");
  const rtf = winner?.rtf ?? null;
  return {
    version: reference.version,
    measuredAt,
    chosen,
    rtf,
    sentenceSeconds: rtf === null ? null : Number((rtf * TYPICAL_SPEECH_SECONDS).toFixed(1)),
    tier: speedTier(rtf),
    gpu: gpuStatus(probe, gpuTrial, chosen),
    gpuMessage: gpuTrial?.message || probe.message || "",
    int8Failed: Boolean(int8Trial && !int8Trial.ok),
    threads,
    framed,
    plan,
    trials: judged,
  };
}
