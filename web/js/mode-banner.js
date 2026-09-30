/**
 * What the page says about the fallback chain: the mode banner above the
 * player, the badge beside the statistics, and the one-line diagnostics a
 * tester pastes into a report.
 *
 * Pure: the page renders these, the tests check them without a DOM.
 */

const hasFailure = (fallbacks, backend, precision) =>
  fallbacks.some((f) => f.backend === backend && (!precision || f.precision === precision));

/**
 * Which banner to show.
 *   kind: "none" | "fast" (int8 on the CPU) | "full" (fp32 on the CPU, chosen)
 *         | "int8-failed" (fp32 on the CPU because int8 did not work)
 *         | "gpu-failed" (no int8 for this voice, but WebGPU failed)
 *   action: the button to offer -- "full", "fast" or null
 *   gpuFailure: WebGPU's error message, when it failed on this page
 */
export function bannerState({ voice, backend, precision, fallbacks = [], fullQuality = false }) {
  const gpuFailed = fallbacks.find((f) => f.backend === "webgpu");
  const gpuFailure = gpuFailed ? gpuFailed.message : null;
  const none = { kind: "none", action: null, gpuFailure: null };
  if (backend !== "wasm") return none;

  if (precision === "int8") return { kind: "fast", action: "full", gpuFailure };
  if (voice === "az" && hasFailure(fallbacks, "wasm", "int8")) {
    return { kind: "int8-failed", action: null, gpuFailure };
  }
  if (voice === "az" && fullQuality) return { kind: "full", action: "fast", gpuFailure };
  return gpuFailure ? { kind: "gpu-failed", action: null, gpuFailure } : none;
}

/** The badge's parts: ["GPU", "fp32"] or ["CPU", "int8", { threads: 4 }]. */
export function badgeParts({ backend, precision, threads = 1 }) {
  if (!backend || !precision) return [];
  const parts = [backend === "webgpu" ? "GPU" : "CPU", precision];
  return backend === "wasm" && threads > 1 ? [...parts, { threads }] : parts;
}

const oneLine = (text) => String(text ?? "").replace(/\s+/gu, " ").trim();

/** Everything a bug report needs, on one line with no line breaks. */
export function diagnosticsLine({
  voice, backend, precision, threads = 1, isolated = false, framed = false, fullQuality = false,
  fallbacks = [], seconds, elapsed, chunks, userAgent = "",
}) {
  const failed = fallbacks.length
    ? fallbacks.map((f) => `${f.backend}/${f.precision}:${f.stage}(${oneLine(f.message)})`).join(", ")
    : "none";
  const speed =
    seconds > 0 && elapsed > 0
      ? `${seconds.toFixed(2)} s in ${elapsed.toFixed(2)} s (${(seconds / elapsed).toFixed(1)}x)`
      : "no audio yet";
  return [
    "aztts-web",
    `voice=${voice}`,
    `backend=${backend}/${precision}`,
    `threads=${threads}`,
    `isolated=${isolated ? "yes" : "no"}`,
    `framed=${framed ? "yes" : "no"}`,
    `full-quality=${fullQuality ? "yes" : "no"}`,
    `fallbacks=${failed}`,
    speed,
    `chunks=${chunks ?? "-"}`,
    oneLine(userAgent),
  ].join(" | ");
}
