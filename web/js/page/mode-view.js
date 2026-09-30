/**
 * What the page says about how it ran: the CPU note, the fallback chain's
 * banner, the badge beside the download, the statistics and the diagnostics.
 */

import { chunkCount } from "../ui/i18n.js";
import { SAMPLE_RATE } from "../engine/audio.js";
import { bannerState, badgeParts, diagnosticsLine, bannerCopyKeys } from "../ui/mode-banner.js";
import { els } from "./dom.js";
import { DIRECT_URL, IN_FRAME, PHONE, QUERY, state, t } from "./context.js";

/** The CPU-path note: how slow to expect, and the faster link when framed. */
export function renderSlowNote() {
  // On a phone the phone note already says this, in plainer words.
  els.slowNote.hidden = state.backend !== "wasm" || PHONE;
  if (els.slowNote.hidden) return;
  if (state.threads > 1) {
    els.slowNote.textContent = t("slow_note_threads", { threads: state.threads });
    return;
  }
  els.slowNote.textContent = t("slow_note");
  if (IN_FRAME) {
    const link = document.createElement("a");
    link.href = DIRECT_URL;
    link.target = "_blank";
    link.rel = "noopener";
    link.textContent = DIRECT_URL.replace("https://", "").replace("/index.html", "");
    els.slowNote.append(" ", t("open_direct"), " ", link);
  }
}

/** The banner above the player: which step of the fallback chain is speaking. */
export function renderModeBanner() {
  const { modeVoice, backend, precision, fallbacks, fullQuality } = state;
  const banner = bannerState({ voice: modeVoice, backend, precision, fallbacks, fullQuality });
  els.modeBanner.hidden = banner.kind === "none";
  if (els.modeBanner.hidden) return;

  els.modeBanner.dataset.kind = banner.kind;
  const copy = bannerCopyKeys(banner, { phone: PHONE && !QUERY.forceGpu });
  els.modeTitle.textContent = t(copy.title);
  els.modeText.textContent = copy.text ? t(copy.text) : "";
  els.modeText.hidden = !copy.text;
  els.modeGpu.hidden = !banner.gpuFailure;
  if (banner.gpuFailure) els.modeGpu.textContent = t("mode_gpu_failed", { reason: banner.gpuFailure });
  els.modeAction.hidden = !banner.action;
  if (banner.action) {
    els.modeAction.textContent = t(`action_${banner.action}`);
    els.modeAction.dataset.action = banner.action;
  }
}

/** "CPU · int8 · 4 threads" beside the download button. */
export function renderBadge(result) {
  const parts = result ? badgeParts(result) : [];
  els.modeBadge.hidden = parts.length === 0;
  els.modeBadge.dataset.precision = result?.precision ?? "";
  els.modeBadge.textContent = parts
    .map((part) => (typeof part === "string" ? part : t("badge_threads", part)))
    .join(" · ");
}

function backendName(name, threads = 1) {
  if (!name) return "";
  if (name === "wasm" && threads > 1) return t("backend_wasm_threads", { threads });
  return t(`backend_${name}`);
}

export function showStats(result) {
  els.stats.textContent = t("stats", {
    seconds: result.seconds.toFixed(2),
    elapsed: result.elapsed.toFixed(2),
    realtime: result.realtime.toFixed(1),
    chunks: chunkCount(state.language, result.chunks),
    rate: SAMPLE_RATE,
  }) + (result.backend ? ` · ${backendName(result.backend, result.threads)}` : "");
}

function diagnostics() {
  const { lastResult } = state;
  return diagnosticsLine({
    voice: lastResult?.voice ?? state.modeVoice,
    backend: state.backend,
    precision: state.precision,
    threads: state.threads,
    isolated: self.crossOriginIsolated,
    framed: IN_FRAME,
    fullQuality: state.fullQuality,
    phone: PHONE,
    forceGpu: QUERY.forceGpu,
    fallbacks: state.fallbacks,
    seconds: lastResult?.seconds,
    elapsed: lastResult?.elapsed,
    chunks: lastResult?.chunks,
    userAgent: navigator.userAgent,
  });
}

/** Copy the diagnostics line; where the clipboard is refused, show it to select. */
export async function copyDiagnostics() {
  const line = diagnostics();
  els.diagnostics.hidden = true;
  try {
    await navigator.clipboard.writeText(line);
    els.copyStatus.textContent = t("copied");
  } catch (error) {
    console.warn("[aztts] clipboard refused", error);
    els.copyStatus.textContent = t("copy_failed");
    els.diagnostics.textContent = line;
    els.diagnostics.hidden = false;
  }
}
