/**
 * Does a backend compute what the model computes? The device check's answer.
 *
 * The check's sentence is synthesised with the noise switched off, so the
 * output is deterministic, and compared with what onnxruntime on the CPU gave
 * for the same phonemes (js/engine/calibration-reference.js). Not sample by
 * sample -- a GPU rounds differently and that is fine -- but by what a listener
 * would notice: the length, the loudness, and where the loudness falls.
 *
 * A Redmi Note 10 Pro's WebGPU made the same sentence 0.09-0.33 s longer than
 * the CPU. That is the failure this exists to catch.
 *
 * Pure, so every rule is tested without a model.
 */

// A GPU may round one phoneme's duration the other way: one or two frames.
// 0.09 s is eight frames.
const MAX_FRAME_DRIFT = 2;
const HOP = 256; // samples per frame
const MAX_LEVEL_DRIFT = 0.05; // relative RMS
const MIN_SHAPE = 0.98; // correlation of the loudness envelopes
const SILENT = 0.1; // below this fraction of the reference's RMS

/** Length, loudness, the loudness envelope, and whether every sample is a number. */
export function fingerprint(waveform, window) {
  let sum = 0;
  let finite = true;
  const envelope = [];
  let block = 0;
  for (let index = 0; index < waveform.length; index += 1) {
    const value = waveform[index];
    if (!Number.isFinite(value)) finite = false;
    const square = value * value;
    sum += square;
    block += square;
    if ((index + 1) % window === 0) {
      envelope.push(Math.sqrt(block / window));
      block = 0;
    }
  }
  const rms = waveform.length ? Math.sqrt(sum / waveform.length) : 0;
  return { samples: waveform.length, rms, envelope, finite };
}

/** Pearson correlation over the shorter series; 0 when either has no variation. */
export function correlation(a, b) {
  const length = Math.min(a.length, b.length);
  if (length < 2) return 0;
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < length; i += 1) {
    meanA += a[i];
    meanB += b[i];
  }
  meanA /= length;
  meanB /= length;
  let cov = 0;
  let varA = 0;
  let varB = 0;
  for (let i = 0; i < length; i += 1) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  if (varA === 0 || varB === 0) return 0;
  return Math.round((cov / Math.sqrt(varA * varB)) * 1e9) / 1e9;
}

/**
 * @returns {{ ok: boolean,
 *             reason: null|"not-finite"|"silent"|"length"|"level"|"shape" }}
 */
export function matchReference(print, reference) {
  const fail = (reason) => ({ ok: false, reason });
  if (!print.finite) return fail("not-finite");
  if (print.rms < reference.rms * SILENT) return fail("silent");
  if (Math.abs(print.samples - reference.samples) > MAX_FRAME_DRIFT * HOP) return fail("length");
  if (Math.abs(print.rms - reference.rms) / reference.rms > MAX_LEVEL_DRIFT) return fail("level");
  if (correlation(print.envelope, reference.envelope) < MIN_SHAPE) return fail("shape");
  return { ok: true, reason: null };
}
