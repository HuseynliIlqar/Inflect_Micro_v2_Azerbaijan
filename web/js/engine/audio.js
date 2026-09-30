/**
 * The audio side of synthesis: pauses between chunks, click-free joins, the
 * bleep that stands in for a censored word, and the WAV file at the end.
 */

export const SAMPLE_RATE = 24000;

// How long to rest after a chunk, taken from the mark it ends with.
const PAUSES = {
  "?": 0.28, "！": 0.24, "!": 0.24, "？": 0.28,
  ".": 0.22, "。": 0.22, ";": 0.16, "；": 0.16,
  ":": 0.13, "：": 0.13, ",": 0.09, "，": 0.09,
};
const DEFAULT_PAUSE = 0.08;
const EDGE_FADE_MS = 5;

// The tone that stands in for an obscenity -- the same as aztts/engine.py.
const BLEEP_HZ = 1000;
const BLEEP_SECONDS = 0.35;
const BLEEP_LEVEL = 0.25;

export function pauseAfter(chunk) {
  const trimmed = chunk.replace(/\s+$/u, "");
  const ending = trimmed ? trimmed[trimmed.length - 1] : "";
  return PAUSES[ending] ?? DEFAULT_PAUSE;
}

/** Ramp the first and last few milliseconds, so chunks join without a click. */
export function edgeFade(waveform, sampleRate = SAMPLE_RATE) {
  const frames = Math.min(
    Math.round((sampleRate * EDGE_FADE_MS) / 1000),
    Math.floor(waveform.length / 2),
  );
  if (frames <= 0) return waveform;
  const output = Float32Array.from(waveform);
  for (let index = 0; index < frames; index += 1) {
    const ramp = frames === 1 ? 1 : index / (frames - 1);
    output[index] *= ramp;
    output[output.length - 1 - index] *= ramp;
  }
  return output;
}

/** The tone played in place of a censored word. */
export function bleep(sampleRate = SAMPLE_RATE) {
  const tone = new Float32Array(Math.round(sampleRate * BLEEP_SECONDS));
  for (let index = 0; index < tone.length; index += 1) {
    tone[index] = BLEEP_LEVEL * Math.sin((2 * Math.PI * BLEEP_HZ * index) / sampleRate);
  }
  return edgeFade(tone, sampleRate);
}

export function concatenate(pieces) {
  const joined = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
  let offset = 0;
  for (const piece of pieces) {
    joined.set(piece, offset);
    offset += piece.length;
  }
  return joined;
}

/** A finished waveform as a 16-bit PCM WAV file. */
export function toWav(waveform, sampleRate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + waveform.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };
  text(0, "RIFF");
  view.setUint32(4, 36 + waveform.length * 2, true);
  text(8, "WAVEfmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  text(36, "data");
  view.setUint32(40, waveform.length * 2, true);
  for (let index = 0; index < waveform.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, waveform[index]));
    view.setInt16(44 + index * 2, sample * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
