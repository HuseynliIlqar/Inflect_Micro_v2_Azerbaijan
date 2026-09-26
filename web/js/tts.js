/**
 * Synthesis in the browser: phonemes in, waveform out.
 *
 * The model was exported as two ONNX graphs, and this drives them the same way
 * `model/inference.py` drives the PyTorch module:
 *
 *   duration(tokens, lengths, length_scale) -> m_p_exp, logs_p_exp, y_mask
 *   decode(m_p_exp, logs_p_exp, y_mask, zp_noise, noise_scale) -> waveform
 *
 * The export was verified against PyTorch at a waveform correlation of
 * 0.9999999999916, so the difference is inaudible. One thing genuinely differs:
 * `zp_noise` is drawn here rather than by torch, so the same seed gives the
 * same audio within this page but not the same audio as the command line.
 */

import { SYMBOLS } from "./symbols.js";
import { chunkText, DEFAULT_MAX_WORDS } from "./az-chunk.js";
import { normalizeAz } from "./az-text.js";

export const SAMPLE_RATE = 24000;

const SYMBOL_TO_ID = new Map(SYMBOLS.map((symbol, index) => [symbol, index]));

// How long to rest after a chunk, taken from the mark it ends with.
const PAUSES = {
  "?": 0.28, "！": 0.24, "!": 0.24, "？": 0.28,
  ".": 0.22, "。": 0.22, ";": 0.16, "；": 0.16,
  ":": 0.13, "：": 0.13, ",": 0.09, "，": 0.09,
};
const DEFAULT_PAUSE = 0.08;
const EDGE_FADE_MS = 5;

/** A deterministic normal generator, so a seed reproduces a reading. */
function makeNoise(seed) {
  // mulberry32, then Box-Muller. Not torch's generator -- see the note above.
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    const u = Math.max(uniform(), Number.MIN_VALUE);
    const v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** Phoneme string -> token ids, with the blank symbol interspersed. */
export function tokenise(phonemes) {
  const ids = [];
  for (const character of phonemes) {
    const id = SYMBOL_TO_ID.get(character);
    if (id !== undefined) ids.push(id);
  }
  // add_blank: a 0 before, between and after every token.
  const spread = new Array(ids.length * 2 + 1).fill(0);
  for (let index = 0; index < ids.length; index += 1) {
    spread[index * 2 + 1] = ids[index];
  }
  return spread;
}

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

/**
 * Loads the two graphs once and keeps them resident.
 *
 * `onnxruntime-web` is passed in rather than imported, so this module stays
 * testable in Node with `onnxruntime-node`.
 */
export class Engine {
  constructor(ort, { durationPath, decodePath, phonemize }) {
    this.ort = ort;
    this.durationPath = durationPath;
    this.decodePath = decodePath;
    this.phonemize = phonemize;
    this.duration = null;
    this.decode = null;
  }

  async load(onProgress) {
    if (this.duration && this.decode) return;
    // WebGPU where the browser has it: the decoder is a convolutional vocoder
    // and runs several times faster there than in single-threaded wasm, which
    // is all a page without cross-origin isolation can use. Falls back quietly.
    const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
    const options = { executionProviders: providers, graphOptimizationLevel: "all" };
    onProgress?.("duration");
    this.duration = await this.ort.InferenceSession.create(this.durationPath, options);
    onProgress?.("decode");
    this.decode = await this.ort.InferenceSession.create(this.decodePath, options);
    this.backend = providers[0];
    onProgress?.("ready");
  }

  /**
   * One chunk of phonemes -> a float32 waveform.
   *
   * `noise` overrides the drawn `zp_noise`; the parity test passes zeros so
   * the browser's output can be compared with onnxruntime's in Python.
   */
  async speakPhonemes(phonemes, { speed, variation, seed, noise: given }) {
    const { Tensor } = this.ort;
    const tokens = tokenise(phonemes);
    if (!tokens.length) return new Float32Array(0);

    const durationOutputs = await this.duration.run({
      tokens: new Tensor("int64", BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]),
      lengths: new Tensor("int64", BigInt64Array.from([BigInt(tokens.length)]), [1]),
      length_scale: new Tensor("float32", Float32Array.from([1 / speed]), []),
    });

    const mean = durationOutputs.m_p_exp;
    const logs = durationOutputs.logs_p_exp;
    const mask = durationOutputs.y_mask;

    let noise = given;
    if (!noise) {
      const normal = makeNoise(seed);
      noise = new Float32Array(mean.data.length);
      for (let index = 0; index < noise.length; index += 1) noise[index] = normal();
    }

    const decodeOutputs = await this.decode.run({
      m_p_exp: mean,
      logs_p_exp: logs,
      y_mask: mask,
      zp_noise: new Tensor("float32", noise, mean.dims),
      noise_scale: new Tensor("float32", Float32Array.from([variation]), []),
    });
    return edgeFade(decodeOutputs.waveform.data);
  }

  /**
   * Text -> waveform, through the same steps as the command line: normalise,
   * chunk, phonemise, synthesise, and rest between chunks.
   */
  async speak(text, options = {}) {
    const {
      speed = 1.0,
      variation = 0.667,
      seed = 7,
      normalize = true,
      maxWords = DEFAULT_MAX_WORDS,
      onChunk,
    } = options;

    const prepared = normalize ? normalizeAz(text) : text.split(/\s+/u).filter(Boolean).join(" ");
    if (!prepared) throw new Error("empty");
    const chunks = maxWords > 0 ? chunkText(prepared, maxWords) : [prepared];
    if (!chunks.length) throw new Error("empty");

    const pieces = [];
    let total = 0;
    for (let index = 0; index < chunks.length; index += 1) {
      if (index) {
        const silence = new Float32Array(
          Math.round(SAMPLE_RATE * pauseAfter(chunks[index - 1])),
        );
        pieces.push(silence);
        total += silence.length;
      }
      onChunk?.(index, chunks.length);
      const phonemes = await this.phonemize(chunks[index]);
      const piece = await this.speakPhonemes(phonemes, {
        speed,
        variation,
        seed: seed + index,
      });
      pieces.push(piece);
      total += piece.length;
    }

    const waveform = new Float32Array(total);
    let offset = 0;
    for (const piece of pieces) {
      waveform.set(piece, offset);
      offset += piece.length;
    }
    for (let index = 0; index < waveform.length; index += 1) {
      waveform[index] = Math.max(-1, Math.min(1, waveform[index]));
    }
    return { waveform, chunks, sampleRate: SAMPLE_RATE };
  }
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
