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
import { BLEEP, bleepSegments, censorAz } from "./az-censor.js";
import { fallbackPlan, sameStep, stepsAfter } from "./backend-plan.js";

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

// The tone that stands in for an obscenity -- the same as aztts/engine.py.
const BLEEP_HZ = 1000;
const BLEEP_SECONDS = 0.35;
const BLEEP_LEVEL = 0.25;

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
 * The visitor pressed Cancel. Thrown between chunks, never inside
 * `runWithFallback`, where it would count as the backend failing.
 */
export class CancelledError extends Error {
  constructor() {
    super("cancelled");
    this.name = "CancelledError";
  }
}

function throwIfCancelled(isCancelled) {
  if (isCancelled?.()) throw new CancelledError();
}

/** The tone played in place of a censored word. */
export function bleep(sampleRate = SAMPLE_RATE) {
  const tone = new Float32Array(Math.round(sampleRate * BLEEP_SECONDS));
  for (let index = 0; index < tone.length; index += 1) {
    tone[index] = BLEEP_LEVEL * Math.sin((2 * Math.PI * BLEEP_HZ * index) / sampleRate);
  }
  return edgeFade(tone, sampleRate);
}

function concatenate(pieces) {
  const joined = new Float32Array(pieces.reduce((sum, piece) => sum + piece.length, 0));
  let offset = 0;
  for (const piece of pieces) {
    joined.set(piece, offset);
    offset += piece.length;
  }
  return joined;
}

/**
 * Loads the two graphs once and keeps them resident.
 *
 * `onnxruntime-web` is passed in rather than imported, so this module stays
 * testable in Node with `onnxruntime-node`.
 */
/**
 * True when WebGPU can actually be used. `navigator.gpu` alone is not enough:
 * many phones expose the object and then return no adapter.
 */
export async function hasWebGpu() {
  try {
    const gpu = globalThis.navigator?.gpu;
    return Boolean(gpu && (await gpu.requestAdapter()));
  } catch {
    return false;
  }
}

const messageOf = (error) => String(error?.message ?? error);
const stepName = ({ backend, precision }) => `${backend}/${precision}`;

export class Engine {
  /**
   * `fetchModels(urls, onProgress)` is optional: given, the graphs are
   * downloaded as bytes (with progress, and cached); omitted, onnxruntime
   * fetches the paths itself, which is what a Node test wants.
   *
   * `int8DecodePath` is the quantised decoder for the CPU; without it the
   * chain skips that step. `fullQuality` skips it too, on request.
   */
  constructor(ort, {
    durationPath, decodePath, int8DecodePath = null, phonemize, fetchModels,
    preferWasm = false, fullQuality = false, censor = true,
  }) {
    this.ort = ort;
    this.durationPath = durationPath;
    this.decodePath = decodePath;
    this.int8DecodePath = int8DecodePath;
    this.phonemize = phonemize;
    this.fetchModels = fetchModels;
    this.preferWasm = preferWasm;
    this.fullQuality = fullQuality;
    // On unless a caller builds the engine without it; `speak()` takes no
    // option that could lift it.
    this.censor = censor;
    this.duration = null;
    this.decode = null;
    this.backend = null;
    this.precision = null;
    this.plan = [];
    // Every step that failed, in order: { backend, precision, stage, message }.
    this.failures = [];
  }

  /** What runs now and what failed on the way, for the page and for bug reports. */
  describe() {
    return { backend: this.backend, precision: this.precision, failures: this.failures };
  }

  /**
   * Walk the fallback chain (js/backend-plan.js) until a step loads.
   * `onProgress(stage, detail)`: "download" with {loaded, total}, then "compile".
   */
  async load(onProgress) {
    if (this.duration && this.decode) return;
    const plan = fallbackPlan({
      gpu: !this.preferWasm && (await hasWebGpu()),
      preferWasm: this.preferWasm,
      hasInt8: Boolean(this.int8DecodePath),
      fullQuality: this.fullQuality,
    });
    // A step that already failed in this page is not tried again -- except
    // the last, which is all that is left.
    this.plan = plan.filter((step, index) => index === plan.length - 1 || !this.hasFailed(step));
    await this.loadFirstWorking(this.plan, onProgress);
  }

  hasFailed(step) {
    return this.failures.some((failure) => sameStep(failure, step));
  }

  async loadFirstWorking(steps, onProgress) {
    const attempt = [];
    for (const step of steps) {
      try {
        await this.createSessions(step, onProgress);
        return;
      } catch (error) {
        // Many phone drivers expose a WebGPU adapter and then fail here.
        console.warn(`[aztts] ${stepName(step)} failed to load, trying the next step`, error);
        const failure = { ...step, stage: "load", message: messageOf(error) };
        attempt.push(failure);
        this.failures = [...this.failures, failure];
      }
    }
    throw new Error(
      `no backend could load the model (${attempt.map((f) => `${stepName(f)}: ${f.message}`).join("; ")})`,
    );
  }

  /**
   * The "full quality" choice. Reloads only when it changes what would run:
   * leaving int8, or returning to it from a CPU fp32 that int8 could replace.
   * WebGPU is already full quality, and a voice without an int8 decoder has
   * nothing to switch.
   */
  async setFullQuality(value) {
    if (value === this.fullQuality) return;
    this.fullQuality = value;
    if (!this.int8DecodePath || !this.decode) return;
    const int8Failed = this.hasFailed({ backend: "wasm", precision: "int8" });
    const reload = value
      ? this.precision === "int8"
      : this.backend === "wasm" && this.precision === "fp32" && !int8Failed;
    if (reload) await this.release();
  }

  /**
   * Both sessions on exactly one backend and precision, so `this.backend`
   * names what really runs -- onnxruntime's own fallback would switch
   * silently. Committed only when both exist, so a failure halfway leaves
   * nothing behind.
   */
  async createSessions({ backend, precision }, onProgress) {
    const decodePath = precision === "int8" ? this.int8DecodePath : this.decodePath;
    const paths = [this.durationPath, decodePath];
    const [durationModel, decodeModel] = this.fetchModels
      ? await this.fetchModels(paths, (loaded, total) =>
          onProgress?.("download", { loaded, total }),
        )
      : paths;

    onProgress?.("compile");
    const options = {
      executionProviders: [backend],
      graphOptimizationLevel: "all",
      // Errors only: with a single provider, onnxruntime warns on every load
      // that it moved shape ops to the CPU -- expected, and it prints as an error.
      logSeverityLevel: 3,
    };
    const duration = await this.ort.InferenceSession.create(durationModel, options);
    let decode;
    try {
      decode = await this.ort.InferenceSession.create(decodeModel, options);
    } catch (error) {
      await duration.release?.();
      throw error;
    }
    await this.release();
    this.duration = duration;
    this.decode = decode;
    this.backend = backend;
    this.precision = precision;
  }

  async release() {
    await Promise.all([this.duration?.release?.(), this.decode?.release?.()]);
    this.duration = null;
    this.decode = null;
    this.backend = null;
    this.precision = null;
  }

  /**
   * Run `run`; if the current step fails twice at run time -- a GPU can load
   * the graphs and still fail to execute them -- move down the chain and
   * retry. One failure is retried on the same step first, so a hiccup does not
   * cost a slower backend for the rest of the page's life.
   */
  async runWithFallback(run, onBackend) {
    let retried = false;
    for (;;) {
      try {
        return await run();
      } catch (error) {
        if (!retried) {
          retried = true;
          continue;
        }
        retried = false;
        const current = { backend: this.backend, precision: this.precision };
        const rest = stepsAfter(this.plan, current);
        this.failures = [...this.failures, { ...current, stage: "run", message: messageOf(error) }];
        if (!rest.length || rest === this.plan) throw error;
        console.warn(`[aztts] ${stepName(current)} failed while running, trying the next step`, error);
        try {
          await this.loadFirstWorking(rest);
        } catch {
          // Nothing below loads: drop the broken sessions so the next request
          // starts over, and report the error that started this.
          await this.release();
          throw error;
        }
        onBackend?.(this.describe());
      }
    }
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

  /** One chunk -> waveform, with a bleep wherever the censor left one. */
  async speakChunk(chunk, settings, onBackend, isCancelled) {
    const segments = chunk.includes(BLEEP) ? bleepSegments(chunk) : [chunk];
    const pieces = [];
    for (const segment of segments) {
      if (segment === BLEEP) {
        pieces.push(bleep());
        continue;
      }
      throwIfCancelled(isCancelled);
      const phonemes = await this.phonemize(segment);
      pieces.push(await this.runWithFallback(
        () => this.speakPhonemes(phonemes, settings),
        onBackend,
      ));
    }
    return pieces.length === 1 ? pieces[0] : concatenate(pieces);
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
      onBackend,
      // Polled between chunks; true stops the synthesis with a CancelledError.
      isCancelled,
    } = options;

    const source = this.censor ? censorAz(text) : text;
    const prepared = normalize ? normalizeAz(source) : source.split(/\s+/u).filter(Boolean).join(" ");
    if (!prepared) throw new Error("empty");
    const chunks = maxWords > 0 ? chunkText(prepared, maxWords) : [prepared];
    if (!chunks.length) throw new Error("empty");
    // A failure of the step now running is stale once a new request starts:
    // the steps before it, which explain why it runs, stay on record.
    this.failures = this.failures.filter(
      (failure) => !sameStep(failure, { backend: this.backend, precision: this.precision }),
    );
    if (!this.decode) await this.load();
    throwIfCancelled(isCancelled);

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
      const settings = { speed, variation, seed: seed + index };
      const piece = await this.speakChunk(chunks[index], settings, onBackend, isCancelled);
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
