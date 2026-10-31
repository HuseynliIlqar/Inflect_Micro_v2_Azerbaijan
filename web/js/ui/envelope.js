/**
 * How loud a clip is, frame by frame, so the orb can swell with the voice.
 *
 * Read from the synthesised samples, not from a Web Audio analyser: routing an
 * <audio> element through an AudioContext makes iPhones obey the silent
 * switch and go quiet, and on a phone that reads as "it does not work".
 */

export const ENVELOPE_FPS = 30;

/**
 * Root-mean-square loudness per frame, scaled so the loudest frame is 1.
 *
 * @param {Float32Array} samples
 * @param {number} sampleRate
 * @param {number} [fps]
 * @returns {Float32Array}
 */
export function envelope(samples, sampleRate, fps = ENVELOPE_FPS) {
  const frameLength = Math.max(1, Math.round(sampleRate / fps));
  const frames = new Float32Array(Math.ceil(samples.length / frameLength));
  for (let frame = 0; frame < frames.length; frame += 1) {
    const start = frame * frameLength;
    const end = Math.min(samples.length, start + frameLength);
    let sum = 0;
    for (let index = start; index < end; index += 1) sum += samples[index] * samples[index];
    frames[frame] = Math.sqrt(sum / (end - start));
  }
  const loudest = frames.reduce((max, value) => Math.max(max, value), 0);
  return loudest > 0 ? frames.map((value) => value / loudest) : frames;
}

/** The level at `seconds` into the clip; 0 outside it. */
export function levelAt(frames, seconds, fps = ENVELOPE_FPS) {
  const index = Math.floor(seconds * fps);
  return index >= 0 && index < frames.length ? frames[index] : 0;
}
