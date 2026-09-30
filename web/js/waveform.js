/**
 * The synthesised audio drawn as bars, with the part already heard coloured.
 *
 * It is a picture of the actual waveform the model produced -- pauses between
 * chunks show as gaps, a censored word as a flat block of tone. The native
 * <audio> element stays underneath as the accessible control; clicking the
 * bars is a pointer shortcut for seeking, not the only way to do it.
 */

const PX_PER_BAR = 4;
const MIN_BARS = 24;
const MAX_BARS = 400;
// A silent stretch still draws a hairline, so the clip's length is visible.
const MIN_BAR_PX = 1.5;

/** How many bars suit a canvas this many CSS pixels wide. */
export function barCount(width) {
  return Math.min(MAX_BARS, Math.max(MIN_BARS, Math.round(width / PX_PER_BAR)));
}

/** The loudest sample of each of `bars` equal slices, scaled so the peak is 1. */
export function peaks(samples, bars) {
  const heights = new Float32Array(bars);
  if (!samples.length) return heights;
  for (let index = 0; index < samples.length; index += 1) {
    const bar = Math.min(bars - 1, Math.floor((index * bars) / samples.length));
    const level = Math.abs(samples[index]);
    if (level > heights[bar]) heights[bar] = level;
  }
  const loudest = heights.reduce((max, value) => Math.max(max, value), 0);
  return loudest > 0 ? heights.map((value) => value / loudest) : heights;
}

/** Where along the clip a pointer at `clientX` points, from 0 to 1. */
export function seekFraction(clientX, { left, width }) {
  if (!(width > 0)) return 0;
  return Math.min(1, Math.max(0, (clientX - left) / width));
}

/**
 * Draw `heights` into the canvas, mirrored about the middle line; bars left of
 * `played` (0..1) take the `heard` colour.
 */
export function drawWave(canvas, heights, { played = 0, rest, heard }) {
  const ratio = globalThis.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!heights.length) return;

  const step = width / heights.length;
  const barWidth = Math.max(1, step * 0.6);
  const middle = height / 2;
  const playedBars = played * heights.length;
  for (let index = 0; index < heights.length; index += 1) {
    const barHeight = Math.max(MIN_BAR_PX, heights[index] * (height - 2));
    context.fillStyle = index < playedBars ? heard : rest;
    context.fillRect(index * step + (step - barWidth) / 2, middle - barHeight / 2, barWidth, barHeight);
  }
}
