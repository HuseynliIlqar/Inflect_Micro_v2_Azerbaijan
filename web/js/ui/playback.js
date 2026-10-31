/**
 * The arithmetic behind the result's player: its clock, which chunk is being
 * heard, and the keys that move the playhead. The page wiring is
 * js/page/player-view.js.
 */

/** "1:05" for 65 seconds. */
export function formatTime(seconds) {
  const whole = Number.isFinite(seconds) ? Math.max(0, Math.floor(seconds)) : 0;
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/**
 * The chunk heard at `sample`. Spans are [start, end) in samples; the pause
 * after a chunk still counts as that chunk, so the highlight does not flicker.
 *
 * @param {Array<[number, number]>} spans
 * @param {number} sample
 * @returns {number} the chunk's index, or -1 before the first and after the last
 */
export function chunkAt(spans, sample) {
  if (!spans.length || sample < spans[0][0] || sample >= spans[spans.length - 1][1]) return -1;
  let found = -1;
  for (let index = 0; index < spans.length; index += 1) {
    if (spans[index][0] <= sample) found = index;
  }
  return found;
}

const STEP_SECONDS = 1;

/**
 * Where a key press on the waveform (a WAI-ARIA slider) moves the playhead:
 * arrows by a second, Page keys by a tenth, Home and End to the edges.
 *
 * @returns {number | null} the new time, or null when the key does not seek
 */
export function seekTime(key, current, duration) {
  if (!(duration > 0)) return null;
  const tenth = duration / 10;
  const target = {
    ArrowRight: current + STEP_SECONDS,
    ArrowUp: current + STEP_SECONDS,
    ArrowLeft: current - STEP_SECONDS,
    ArrowDown: current - STEP_SECONDS,
    PageUp: current + tenth,
    PageDown: current - tenth,
    Home: 0,
    End: duration,
  }[key];
  return target === undefined ? null : Math.min(duration, Math.max(0, target));
}
