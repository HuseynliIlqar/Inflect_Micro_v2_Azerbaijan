// The waveform drawn under the player: bar heights from the synthesised audio.
import { barCount, peaks, seekFraction } from "../js/waveform.js";

let checks = 0;
function check(condition, message) {
  checks += 1;
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

// One bar per bucket, each the loudest sample in it, scaled to the loudest bar.
{
  const samples = Float32Array.from([0.1, -0.2, 0.05, 0.4, -0.8, 0.2, 0, 0]);
  const bars = peaks(samples, 4);
  check(bars.length === 4, "four bars");
  check(Math.abs(bars[0] - 0.25) < 1e-6, `first bar is 0.2/0.8, got ${bars[0]}`);
  check(bars[2] === 1, "the loudest bucket fills the height");
  check(bars[3] === 0, "silence stays flat");
}

// More bars than samples: every sample still lands in a bar, none is NaN.
{
  const bars = peaks(Float32Array.from([0.5, -0.5]), 6);
  check(bars.length === 6 && bars.every(Number.isFinite), "no empty bucket yields NaN");
}

// Silence and an empty waveform draw a flat line rather than dividing by zero.
{
  check(peaks(new Float32Array(100), 10).every((bar) => bar === 0), "silence is flat");
  check(peaks(new Float32Array(0), 10).length === 10, "empty audio still has bars");
}

// The bar count follows the canvas width, within sane bounds.
{
  check(barCount(600) === 150, `600px -> 150 bars, got ${barCount(600)}`);
  check(barCount(40) === 24, "a tiny canvas keeps a minimum");
  check(barCount(10000) === 400, "a huge canvas is capped");
}

// A click maps to a playback fraction, clamped to the clip.
{
  const box = { left: 100, width: 400 };
  check(seekFraction(300, box) === 0.5, "the middle is half way");
  check(seekFraction(50, box) === 0, "left of the canvas clamps to the start");
  check(seekFraction(900, box) === 1, "right of the canvas clamps to the end");
  check(seekFraction(300, { left: 0, width: 0 }) === 0, "a zero-width canvas does not divide by zero");
}

console.log(`ok  waveform: ${checks} checks on the bars and seeking`);
