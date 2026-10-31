// The orb's shape (js/ui/blob.js) and the loudness it follows while a clip
// plays (js/ui/envelope.js): pure arithmetic, so it is checked here, not in a browser.
import { blobPath, blobPoints, smoothPath } from "../js/ui/blob.js";
import { ENVELOPE_FPS, envelope, levelAt } from "../js/ui/envelope.js";

const failures = [];
let checked = 0;
const expect = (name, ok) => {
  checked += 1;
  if (!ok) failures.push(name);
};
const near = (a, b, tolerance = 1e-6) => Math.abs(a - b) <= tolerance;

// -- blob ---------------------------------------------------------------------
const still = blobPoints(90, 0, { count: 48 });
expect("count points", still.length === 48);
expect("no wobble is a circle", still.every(([x, y]) => near(Math.hypot(x, y), 90)));

const moving = blobPoints(90, 0.12, { phase: 1.3, offset: 0.6, level: 0.5 });
expect("default 72 points", moving.length === 72);
expect("every point finite", moving.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
expect("wobble stays within its amplitude",
  moving.every(([x, y]) => Math.hypot(x, y) <= 90 * (1 + 0.12 * 1.35) + 1e-9));
expect("same inputs, same shape",
  blobPath(90, 0.1, { phase: 2 }) === blobPath(90, 0.1, { phase: 2 }));
expect("phase moves the shape", blobPath(90, 0.1, { phase: 2 }) !== blobPath(90, 0.1, { phase: 2.5 }));

const path = smoothPath(still);
expect("path starts with a move", path.startsWith("M"));
expect("path is closed", path.endsWith("Z"));
expect("one curve per point", (path.match(/Q/gu) ?? []).length === still.length);
expect("no NaN in the path", !/NaN|Infinity/u.test(blobPath(90, 0.2, { level: 1 })));
expect("empty point list draws nothing", smoothPath([]) === "");

// -- envelope -----------------------------------------------------------------
const RATE = 24000;
const tone = (seconds, amplitude) =>
  Float32Array.from({ length: RATE * seconds }, (_, i) => amplitude * Math.sin((2 * Math.PI * 220 * i) / RATE));

expect("empty clip, empty envelope", envelope(new Float32Array(0), RATE).length === 0);
const silent = envelope(new Float32Array(RATE), RATE);
expect("one value per frame", silent.length === ENVELOPE_FPS);
expect("silence is zero", silent.every((value) => value === 0));

const steady = envelope(tone(1, 0.4), RATE);
expect("a steady tone is level", steady.every((value) => value > 0.95 && value <= 1));

const halves = new Float32Array(RATE * 2);
halves.set(tone(1, 0.8), 0);
halves.set(tone(1, 0.4), RATE);
const stepped = envelope(halves, RATE);
expect("loudest frame is 1", near(Math.max(...stepped), 1, 1e-6));
expect("half the amplitude, half the level", near(levelAt(stepped, 1.5), 0.5, 0.05));
expect("before the clip is quiet", levelAt(stepped, -1) === 0);
expect("after the clip is quiet", levelAt(stepped, 5) === 0);
expect("first frame is read at 0 s", near(levelAt(stepped, 0), stepped[0]));

if (failures.length) {
  for (const f of failures) console.error("FAIL", f);
  process.exit(1);
}
console.log(`ok  orb-motion: ${checked} checks on the orb's shape and the loudness it follows`);
