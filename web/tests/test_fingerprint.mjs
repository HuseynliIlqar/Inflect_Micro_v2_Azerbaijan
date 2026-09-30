// The device check's audio comparison: is this backend computing the same thing?
import { correlation, fingerprint, matchReference } from "../js/engine/fingerprint.js";
import { REFERENCE } from "../js/engine/calibration-reference.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

// A waveform whose 50 ms loudness follows `levels`: a square wave at each level.
function shaped(levels, window = REFERENCE.window) {
  const out = new Float32Array(levels.length * window);
  levels.forEach((level, block) => {
    for (let i = 0; i < window; i += 1) out[block * window + i] = i % 2 ? level : -level;
  });
  return out;
}

{
  const print = fingerprint(shaped([0.5, 0.25]), REFERENCE.window);
  expect("samples", print.samples, 2 * REFERENCE.window);
  expect("envelope is the RMS per window", print.envelope, [0.5, 0.25]);
  expect("finite", print.finite, true);
  expect("rms of the whole", Number(print.rms.toFixed(6)), Number(Math.sqrt((0.25 + 0.0625) / 2).toFixed(6)));
}

{
  const wave = shaped([0.5]);
  wave[3] = NaN;
  expect("a NaN is caught", fingerprint(wave, REFERENCE.window).finite, false);
  expect("a partial window is left out", fingerprint(new Float32Array(REFERENCE.window + 10), REFERENCE.window).envelope.length, 1);
}

expect("identical series correlate 1", correlation([1, 2, 3], [1, 2, 3]), 1);
expect("opposite series correlate -1", correlation([1, 2, 3], [3, 2, 1]), -1);
expect("a flat series has no shape to match", correlation([1, 1, 1], [1, 2, 3]), 0);
expect("compared over the shorter length", correlation([1, 2, 3, 99], [1, 2, 3]), 1);

// The reference matched against itself, and against every way a backend can be wrong.
const ref = REFERENCE.fp32;
const good = { samples: ref.samples, rms: ref.rms, envelope: ref.envelope, finite: true };
expect("the reference matches itself", matchReference(good, ref), { ok: true, reason: null });
expect("one frame longer still matches", matchReference({ ...good, samples: ref.samples + 256 }, ref).ok, true);
expect(
  "0.09 s longer is a wrong duration (the Redmi case)",
  matchReference({ ...good, samples: ref.samples + 2160 }, ref),
  { ok: false, reason: "length" },
);
expect("NaN in the output", matchReference({ ...good, finite: false }, ref), { ok: false, reason: "not-finite" });
expect(
  "silence",
  matchReference({ ...good, rms: 0, envelope: good.envelope.map(() => 0) }, ref),
  { ok: false, reason: "silent" },
);
expect("20% louder", matchReference({ ...good, rms: ref.rms * 1.2 }, ref), { ok: false, reason: "level" });
expect(
  "the right loudness in the wrong places",
  matchReference({ ...good, envelope: [...good.envelope].reverse() }, ref),
  { ok: false, reason: "shape" },
);
expect("the int8 reference matches itself", matchReference({ ...REFERENCE.int8, finite: true }, REFERENCE.int8).ok, true);

// The reference file itself.
expect("reference has both precisions", [Boolean(REFERENCE.fp32), Boolean(REFERENCE.int8)], [true, true]);
expect("envelope covers the whole clip", ref.envelope.length, Math.floor(ref.samples / REFERENCE.window));
expect("reference has a version", /^[0-9a-f]{16}$/u.test(REFERENCE.version), true);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  fingerprint: ${checked} checks on the device check's audio comparison`);
