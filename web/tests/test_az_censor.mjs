// The JavaScript censor must bleep exactly what the Python one does.
import { readFileSync } from "node:fs";
import { BLEEP, bleepSegments, censorAz } from "../js/text/az-censor.js";
import { Engine } from "../js/engine/engine.js";
import { SAMPLE_RATE, bleep } from "../js/engine/audio.js";

const golden = JSON.parse(
  readFileSync(new URL("./golden/censor.json", import.meta.url), "utf-8"),
);

const failures = [];
let checked = 0;
for (const [input, [censored, segments]] of Object.entries(golden)) {
  const got = censorAz(input);
  const pieces = bleepSegments(got);
  if (got !== censored || JSON.stringify(pieces) !== JSON.stringify(segments)) {
    failures.push({ input, got: `${got} ${JSON.stringify(pieces)}`, expected: `${censored} ${JSON.stringify(segments)}` });
  }
  checked += 1;
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures.slice(0, 12)) {
    console.log(`  in : ${f.input}`);
    console.log(`  got: ${f.got}`);
    console.log(`  want: ${f.expected}`);
    console.log("");
  }
  process.exit(1);
}
console.log(`ok  az-censor: ${checked} cases match the Python censor`);

// -- the engine: the bleep, and that a request cannot lift the censor --------


function check(condition, message) {
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

const tone = bleep();
check(tone.length === Math.round(SAMPLE_RATE * 0.35), "bleep length");
check(Math.max(...tone.map(Math.abs)) <= 0.25, "bleep level");
check(tone[0] === 0 && tone[tone.length - 1] === 0, "bleep fades in and out");

function standIn(censor) {
  const spoken = [];
  const engine = new Engine({}, { phonemize: async (text) => text, censor });
  engine.decode = {}; // loaded
  engine.runWithFallback = (run) => run();
  engine.speakPhonemes = async (phonemes) => {
    spoken.push(phonemes);
    return new Float32Array(10);
  };
  return { engine, spoken };
}

const censored = standIn(true);
// `censor: false` is not an option `speak()` reads; passing it changes nothing.
const { chunks, waveform } = await censored.engine.speak("Sən qəhbəsən, bildin?", { censor: false });
check(chunks[0] === `Sən ${BLEEP}, bildin?`, `censored chunk: ${chunks[0]}`);
check(JSON.stringify(censored.spoken) === JSON.stringify(["Sən", "bildin?"]), `spoken: ${censored.spoken}`);
check(waveform.length === 20 + tone.length, "two pieces and a bleep");

const uncensored = standIn(false);
await uncensored.engine.speak("Sən qəhbəsən.", { normalize: false });
check(uncensored.spoken[0] === "Sən qəhbəsən.", "an engine built without the censor speaks as written");

console.log("ok  az-censor engine: the bleep, and speak() cannot turn censoring off");

// Fail closed: an engine built without saying otherwise censors -- the English
// voice's engine in worker.js relies on this.
const byDefault = new Engine({}, { phonemize: async (text) => text });
check(byDefault.censor === true, "the engine censors by default");
console.log("ok  az-censor engine: censoring is the default");
