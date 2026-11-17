// The JavaScript stress frontend must do exactly what model/frontend_hook.py does.
// Golden file: tools/build_web_stress.py, from the vendored hook and real eSpeak NG output.
import { readFileSync } from "node:fs";
import { finishPhonemes, frontendNormalize, loadLexicon, stressPhonemizer } from "../js/engine/stress.js";
import { stressRule } from "../js/engine/stress-rules.js";

const golden = JSON.parse(readFileSync(new URL("./golden/stress.json", import.meta.url), "utf-8"));
const failures = [];
const fail = (what, input, got, want) => failures.push({ what, input, got, want });

const lexicon = await loadLexicon();
for (const name of ["verbs", "lemmas", "forms"]) {
  if (lexicon[name].size !== golden.lexicon[name]) fail("lexicon size", name, lexicon[name].size, golden.lexicon[name]);
}

const settings = [{}, { question: true }, { sentenceFinal: true }];
for (const [word, expected] of Object.entries(golden.words)) {
  settings.forEach((options, index) => {
    const got = stressRule(word, lexicon, options);
    if (JSON.stringify(got) !== JSON.stringify(expected[index])) {
      fail(`stressRule ${JSON.stringify(options)}`, word, got, expected[index]);
    }
  });
}

let changed = 0;
for (const { text, raw, out } of golden.cases) {
  const got = finishPhonemes(frontendNormalize(text), raw, lexicon);
  if (got !== out) fail("phonemes", text, got, out);
  if (out !== raw) changed += 1;
}

// The wrapper hands eSpeak the normalised text and waits for both halves.
const asked = [];
const wrapped = stressPhonemizer(async (text) => {
  asked.push(text);
  return golden.cases[0].raw;
});
const first = golden.cases[0];
if ((await wrapped(`  ${first.text}\n`)) !== first.out) fail("stressPhonemizer", first.text, "", first.out);
if (asked[0] !== frontendNormalize(first.text)) fail("stressPhonemizer input", first.text, asked[0], "normalised");
if ((await wrapped(" \t ")) !== "") fail("empty text", "blank", "non-empty", "");

if (failures.length) {
  console.log(`FAIL ${failures.length}`);
  for (const f of failures.slice(0, 12)) {
    console.log(`  ${f.what}: ${f.input}`);
    console.log(`    got : ${JSON.stringify(f.got)}`);
    console.log(`    want: ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
const words = Object.keys(golden.words).length;
console.log(`ok  stress: ${golden.cases.length} chunks (${changed} restressed) and ${words} words x3 match the Python hook`);
