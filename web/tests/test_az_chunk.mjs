// The JavaScript chunker must cut in exactly the same places as the Python one.
import { readFileSync } from "node:fs";
import { normalizeAz } from "../js/text/az-text.js";
import { chunkText } from "../js/text/az-chunk.js";

const golden = JSON.parse(
  readFileSync(new URL("./golden/chunks.json", import.meta.url), "utf-8"),
);

const failures = [];
let checked = 0;
for (const [input, expected] of Object.entries(golden)) {
  const got = chunkText(normalizeAz(input));
  if (JSON.stringify(got) !== JSON.stringify(expected)) {
    failures.push({ input, got, expected });
  }
  checked += 1;
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures.slice(0, 8)) {
    console.log(`  in  : ${f.input}`);
    console.log(`  got : ${JSON.stringify(f.got)}`);
    console.log(`  want: ${JSON.stringify(f.expected)}`);
    console.log("");
  }
  process.exit(1);
}
console.log(`ok  az-chunk: ${checked} cases match the Python chunker`);
