// The JavaScript normaliser must produce exactly what the Python one does.
import { readFileSync } from "node:fs";
import { normalizeAz } from "../js/az-text.js";

const golden = JSON.parse(
  readFileSync(new URL("./golden/normalise.json", import.meta.url), "utf-8"),
);

const failures = [];
let checked = 0;
for (const [input, expected] of Object.entries(golden)) {
  const got = normalizeAz(input);
  if (got !== expected) failures.push({ input, got, expected });
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
console.log(`ok  az-text: ${checked} cases match the Python normaliser`);
