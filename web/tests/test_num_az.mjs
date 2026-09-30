// Every number in the golden file must come out exactly as num2words wrote it.
import { readFileSync } from "node:fs";
import { numberToWords, ordinalToWords } from "../js/text/num-az.js";

const golden = JSON.parse(
  readFileSync(new URL("./golden/numbers.json", import.meta.url), "utf-8"),
);

let checked = 0;
const failures = [];
for (const [raw, [cardinal, ordinal]] of Object.entries(golden)) {
  const value = BigInt(raw);
  const gotCardinal = numberToWords(value);
  const gotOrdinal = ordinalToWords(value);
  if (gotCardinal !== cardinal) {
    failures.push(`${raw} cardinal: got "${gotCardinal}", want "${cardinal}"`);
  }
  if (gotOrdinal !== ordinal) {
    failures.push(`${raw} ordinal: got "${gotOrdinal}", want "${ordinal}"`);
  }
  checked += 1;
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked * 2} checks`);
  for (const line of failures.slice(0, 15)) console.log("  " + line);
  process.exit(1);
}
console.log(`ok  num-az: ${checked * 2} checks against num2words`);
