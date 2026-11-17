// Punctuation must be cut and put back exactly as phonemizer's preserve_punctuation does.
// Golden file: tools/build_web_frontend.py, from phonemizer's own Punctuation class.
import { readFileSync } from "node:fs";
import { preserve, restore, withPunctuation } from "../js/engine/punctuation.js";

const golden = JSON.parse(readFileSync(new URL("./golden/punctuation.json", import.meta.url), "utf-8"));
const failures = [];
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

for (const [line, pieces, marks, restored] of golden) {
  const [gotPieces, gotMarks] = preserve(line);
  const markPairs = gotMarks.map(({ mark, position }) => [mark, position]);
  if (!same(gotPieces, pieces) || !same(markPairs, marks)) {
    failures.push({ line, got: [gotPieces, markPairs], want: [pieces, marks] });
    continue;
  }
  const gotRestored = restore(pieces.map((piece) => `<${piece}>`), gotMarks);
  if (gotRestored !== restored) failures.push({ line, got: gotRestored, want: restored });
}

// An empty line has no pieces, as in phonemizer, so eSpeak is never called for it.
if (!same(preserve(""), [[], []])) failures.push({ line: "", got: preserve(""), want: [[], []] });
if ((await withPunctuation("", async () => "x")) !== "") failures.push({ line: "", got: "x", want: "" });

// The wrapper hands each piece to the phonemiser, in order, and nothing else.
const asked = [];
const result = await withPunctuation("Salam, dünya!", async (piece) => {
  asked.push(piece);
  return piece.toUpperCase();
});
if (!same(asked, ["Salam", "dünya"]) || result !== "SALAM, DÜNYA!") {
  failures.push({ line: "withPunctuation", got: [asked, result], want: [["Salam", "dünya"], "SALAM, DÜNYA!"] });
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${golden.length}`);
  for (const f of failures.slice(0, 10)) {
    console.log(`  line: ${JSON.stringify(f.line)}`);
    console.log(`    got : ${JSON.stringify(f.got)}`);
    console.log(`    want: ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  punctuation: ${golden.length} lines cut and restored as phonemizer does`);
