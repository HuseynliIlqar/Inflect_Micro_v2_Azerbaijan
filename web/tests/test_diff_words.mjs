// "The model will read it like this": which words normalisation rewrote, so the
// preview can mark them (js/text/diff-words.js).
import { diffWords } from "../js/text/diff-words.js";
import { normalizeAz } from "../js/text/az-text.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

expect("identical text is one plain run", diffWords("Salam dünya.", "Salam dünya."), [
  { text: "Salam dünya.", changed: false },
]);
expect("nothing after, nothing to show", diffWords("Salam", ""), []);
expect("all new", diffWords("", "iyirmi beş"), [{ text: "iyirmi beş", changed: true }]);
expect("extra spaces do not count", diffWords("Salam   dünya", "Salam dünya"), [
  { text: "Salam dünya", changed: false },
]);

const before = "II Dünya müharibəsi 01/09/1939 tarixində başladı və 25% artım oldu.";
expect("the date example, against the real normaliser", diffWords(before, normalizeAz(before)), [
  { text: "İkinci", changed: true },
  { text: "Dünya müharibəsi", changed: false },
  { text: "bir sentyabr min doqquz yüz otuz doqquzuncu il", changed: true },
  { text: "tarixində başladı və", changed: false },
  { text: "iyirmi beş faiz", changed: true },
  { text: "artım oldu.", changed: false },
]);

const money = "Qiymət 19,99 AZN-dir.";
const runs = diffWords(money, normalizeAz(money));
expect("joined runs give back the normalised text", runs.map((r) => r.text).join(" "), normalizeAz(money));
expect("the amount is marked", runs.some((r) => r.changed && /manat/u.test(r.text)), true);
expect("runs alternate", runs.every((r, i) => i === 0 || r.changed !== runs[i - 1].changed), true);

if (failures.length) {
  for (const f of failures) console.error("FAIL", f.name, "\n  got ", JSON.stringify(f.got), "\n  want", JSON.stringify(f.want));
  process.exit(1);
}
console.log(`ok  diff-words: ${checked} checks on what the preview marks as rewritten`);
