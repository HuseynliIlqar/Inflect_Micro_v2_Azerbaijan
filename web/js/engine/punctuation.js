/**
 * Punctuation the way the Python frontends handle it, so the browser feeds the
 * model the phonemes it was trained on.
 *
 * Both voices were phonemised with `phonemizer`'s `preserve_punctuation=True`:
 * the text is cut at every punctuation mark, each piece goes to eSpeak NG on
 * its own, and the marks -- with the spaces around them -- are put back. eSpeak
 * on a whole sentence is not the same: it drops brackets and quotes, and words
 * on either side of them then share one clause, which moves the stress
 * ("Bu (söz)" -> "bu" instead of "bˈu"). This is phonemizer 3.x's
 * `Punctuation.preserve` / `restore` for one line with `strip=True`, ported.
 */

// phonemizer's default marks. A comma or full stop between two digits is a
// decimal separator, not a mark.
const MARKS = /(\s*(?:[;:!?¡¿—…"«»“”(){}[\]]|(?<![0-9])[,.]|[,.](?![0-9]))+\s*)+/gu;

/**
 * One line -> [the pieces between the marks, the marks with their positions]:
 * "B" begins the line, "E" ends it, "I" is inside, "A" is all of it.
 */
export function preserve(line) {
  const matches = [...line.matchAll(MARKS)];
  // phonemizer drops empty pieces, so an empty line never reaches eSpeak.
  if (!matches.length) return [[line].filter(Boolean), []];
  if (matches.length === 1 && matches[0][0] === line) return [[], [{ mark: line, position: "A" }]];
  const last = matches.length - 1;
  const marks = matches.map(([mark], index) => {
    if (index === 0 && line.startsWith(mark)) return { mark, position: "B" };
    if (index === last && line.endsWith(mark)) return { mark, position: "E" };
    return { mark, position: "I" };
  });
  const pieces = [];
  let rest = line;
  for (const { mark } of marks) {
    const at = rest.indexOf(mark);
    pieces.push(rest.slice(0, at));
    rest = rest.slice(at + mark.length);
  }
  pieces.push(rest);
  return [pieces.filter(Boolean), marks];
}

/** The phonemised pieces and the marks of one line -> that line, punctuated. */
export function restore(phonemized, marks) {
  const text = [...phonemized];
  let pending = [...marks];
  const lines = [];
  // phonemizer compares each mark's line number with this; one line is number 0.
  let position = 0;
  while (text.length || pending.length) {
    if (!pending.length) {
      lines.push(...text.splice(0));
    } else if (!text.length) {
      lines.push(pending.map(({ mark }) => mark).join(""));
      pending = [];
    } else if (position === 0) {
      const [{ mark, position: where }] = pending;
      pending = pending.slice(1);
      if (text[0].endsWith(" ")) text[0] = text[0].slice(0, -1);
      if (where === "B") {
        text[0] = mark + text[0];
      } else if (where === "E") {
        lines.push(text.shift() + mark);
        position += 1;
      } else if (where === "A") {
        lines.push(mark);
        position += 1;
      } else if (text.length === 1) {
        text[0] += mark;
      } else {
        const first = text.shift();
        text[0] = first + mark + text[0];
      }
    } else {
      lines.push(text.shift());
      position += 1;
    }
  }
  // The backend is asked for one line and its first result is taken.
  return lines[0] ?? "";
}

/** text -> phonemes, with `phonemizePiece` (eSpeak on punctuation-free text) per piece. */
export async function withPunctuation(text, phonemizePiece) {
  const [pieces, marks] = preserve(text);
  const phonemized = [];
  for (const piece of pieces) phonemized.push(await phonemizePiece(piece));
  return restore(phonemized, marks);
}
