/**
 * Which words of the normalised text are new: the preview marks them, so a
 * visitor sees "25%" become "iyirmi beş faiz" before pressing Speak.
 *
 * A word-level longest common subsequence -- a text is at most 500 characters,
 * so the table stays small.
 */

const words = (text) => text.split(/\s+/u).filter(Boolean);

/**
 * @param {string} before what was typed
 * @param {string} after what the model will read
 * @returns {Array<{ text: string, changed: boolean }>} runs of `after`, in order;
 *   joined with single spaces they give `after` back
 */
export function diffWords(before, after) {
  const a = words(before);
  const b = words(after);
  if (!b.length) return [];

  // lengths[i][j]: the common subsequence of a[i..] and b[j..].
  const lengths = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      lengths[i][j] = a[i] === b[j]
        ? lengths[i + 1][j + 1] + 1
        : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }

  const marks = [];
  let i = 0;
  let j = 0;
  while (j < b.length) {
    if (i < a.length && a[i] === b[j]) {
      marks.push(false);
      i += 1;
      j += 1;
    } else if (i < a.length && lengths[i + 1][j] >= lengths[i][j + 1]) {
      i += 1;
    } else {
      marks.push(true);
      j += 1;
    }
  }

  const runs = [];
  b.forEach((word, index) => {
    const last = runs[runs.length - 1];
    if (last && last.changed === marks[index]) last.text += ` ${word}`;
    else runs.push({ text: word, changed: marks[index] });
  });
  return runs;
}
