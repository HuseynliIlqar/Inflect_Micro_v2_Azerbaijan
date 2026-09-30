/**
 * Split Azerbaijani text into chunks short enough for a 9.36M model.
 *
 * A port of `aztts/az_chunk.py`. The model loses intonation well before the
 * package's own 280-character limit: the tail of a long clause flattens and
 * word endings get clipped. Fifteen words keeps every chunk inside the range
 * it handles well.
 *
 * Punctuation stays on each chunk, because the pause between chunks is derived
 * from the trailing mark.
 */

export const DEFAULT_MAX_WORDS = 15;

// Conjunctions that carry a natural breath before them; the second-choice
// split point when a clause has no comma to cut at.
const CONJUNCTIONS = new Set([
  "və", "amma", "ancaq", "lakin", "çünki", "ki", "ya", "yaxud",
  "ona görə", "buna görə", "həmçinin", "yəni",
]);

function words(text) {
  return text.split(/\s+/u).filter(Boolean);
}

/** Cut after commas, keeping the comma so the pause survives. */
function splitAtCommas(sentence) {
  return sentence
    .split(/(?<=,)\s+/u)
    .map((part) => part.trim())
    .filter(Boolean);
}

/** Last resort: cut on a fixed word count. */
function splitHard(clause, maxWords) {
  const all = words(clause);
  const chunks = [];
  for (let start = 0; start < all.length; start += maxWords) {
    chunks.push(all.slice(start, start + maxWords).join(" "));
  }
  return chunks;
}

/** Break before a conjunction near the middle, when one is available. */
function splitAtConjunctions(clause, maxWords) {
  const all = words(clause);
  if (all.length <= maxWords) return [clause];

  const lowered = all.map((word) =>
    word.replace(/^[,.;:!?]+|[,.;:!?]+$/gu, "").toLowerCase(),
  );
  const candidates = [];
  for (let index = 0; index < all.length; index += 1) {
    if (CONJUNCTIONS.has(lowered[index]) && index >= 2 && index <= all.length - 3) {
      candidates.push(index);
    }
  }
  if (!candidates.length) return splitHard(clause, maxWords);

  const middle = Math.floor(all.length / 2);
  let cut = candidates[0];
  for (const index of candidates) {
    if (Math.abs(index - middle) < Math.abs(cut - middle)) cut = index;
  }
  return [
    ...splitAtConjunctions(all.slice(0, cut).join(" "), maxWords),
    ...splitAtConjunctions(all.slice(cut).join(" "), maxWords),
  ];
}

/**
 * The text as chunks of at most `maxWords` words each.
 *
 * Split points are tried in order of how natural the resulting pause sounds:
 * sentence end, then comma, then conjunction, then a plain word count.
 */
export function chunkText(text, maxWords = DEFAULT_MAX_WORDS) {
  if (maxWords < 1) throw new Error("maxWords must be at least 1");
  const normalized = text.split(/\s+/u).filter(Boolean).join(" ");
  if (!normalized) return [];

  const chunks = [];
  for (const raw of normalized.split(/(?<=[.!?;:])\s+/u)) {
    const sentence = raw.trim();
    if (!sentence) continue;
    if (words(sentence).length <= maxWords) {
      chunks.push(sentence);
      continue;
    }
    for (const clause of splitAtCommas(sentence)) {
      chunks.push(...splitAtConjunctions(clause, maxWords));
    }
  }
  return chunks.filter(Boolean);
}
