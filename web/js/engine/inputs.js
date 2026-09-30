/**
 * What the two graphs are fed: token ids from a phoneme string, and the
 * seeded noise the decoder draws a reading from.
 */

import { SYMBOLS } from "./symbols.js";

const SYMBOL_TO_ID = new Map(SYMBOLS.map((symbol, index) => [symbol, index]));

/** A deterministic normal generator, so a seed reproduces a reading. */
export function makeNoise(seed) {
  // mulberry32, then Box-Muller. Not torch's generator -- see the note above.
  let state = seed >>> 0;
  const uniform = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => {
    const u = Math.max(uniform(), Number.MIN_VALUE);
    const v = uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

/** Phoneme string -> token ids, with the blank symbol interspersed. */
export function tokenise(phonemes) {
  const ids = [];
  for (const character of phonemes) {
    const id = SYMBOL_TO_ID.get(character);
    if (id !== undefined) ids.push(id);
  }
  // add_blank: a 0 before, between and after every token.
  const spread = new Array(ids.length * 2 + 1).fill(0);
  for (let index = 0; index < ids.length; index += 1) {
    spread[index * 2 + 1] = ids[index];
  }
  return spread;
}
