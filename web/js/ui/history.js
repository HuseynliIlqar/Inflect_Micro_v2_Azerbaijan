/**
 * The clips made in this visit, newest first. Pure: the page keeps the list
 * and revokes the object URL of every clip this hands back as dropped.
 */

export const HISTORY_MAX = 6;

// What makes two clips the same take: the clip's key (text, voice and every
// setting) when it has one, else its text and voice.
const takeOf = (clip) => clip.key ?? `${clip.voice}|${clip.text}`;

/**
 * @template {{ text: string, voice: string, key?: string }} Clip
 * @param {Clip[]} list
 * @param {Clip} clip
 * @param {number} [max]
 * @returns {{ list: Clip[], dropped: Clip[] }}
 */
export function addToHistory(list, clip, max = HISTORY_MAX) {
  const same = (other) => takeOf(other) === takeOf(clip);
  const replaced = list.filter(same);
  const kept = [clip, ...list.filter((other) => !same(other))];
  return { list: kept.slice(0, max), dropped: [...replaced, ...kept.slice(max)] };
}
