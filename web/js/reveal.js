/**
 * After a clip is ready, bring the result into view -- on a phone it sits a
 * screen or two below the Speak button, and people did not find it.
 *
 * Pure, so the rule is tested without a browser.
 */

/**
 * Whether the result's box should be scrolled to: its top is above the screen,
 * or in the lower half of it, where only the heading would show.
 *
 * @param {{ top: number, bottom: number }} rect  from getBoundingClientRect()
 * @param {number} viewportHeight
 */
export function needsReveal({ top }, viewportHeight) {
  if (!(viewportHeight > 0)) return false;
  return top < 0 || top > viewportHeight * 0.5;
}
