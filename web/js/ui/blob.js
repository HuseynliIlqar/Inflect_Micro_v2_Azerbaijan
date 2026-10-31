/**
 * The orb's outline: a circle pushed in and out by a few slow sine waves, so it
 * breathes rather than spins. Pure arithmetic -- js/page/orb-view.js decides
 * when to draw it, and stops drawing when nothing is happening.
 */

const DEFAULT_POINTS = 72;

/**
 * Points round the outline.
 *
 * @param {number} radius
 * @param {number} amplitude how far the outline strays, as a share of the radius
 * @param {{ phase?: number, offset?: number, level?: number, count?: number }} [options]
 *   phase: time, in radians; offset: a fixed shift so stacked rings differ;
 *   level: 0..1 loudness, which adds a faster ripple
 * @returns {Array<[number, number]>}
 */
export function blobPoints(radius, amplitude, { phase = 0, offset = 0, level = 0, count = DEFAULT_POINTS } = {}) {
  const points = [];
  for (let index = 0; index < count; index += 1) {
    const angle = (index / count) * Math.PI * 2;
    const wobble =
      0.55 * Math.sin(3 * angle + phase * 0.9 + offset) +
      0.3 * Math.sin(5 * angle - phase * 1.3 + 1.7 + offset) +
      0.15 * Math.sin(7 * angle + phase * 2.1 + 0.4 + offset) +
      level * 0.35 * Math.sin(11 * angle + phase * 4 + offset);
    const distance = radius * (1 + amplitude * wobble);
    points.push([distance * Math.cos(angle), distance * Math.sin(angle)]);
  }
  return points;
}

/** A closed path through the midpoints, each point a quadratic control. */
export function smoothPath(points) {
  const count = points.length;
  if (!count) return "";
  const fixed = (value) => value.toFixed(2);
  const middle = (index) => {
    const [ax, ay] = points[index % count];
    const [bx, by] = points[(index + 1) % count];
    return [(ax + bx) / 2, (ay + by) / 2];
  };
  const [startX, startY] = middle(0);
  let path = `M${fixed(startX)} ${fixed(startY)}`;
  for (let index = 1; index <= count; index += 1) {
    const [px, py] = points[index % count];
    const [mx, my] = middle(index);
    path += `Q${fixed(px)} ${fixed(py)} ${fixed(mx)} ${fixed(my)}`;
  }
  return `${path}Z`;
}

export const blobPath = (radius, amplitude, options) => smoothPath(blobPoints(radius, amplitude, options));
