/**
 * The orb above the heading: its shape follows what the page is doing --
 * calm when idle, restless while it works, swelling with the voice while a
 * clip plays.
 *
 * It animates only while something happens and stops once it has settled, so
 * an open tab costs a phone nothing. Under reduced motion it never moves; a
 * hidden tab pauses it.
 */

import { blobPath } from "../ui/blob.js";
import { els } from "./dom.js";

// amplitude: how far the outline strays; speed: radians a second; scale: of the radius.
const TARGETS = {
  check: { amp: 0.13, speed: 2.2, scale: 0.78 },
  busy: { amp: 0.11, speed: 3.2, scale: 0.86 },
  playing: { amp: 0.035, speed: 2.4, scale: 0.98 },
  idle: { amp: 0.05, speed: 0.7, scale: 1 },
};
const ACTIVE = new Set(["check", "busy", "playing"]);
const RADIUS = 90;
// A phone is busy synthesising on its CPU: half the frames and fewer points
// look the same at that size and leave it more time for the model.
const TOUCH = matchMedia("(pointer: coarse)").matches;
const POINTS = TOUCH ? 48 : 72;
const FRAME_MS = TOUCH ? 1000 / 30 : 0;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");

function createOrb() {
  let mode = "idle";
  let levelSource = () => 0;
  const now = { ...TARGETS.idle, level: 0 };
  let phase = 0;
  let last = 0;
  let frame = 0;

  const goal = () => {
    const base = TARGETS[mode] ?? TARGETS.idle;
    if (mode !== "playing") return { ...base, level: 0 };
    const level = levelSource();
    return { amp: base.amp + level * 0.42, speed: base.speed, scale: base.scale + level * 0.14, level };
  };

  const draw = () => {
    const radius = RADIUS * now.scale;
    const options = { phase, level: now.level, count: POINTS };
    els.orbBlob.setAttribute("d", blobPath(radius, now.amp, options));
    els.orbRingInner.setAttribute("d", blobPath(radius * 1.18, now.amp * 1.5, { ...options, offset: 2.1 }));
    els.orbRingOuter.setAttribute("d", blobPath(radius * 1.36, now.amp * 1.9, { ...options, offset: 4.2 }));
    // A compositor-only transform: no filter is recomputed per frame.
    els.orbGlow.style.transform = `scale(${(now.scale * (1 + now.level * 0.3)).toFixed(3)})`;
  };

  const settled = (target) =>
    Math.abs(now.amp - target.amp) < 0.002 && Math.abs(now.scale - target.scale) < 0.002 && now.level < 0.01;

  const tick = (time) => {
    if (FRAME_MS && time - last < FRAME_MS - 1) {
      frame = requestAnimationFrame(tick);
      return;
    }
    const seconds = Math.min(0.05, (time - last) / 1000 || 0.016);
    last = time;
    const target = goal();
    const ease = 1 - Math.exp(-seconds * 5);
    now.amp += (target.amp - now.amp) * ease;
    now.speed += (target.speed - now.speed) * ease;
    now.scale += (target.scale - now.scale) * ease;
    now.level += (target.level - now.level) * (1 - Math.exp(-seconds * 14));
    phase += seconds * now.speed;
    draw();
    const keepGoing = (ACTIVE.has(mode) || !settled(target)) && !document.hidden;
    frame = keepGoing ? requestAnimationFrame(tick) : 0;
  };

  const wake = () => {
    if (frame || reducedMotion.matches || document.hidden) return;
    last = performance.now();
    frame = requestAnimationFrame(tick);
  };

  document.addEventListener("visibilitychange", wake);
  reducedMotion.addEventListener("change", () => {
    if (!reducedMotion.matches) return wake();
    cancelAnimationFrame(frame);
    frame = 0;
  });

  return {
    get mode() {
      return mode;
    },
    /** "idle" | "check" | "busy" | "playing" */
    setMode(next) {
      if (next === mode) return;
      mode = next;
      els.orb.dataset.state = next;
      wake();
    },
    /** A function returning the loudness heard right now, 0..1. */
    setLevelSource(source) {
      levelSource = source;
    },
    /** The ring round the orb: undefined hides it, null spins it, 0..1 fills it. */
    setProgress(fraction) {
      const ring = els.orbProgress;
      ring.classList.toggle("is-visible", fraction !== undefined);
      ring.classList.toggle("is-indeterminate", fraction === null);
      ring.style.strokeDashoffset = typeof fraction === "number" ? String(100 * (1 - fraction)) : "";
    },
    /** Draw the resting shape once, for the first paint after the script loads. */
    start: wake,
  };
}

export const orb = createOrb();
