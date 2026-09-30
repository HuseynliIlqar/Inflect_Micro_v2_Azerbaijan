/**
 * The result's waveform on the page: drawn once a clip is ready, redrawn as it
 * plays, when the canvas changes width and when the theme flips. The drawing
 * itself is js/waveform.js; this is the wiring to the DOM.
 */

import { barCount, drawWave, peaks, seekFraction } from "./waveform.js";

/**
 * @param {{ canvas: HTMLCanvasElement, box: HTMLElement, audio: HTMLAudioElement }} els
 * @returns {{ show(samples: Float32Array): void }}
 */
export function createWaveView({ canvas, box, audio }) {
  // The last clip's samples, kept so the bars can be recomputed when the
  // canvas changes width.
  let samples = null;
  let heights = new Float32Array(0);
  let frame = 0;

  const colours = () => {
    const style = getComputedStyle(document.documentElement);
    return {
      rest: style.getPropertyValue("--wave-rest").trim(),
      heard: style.getPropertyValue("--accent").trim(),
    };
  };

  const render = () => {
    if (!samples) return;
    const { duration, currentTime } = audio;
    const played = duration > 0 ? currentTime / duration : 0;
    drawWave(canvas, heights, { played, ...colours() });
  };

  const measure = () => {
    if (!samples) return;
    heights = peaks(samples, barCount(canvas.clientWidth));
    render();
  };

  // While playing, follow the playhead every frame; otherwise redraw on events.
  const followPlayback = () => {
    cancelAnimationFrame(frame);
    const step = () => {
      render();
      if (!audio.paused && !audio.ended) frame = requestAnimationFrame(step);
    };
    step();
  };

  for (const event of ["play", "pause", "seeked", "ended", "loadedmetadata"]) {
    audio.addEventListener(event, followPlayback);
  }

  // A pointer shortcut for seeking; the <audio> controls remain the keyboard way.
  canvas.addEventListener("click", (event) => {
    const { duration } = audio;
    if (!samples || !(duration > 0)) return;
    audio.currentTime = seekFraction(event.clientX, canvas.getBoundingClientRect()) * duration;
    render();
  });

  new ResizeObserver(measure).observe(canvas);
  // The colours are theme tokens: redraw when the theme or the device scheme flips.
  new MutationObserver(render).observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);

  return {
    show(next) {
      samples = next;
      box.dataset.state = "ready";
      measure();
    },
  };
}
