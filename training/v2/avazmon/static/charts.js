// A loss over training steps: the mean as a 2px line, the min-max of each point as a faint band (the noise), the
// training stages as light background bands with their names, clean axes, and a crosshair tooltip.
import { MARGIN, axes, compact, fmt, hover, niceTicks, plotClip, responsive, robustRange, svg, tickDigits } from "./chartkit.js?v=3";

export const STAGE_NAMES = {
  posterior_warmup: "İsinmə",
  linguistic_adaptation: "Dil uyğunlaşması",
  decoder_polish: "Vocoder cilalanması",
  disc_warmup: "Polis isinməsi",
};
const HEIGHT = 210;

/** A centred moving average over about 1/15 of the points: the trend under the step-to-step noise. */
function trend(values) {
  const half = Math.max(1, Math.round(values.length / 30));
  return values.map((_, i) => {
    const window = values.slice(Math.max(0, i - half), i + half + 1);
    return window.reduce((sum, v) => sum + v, 0) / window.length;
  });
}

function stageSegments(points) {
  const segments = [];
  for (const p of points) {
    const last = segments.at(-1);
    if (last && last.stage === p.stage) last.end = p.step;
    else segments.push({ stage: p.stage, start: last ? last.end : p.step, end: p.step });
  }
  return segments;
}

/**
 * @param {HTMLElement} container
 * @param {Array<Record<string, any>>} points history points (step, stage, epoch, key, key_min, key_max)
 * @param {{key: string, label: string, digits?: number, unit?: string, band?: boolean}} spec
 */
export function lineChart(container, points, spec) {
  const { key, label, digits = 3, unit = "", band = true } = spec;
  const usable = points.filter((p) => Number.isFinite(p[key]));
  container.replaceChildren();
  if (usable.length < 2) {
    container.textContent = "Hələ məlumat azdır.";
    return;
  }
  const header = document.createElement("div");
  header.className = "chart-head";
  const now = document.createElement("strong");
  now.textContent = `indi ${fmt(usable.at(-1)[key], digits)}${unit}`;
  header.append(now);
  const plot = document.createElement("div");
  plot.className = "plot";
  container.append(header, plot);

  responsive(plot, (width) => {
    plot.replaceChildren();
    const lows = usable.map((p) => (band ? p[`${key}_min`] ?? p[key] : p[key]));
    const highs = usable.map((p) => (band ? p[`${key}_max`] ?? p[key] : p[key]));
    // The axis follows the mean line (and half the band); a start-up spike is clipped, the tooltip keeps its value.
    const [rlo, rhi] = robustRange([...usable.map((p) => p[key]), ...lows.map((v, i) => (v + usable[i][key]) / 2),
                                    ...highs.map((v, i) => (v + usable[i][key]) / 2)]);
    const yTicks = niceTicks(rlo, rhi, 4).filter((t) => t >= rlo && t <= rhi);
    const lo = rlo, hi = rhi;
    const first = usable[0].step, last = usable.at(-1).step;
    const x = (s) => MARGIN.left + ((s - first) / (last - first || 1)) * (width - MARGIN.left - MARGIN.right);
    const y = (v) => MARGIN.top + (1 - (v - lo) / (hi - lo || 1)) * (HEIGHT - MARGIN.top - MARGIN.bottom);
    const chart = svg("svg", { width, height: HEIGHT, viewBox: `0 0 ${width} ${HEIGHT}`, role: "img" });

    stageSegments(usable).forEach((seg, i) => {
      const x0 = x(seg.start), x1 = x(seg.end);
      chart.append(svg("rect", { x: x0, y: MARGIN.top, width: Math.max(0, x1 - x0), class: `stage-band s${i % 2}`,
                                 height: HEIGHT - MARGIN.top - MARGIN.bottom }));
      if (x1 - x0 > 70) chart.append(svg("text", { x: x0 + 6, y: MARGIN.top - 8, class: "stage-name" },
                                        STAGE_NAMES[seg.stage] || seg.stage));
    });
    const tickFmt = tickDigits(yTicks);
    axes(chart, { width, height: HEIGHT, x, y, yTicks, xTicks: niceTicks(first, last, Math.max(3, Math.floor(width / 90))),
                  yFormat: (t) => fmt(t, tickFmt), xFormat: compact });
    const clip = plotClip(chart, width, HEIGHT);
    if (band) {
      const upper = usable.map((p, i) => `${x(p.step)},${y(highs[i])}`);
      const lower = usable.map((p, i) => `${x(p.step)},${y(lows[i])}`).reverse();
      chart.append(svg("polygon", { points: [...upper, ...lower].join(" "), class: "spread", "clip-path": clip }));
    }
    const smooth = trend(usable.map((p) => p[key]));
    chart.append(svg("polyline", { points: usable.map((p) => `${x(p.step)},${y(p[key])}`).join(" "), class: "series raw",
                                   "clip-path": clip }));
    chart.append(svg("polyline", { points: usable.map((p, i) => `${x(p.step)},${y(smooth[i])}`).join(" "),
                                   class: "series", "clip-path": clip }));
    const end = usable.at(-1);
    chart.append(svg("circle", { cx: x(end.step), cy: Math.min(Math.max(y(smooth.at(-1)), MARGIN.top), HEIGHT - MARGIN.bottom),
                                 r: 4, class: "end-dot" }));
    plot.append(chart);

    hover(plot, chart, {
      xs: usable.map((p) => x(p.step)), height: HEIGHT, label,
      show: (i) => {
        const p = usable[i];
        const lines = [[`${fmt(p[key], digits)}${unit}`, label], [`${fmt(smooth[i], digits)}${unit}`, "meyl (hamarlanmış)"]];
        if (band && p[`${key}_min`] != null) lines.push([`${fmt(p[`${key}_min`], digits)}–${fmt(p[`${key}_max`], digits)}`, "aralıq"]);
        lines.push([`${p.step.toLocaleString("az")}`, `addım${p.epoch ? ` · epoch ${p.epoch}` : ""}`]);
        lines.push(["", STAGE_NAMES[p.stage] || p.stage]);
        return { y: Math.min(Math.max(y(p[key]), MARGIN.top), HEIGHT - MARGIN.bottom), lines };
      },
    });
  });
}
