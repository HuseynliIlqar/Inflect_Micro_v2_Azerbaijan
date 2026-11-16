// Shared pieces of the monitor's charts: an SVG sized to its container in real pixels (so text never stretches),
// clean axis ticks, number formats, a tooltip, and a crosshair that follows the pointer or the arrow keys.

export const NS = "http://www.w3.org/2000/svg";
export const MARGIN = { left: 52, right: 14, top: 22, bottom: 26 };

export function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Up to ~``count`` round tick values covering [lo, hi]. */
export function niceTicks(lo, hi, count = 4) {
  const span = hi - lo || Math.abs(hi) || 1;
  const raw = span / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= raw);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(Number(t.toFixed(10)));
  return ticks;
}

/** As many decimals as the tick step needs: 0.05 steps -> 2, 50 steps -> 0. */
export function tickDigits(ticks) {
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  return step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
}

/** A y-range that follows the bulk of the values: the 1st-99th percentile, so a start-up spike does not flatten
 * the rest of the curve into a line (whatever is outside is clipped at the plot's edge). */
export function robustRange(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const at = (f) => sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(f * (sorted.length - 1))))];
  const lo = sorted.length > 50 ? at(0.01) : sorted[0];
  const hi = sorted.length > 50 ? at(0.99) : sorted.at(-1);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  return [lo - pad, hi + pad];
}

let clipCount = 0;

/** A clip path for the plot area of ``chart``; returns the url() to put on the marks. */
export function plotClip(chart, width, height) {
  clipCount += 1;
  const id = `plot-clip-${clipCount}`;
  const defs = svg("defs");
  const clip = svg("clipPath", { id });
  clip.append(svg("rect", { x: MARGIN.left, y: MARGIN.top, width: Math.max(0, width - MARGIN.left - MARGIN.right),
                            height: height - MARGIN.top - MARGIN.bottom }));
  defs.append(clip);
  chart.append(defs);
  return `url(#${id})`;
}

/** 950 / 12K / 1.5M for steps; fixed digits for values. */
export const compact = (n) => (Math.abs(n) >= 1e6 ? `${+(n / 1e6).toFixed(1)}M`
  : Math.abs(n) >= 1e3 ? `${+(n / 1e3).toFixed(1)}K` : `${n}`);

export function fmt(value, digits) {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("az", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Draws the y gridlines + tick labels and the x tick labels into ``chart``. */
export function axes(chart, { width, height, y, x, yTicks, xTicks, yFormat, xFormat }) {
  for (const t of yTicks) {
    chart.append(svg("line", { x1: MARGIN.left, x2: width - MARGIN.right, y1: y(t), y2: y(t), class: "grid" }));
    chart.append(svg("text", { x: MARGIN.left - 6, y: y(t) + 4, class: "tick", "text-anchor": "end" }, yFormat(t)));
  }
  for (const t of xTicks) {
    chart.append(svg("text", { x: x(t), y: height - 8, class: "tick", "text-anchor": "middle" }, xFormat(t)));
  }
}

/** Renders ``draw(width)`` now and again whenever the container's width changes. */
export function responsive(container, draw) {
  container._observer?.disconnect();
  let last = 0;
  const observer = new ResizeObserver(() => {
    const width = Math.floor(container.clientWidth);
    if (width && width !== last) {
      last = width;
      draw(width);
    }
  });
  observer.observe(container);
  container._observer = observer;
}

/**
 * Crosshair + tooltip over ``chart``: ``xs`` are the points' pixel x positions, ``show(i)`` returns
 * {y, lines: [[strong, rest], ...]} for point i. Pointer, touch and the arrow keys all move it.
 */
export function hover(container, chart, { xs, height, show, label }) {
  const line = svg("line", { y1: MARGIN.top, y2: height - MARGIN.bottom, class: "crosshair", visibility: "hidden" });
  const dot = svg("circle", { r: 5, class: "hover-dot", visibility: "hidden" });
  chart.append(line, dot);
  const tip = document.createElement("div");
  tip.className = "tooltip";
  tip.hidden = true;
  container.append(tip);
  let index = xs.length - 1;

  const place = (i) => {
    index = Math.max(0, Math.min(xs.length - 1, i));
    const { y, lines } = show(index);
    for (const node of [line, dot]) node.setAttribute("visibility", "visible");
    line.setAttribute("x1", xs[index]);
    line.setAttribute("x2", xs[index]);
    dot.setAttribute("cx", xs[index]);
    dot.setAttribute("cy", y);
    tip.replaceChildren(...lines.map(([strong, rest]) => {
      const row = document.createElement("div");
      const value = document.createElement("strong");
      value.textContent = strong;
      row.append(value, document.createTextNode(rest ? ` ${rest}` : ""));
      return row;
    }));
    tip.hidden = false;
    const left = xs[index] + 12 + tip.offsetWidth > container.clientWidth ? xs[index] - 12 - tip.offsetWidth : xs[index] + 12;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${MARGIN.top}px`;
  };
  const hide = () => {
    tip.hidden = true;
    for (const node of [line, dot]) node.setAttribute("visibility", "hidden");
  };
  const nearest = (event) => {
    const box = chart.getBoundingClientRect();
    const px = event.clientX - box.left;
    let best = 0;
    for (let i = 1; i < xs.length; i += 1) if (Math.abs(xs[i] - px) < Math.abs(xs[best] - px)) best = i;
    return best;
  };
  chart.setAttribute("tabindex", "0");
  chart.setAttribute("aria-label", `${label}. Ox düymələri ilə nöqtələr arasında gəzin.`);
  chart.addEventListener("pointermove", (event) => place(nearest(event)));
  chart.addEventListener("pointerdown", (event) => place(nearest(event)));
  chart.addEventListener("pointerleave", (event) => { if (event.pointerType === "mouse") hide(); });
  chart.addEventListener("focus", () => place(index));
  chart.addEventListener("blur", hide);
  chart.addEventListener("keydown", (event) => {
    const moves = { ArrowLeft: -1, ArrowRight: 1, Home: -Infinity, End: Infinity };
    if (!(event.key in moves)) return;
    event.preventDefault();
    place(Number.isFinite(moves[event.key]) ? index + moves[event.key] : moves[event.key] < 0 ? 0 : xs.length - 1);
  });
}
