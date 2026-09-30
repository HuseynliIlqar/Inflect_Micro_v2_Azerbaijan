/**
 * What the device check's screen shows while it measures: a bar, one line
 * naming the current stage, and a row per step with its outcome.
 *
 * A pure reducer over the worker's progress messages (js/engine/calibrate.js):
 * each call returns a new state, so the screen is tested without a DOM. Text
 * is returned as label keys and values; js/ui/i18n-calibration.js holds them.
 */

import { megabytes } from "./progress.js";

// The bar: the GPU check, then the backends in equal shares, then the result.
const PROBE_SHARE = 0.05;
const TRIALS_SHARE = 0.9;
// Within one backend: downloading, compiling, then the runs.
const DOWNLOAD_PART = 0.4;
const COMPILE_PART = 0.45;
const FINALISE_AT = 0.96;

const idOf = ({ backend, precision }) => `${backend}/${precision}`;

function backendKey({ backend, precision }) {
  if (backend === "webgpu") return "cal_backend_gpu";
  return precision === "int8" ? "cal_backend_cpu" : "cal_backend_cpu_full";
}

const rowLabel = (step) => (step.backend === "webgpu" ? "cal_row_gpu"
  : step.precision === "int8" ? "cal_row_cpu" : "cal_row_cpu_full");

function detailOf({ ok, reason, rtf }) {
  if (ok) return { key: "cal_detail_speed", values: { rtf: rtf.toFixed(2) } };
  if (reason === "unstable") return { key: "cal_detail_unstable", values: {} };
  if (["load", "run", "timeout"].includes(reason)) return { key: "cal_detail_error", values: {} };
  return { key: "cal_detail_wrong", values: {} };
}

export function startProgress() {
  return {
    fraction: 0,
    stage: { key: "cal_stage_probe", values: {}, backendKey: null },
    rows: [{ id: "probe", status: "active", label: "cal_row_probe", detail: null }],
    trials: 0,
    done: 0,
    part: 0,
    finalising: false,
  };
}

const setRow = (rows, id, change) => rows.map((row) => (row.id === id ? { ...row, ...change } : row));

/** A backend row, added before the result row if the plan grew (int8 failed). */
function ensureRow(state, step) {
  const id = idOf(step);
  if (state.rows.some((row) => row.id === id)) return state;
  const row = { id, status: "pending", label: rowLabel(step), detail: null };
  const rows = [...state.rows];
  const last = rows.findIndex((r) => r.id === "finalise");
  rows.splice(last === -1 ? rows.length : last, 0, row);
  return { ...state, rows, trials: state.trials + 1 };
}

function withFraction(state) {
  const trials = Math.max(state.trials, 1);
  const measured = PROBE_SHARE + TRIALS_SHARE * Math.min(1, (state.done + state.part) / trials);
  const target = state.finalising ? Math.max(measured, FINALISE_AT) : measured;
  return { ...state, fraction: Math.min(0.99, Math.max(state.fraction, target)) };
}

function onStep(state, message, part, stage) {
  if (state.finalising) return state;
  const grown = ensureRow(state, message);
  const rows = setRow(grown.rows, idOf(message), { status: "active" });
  return withFraction({ ...grown, rows, part, stage: { ...stage, backendKey: backendKey(message) } });
}

/** The next state after one progress message from the worker. */
export function advance(state, message) {
  switch (message.stage) {
    case "probed": {
      const ok = message.gpu === "ok";
      const key = ok ? "cal_probe_ok" : message.gpu === "off" ? "cal_probe_off" : "cal_probe_none";
      const rows = setRow(state.rows, "probe", {
        status: ok ? "done" : "skipped",
        detail: { key, values: {} },
      });
      return withFraction({ ...state, rows, fraction: PROBE_SHARE });
    }
    case "candidates": {
      const rows = [
        ...state.rows,
        ...message.steps.map((step) => ({ id: idOf(step), status: "pending", label: rowLabel(step), detail: null })),
        { id: "finalise", status: "pending", label: "cal_row_finalise", detail: null },
      ];
      return { ...state, rows, trials: message.steps.length };
    }
    case "download": {
      const values = { loaded: megabytes(message.loaded), total: megabytes(message.total) };
      const share = message.total > 0 ? DOWNLOAD_PART * Math.min(1, message.loaded / message.total) : 0;
      return onStep(state, message, share, { key: "cal_stage_download", values });
    }
    case "compile":
      return onStep(state, message, DOWNLOAD_PART, { key: "cal_stage_compile", values: {} });
    case "test": {
      const part = COMPILE_PART + (1 - COMPILE_PART) * (message.run / (message.runs + 1));
      const stage = message.run === 0
        ? { key: "cal_stage_warmup", values: {} }
        : { key: "cal_stage_test", values: { run: message.run, runs: message.runs } };
      return onStep(state, message, part, stage);
    }
    case "tested": {
      const grown = ensureRow(state, message);
      const rows = setRow(grown.rows, idOf(message), {
        status: message.ok ? "done" : "failed",
        detail: detailOf(message),
      });
      return withFraction({ ...grown, rows, done: grown.done + 1, part: 0 });
    }
    case "finalise": {
      const rows = setRow(state.rows, "finalise", { status: "active" });
      return withFraction({
        ...state, rows, finalising: true, stage: { key: "cal_stage_finalise", values: {}, backendKey: null },
      });
    }
    default:
      return state;
  }
}

/** The check is over: the bar full, the last row done. */
export function finish(state) {
  return { ...state, fraction: 1, rows: setRow(state.rows, "finalise", { status: "done" }) };
}
