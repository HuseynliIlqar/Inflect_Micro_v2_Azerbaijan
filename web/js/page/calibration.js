/**
 * The device check on the page: on the first visit, block the page, have the
 * worker measure this device (js/engine/device-check.js), show the verdict,
 * and remember it; on later visits, reuse what was measured.
 *
 * The stored verdict is keyed on everything it measured -- the graphs, the
 * browser, its threads and its GPU -- so a change to any of them measures
 * again (js/ui/calibration-store.js).
 */

import { REFERENCE } from "../engine/calibration-reference.js";
import { adapterKey, probeGpu } from "../engine/gpu-probe.js";
import { advance, finish, startProgress } from "../ui/calibration-progress.js";
import { calibrationCopy, measuredDeviceVerdict } from "../ui/calibration-copy.js";
import { browserStorage, forgetStored, readStored, writeStored } from "../ui/calibration-store.js";
import { createCalibrationView } from "./calibration-view.js";
import { client } from "./client.js";
import { els } from "./dom.js";
import { device, guessDevice } from "./device.js";
import { renderPhoneNote } from "./mode-view.js";
import { renderOrb } from "./busy-view.js";
import {
  APPLE_MOBILE, EXPECTED_THREADS, IN_FRAME, PHONE, QUERY, state, t,
} from "./context.js";

const view = createCalibrationView();
const storage = browserStorage();
const deviceFacts = { phone: PHONE, appleMobile: APPLE_MOBILE };
let storeKey = null;

/** Speak is locked while the check runs; see `state.checking`. */
function setChecking(on) {
  state.checking = on;
  els.speak.disabled = on;
  renderOrb();
}

function accept(calibration, { announce }) {
  state.calibration = calibration;
  const verdict = measuredDeviceVerdict(calibration, deviceFacts);
  if (announce) device.announce(verdict);
  else device.show(verdict);
  renderPhoneNote();
}

/** The check could not run (or was skipped): fall back to the old guess. */
function giveUp() {
  state.calibration = null;
  renderPhoneNote();
  guessDevice();
}

async function measure() {
  if (!view.open(t)) {
    // No modal dialogs in this browser: measure without the screen.
    return measureQuietly();
  }
  let progress = startProgress();
  view.render(progress);
  try {
    const result = await client.synthesise(
      { kind: "calibrate", options: { framed: IN_FRAME } },
      (message) => {
        progress = advance(progress, message);
        view.render(progress);
      },
    );
    view.render(finish(progress));
    writeStored(storage, storeKey, result.calibration);
    accept(result.calibration, { announce: false });
    // The check left the model loaded: "loading the model" no longer holds.
    if (els.status.textContent === t("loading")) els.status.textContent = "";
    await view.showResult(calibrationCopy(result.calibration, deviceFacts));
    view.close();
  } catch (error) {
    console.error("[aztts] device check failed", error);
    const choice = await view.showError(t("cal_error", { reason: String(error?.message ?? error) }));
    if (choice === "retry") return measure();
    view.close();
    giveUp();
  }
  return undefined;
}

async function measureQuietly() {
  try {
    const result = await client.synthesise({ kind: "calibrate", options: { framed: IN_FRAME } }, () => {});
    writeStored(storage, storeKey, result.calibration);
    accept(result.calibration, { announce: true });
  } catch (error) {
    console.error("[aztts] device check failed", error);
    giveUp();
  }
}

/** Reuse a stored verdict, or measure. A stored one that will not read is measured over. */
async function check() {
  // The page's own quick look at the GPU keys the stored verdict; the worker
  // probes again for itself when it measures.
  const probe = await probeGpu({ preferWasm: QUERY.preferWasm });
  storeKey = {
    version: REFERENCE.version,
    userAgent: navigator.userAgent,
    threads: EXPECTED_THREADS,
    adapter: adapterKey(probe),
  };
  const stored = readStored(storage, storeKey, Date.now());
  if (!stored) return measure();
  try {
    accept(stored, { announce: true });
    return undefined;
  } catch (error) {
    console.error("[aztts] stored device check unreadable", error);
    forgetStored(storage);
    return measure();
  }
}

async function locked(run) {
  setChecking(true);
  try {
    await run();
  } catch (error) {
    console.error("[aztts] device check failed", error);
    view.close();
    giveUp();
  } finally {
    setChecking(false);
  }
}

/** On page load: reuse a stored verdict, or measure this device now. */
export function startDeviceCheck() {
  return locked(check);
}

/** "Measure again" in the device dialog. */
export function remeasure() {
  if (state.busy || state.checking || !storeKey) return undefined;
  forgetStored(storage);
  return locked(measure);
}
