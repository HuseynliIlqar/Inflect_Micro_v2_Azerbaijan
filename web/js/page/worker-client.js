/**
 * The page's side of js/worker.js: one request at a time, progress through a
 * callback, cancel, and a worker that went silent or broke is replaced.
 */

import { CancelledError } from "../engine/engine.js";

// A worker killed for memory sends nothing at all. If none of its messages
// arrives for this long, it is treated as dead rather than waited on forever.
// A single long chunk on a slow phone's CPU stays well inside it.
const WORKER_SILENCE_MS = 5 * 60 * 1000;

// After Cancel the worker stops at its next chunk and confirms. One chunk on a
// slow phone's CPU can run for a minute, so a worker that has not confirmed
// within this long is terminated; the next request starts a fresh one, with
// the model from the browser's cache.
const CANCEL_GRACE_MS = 2000;

/**
 * `url` is the worker module; `workerError()` builds the error a request
 * fails with when the worker cannot start or dies. The worker is started at
 * once, so its imports download while the visitor is typing.
 */
export function createWorkerClient({ url, workerError }) {
  let pending = null;
  let nextId = 0;
  let worker = null;
  let watchdog = null;
  let cancelling = null; // { id, timer } until the worker confirms

  function armWatchdog() {
    clearTimeout(watchdog);
    watchdog = setTimeout(() => failWorker("silent"), WORKER_SILENCE_MS);
  }

  function failPending(error) {
    clearTimeout(watchdog);
    pending?.reject(error);
    pending = null;
  }

  /** Drop a broken worker; the next request starts a fresh one. */
  function failWorker(reason) {
    console.error("[aztts] worker failed", reason);
    worker?.terminate();
    worker = null;
    failPending(workerError());
  }

  function settleCancel() {
    clearTimeout(cancelling?.timer);
    cancelling = null;
  }

  function dropUnresponsiveWorker() {
    settleCancel();
    worker?.terminate();
    worker = null;
  }

  function onMessage({ data }) {
    // The cancelled request has stopped (or finished first): the worker is
    // free, so it need not be terminated.
    if (data.id === cancelling?.id && data.type !== "progress") {
      settleCancel();
      return;
    }
    if (!pending || data.id !== pending.id) return;
    armWatchdog();
    if (data.type === "progress") pending.onProgress(data);
    else if (data.type === "result") {
      pending.resolve(data);
      pending = null;
    } else if (data.type === "error") failPending(new Error(data.message));
  }

  function startWorker() {
    try {
      const address = new URL(url);
      address.search = location.search; // passes ?backend=wasm through
      const created = new Worker(address, { type: "module" });
      created.addEventListener("message", onMessage);
      // Fires when the module fails to load (a CDN import on a dropped
      // connection, a browser without module workers) or throws at top level.
      created.addEventListener("error", (event) => {
        event.preventDefault();
        failWorker(event.message || "error");
      });
      created.addEventListener("messageerror", () => failWorker("messageerror"));
      return created;
    } catch (error) {
      console.error("[aztts] worker could not start", error);
      return null;
    }
  }

  /** Stop the running request. The page is free at once; the worker follows. */
  function cancel() {
    if (!pending) return;
    const { id } = pending;
    worker?.postMessage({ type: "cancel", id });
    settleCancel();
    cancelling = { id, timer: setTimeout(dropUnresponsiveWorker, CANCEL_GRACE_MS) };
    failPending(new CancelledError());
  }

  /** One request to the worker; progress arrives through `onProgress`. */
  function synthesise(request, onProgress) {
    // Still inside a cancelled chunk: a new request would queue behind it.
    if (cancelling) dropUnresponsiveWorker();
    worker ??= startWorker();
    if (!worker) return Promise.reject(workerError());
    return new Promise((resolve, reject) => {
      nextId += 1;
      pending = { id: nextId, resolve, reject, onProgress };
      armWatchdog();
      worker.postMessage({ id: nextId, ...request });
    });
  }

  worker = startWorker();
  return { synthesise, cancel };
}
