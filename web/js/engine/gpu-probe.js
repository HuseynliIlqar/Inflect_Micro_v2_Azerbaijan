/**
 * Does this browser offer WebGPU, and does it hand out a working device?
 *
 * The first half of the device check (js/engine/calibrate.js does the rest).
 * `navigator.gpu` alone proves nothing: many phones expose it and return no
 * adapter, and some return one whose device will not start. Only a GPU that
 * passes here is loaded and measured.
 */

const messageOf = (error) => String(error?.message ?? error);

/**
 * @param {{ nav?: object, preferWasm?: boolean }} options
 * @returns {Promise<{ status: "ok"|"unsupported"|"no-adapter"|"software"|"device-failed"|"off",
 *                     vendor?: string, architecture?: string, message?: string }>}
 *   software: the browser's CPU emulation of a GPU, never faster than wasm.
 */
export async function probeGpu({ nav = globalThis.navigator, preferWasm = false } = {}) {
  if (preferWasm) return { status: "off" };
  const gpu = nav?.gpu;
  if (!gpu?.requestAdapter) return { status: "unsupported" };
  let adapter;
  try {
    adapter = await gpu.requestAdapter();
  } catch (error) {
    return { status: "no-adapter", message: messageOf(error) };
  }
  if (!adapter) return { status: "no-adapter" };
  const info = adapter.info ?? {};
  const identity = { vendor: info.vendor ?? "", architecture: info.architecture ?? "" };
  if (info.isFallbackAdapter ?? adapter.isFallbackAdapter) return { status: "software", ...identity };
  try {
    const device = await adapter.requestDevice();
    device?.destroy?.();
  } catch (error) {
    return { status: "device-failed", ...identity, message: messageOf(error) };
  }
  return { status: "ok", ...identity };
}

/**
 * True when a WebGPU adapter exists -- the quick check the chain uses when the
 * device check has not run. `navigator.gpu` alone is not enough.
 */
export async function hasWebGpu() {
  try {
    const gpu = globalThis.navigator?.gpu;
    return Boolean(gpu && (await gpu.requestAdapter()));
  } catch {
    return false;
  }
}

/** What identifies this GPU for a stored verdict: a new driver or GPU is measured again. */
export function adapterKey({ status, vendor = "", architecture = "" }) {
  return `${status}:${vendor}:${architecture}`;
}
