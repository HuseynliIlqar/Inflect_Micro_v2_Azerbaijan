/**
 * Downloads the ONNX graphs with byte progress, and keeps them.
 *
 * The browser's HTTP cache cannot keep them on a Hugging Face static Space:
 * every request for `onnx/decode.onnx` is answered with a `no-store` redirect
 * to a freshly signed CDN URL, so each visit is a new URL and a new 37 MB
 * download -- painful on a phone on mobile data. Cache Storage is keyed on the
 * page's own path instead, so the second visit reads from disk.
 *
 * Bump CACHE_NAME whenever the graphs on the Space change; old caches are
 * deleted on the next load.
 */

export const CACHE_NAME = "aztts-onnx-v1";

// A 37 MB stream arrives in hundreds of reads; the page needs a few a second.
const REPORT_EVERY_MS = 200;

async function openCache() {
  try {
    if (!globalThis.caches) return null;
    const names = await caches.keys();
    await Promise.all(
      names
        .filter((name) => name.startsWith("aztts-onnx-") && name !== CACHE_NAME)
        .map((name) => caches.delete(name)),
    );
    return await caches.open(CACHE_NAME);
  } catch {
    // Private windows and blocked storage throw here; downloading still works.
    return null;
  }
}

async function readCached(cache, url) {
  if (!cache) return null;
  try {
    const hit = await cache.match(url);
    return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
  } catch {
    return null;
  }
}

async function download(url, onBytes, onTotal) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not download ${url} (HTTP ${response.status})`);
  const total = Number(response.headers.get("content-length")) || 0;
  onTotal(total);

  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onBytes(bytes.length);
    return bytes;
  }

  const reader = response.body.getReader();
  const parts = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    received += value.length;
    onBytes(received);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.length;
  }
  return bytes;
}

/**
 * Every URL as bytes, in order. `onProgress(loaded, total)` reports the sum
 * over all files; `total` is 0 until every file's size is known, and also
 * when a size turns out wrong (a compressed response reports the compressed
 * length but reads decompressed bytes).
 */
export async function fetchModels(urls, onProgress) {
  const cache = await openCache();
  const loaded = urls.map(() => 0);
  const totals = urls.map(() => 0);
  let lastReport = -Infinity;
  const report = (force = false) => {
    const now = performance.now();
    if (!force && now - lastReport < REPORT_EVERY_MS) return;
    lastReport = now;
    const sum = (values) => values.reduce((a, b) => a + b, 0);
    const known = totals.every((size, index) => size > 0 && loaded[index] <= size);
    onProgress?.(sum(loaded), known ? sum(totals) : 0);
  };

  return Promise.all(
    urls.map(async (url, index) => {
      const cached = await readCached(cache, url);
      if (cached) {
        loaded[index] = totals[index] = cached.length;
        report(true);
        return cached;
      }
      const bytes = await download(
        url,
        (received) => {
          loaded[index] = received;
          report();
        },
        (size) => {
          totals[index] = size;
          report(true);
        },
      );
      totals[index] = loaded[index] = bytes.length;
      report(true);
      try {
        await cache?.put(
          url,
          new Response(bytes, { headers: { "content-type": "application/octet-stream" } }),
        );
      } catch {
        // A full or blocked quota only costs the next visit a download.
      }
      return bytes;
    }),
  );
}
