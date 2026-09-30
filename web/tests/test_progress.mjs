// The progress bar's arithmetic, and the model download that feeds it.
import { downloadFraction, chunkFraction, megabytes } from "../js/ui/progress.js";
import { fetchModels, CACHE_NAME } from "../js/engine/fetch-model.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

// -- pure fractions ------------------------------------------------------------

expect("download, size unknown", downloadFraction(500, 0), null);
expect("download, halfway", downloadFraction(50, 100), 0.5);
expect("download, overshoot clamps", downloadFraction(120, 100), 1);
expect("one chunk is indeterminate", chunkFraction(0, 1), null);
expect("no chunks is indeterminate", chunkFraction(0, 0), null);
expect("second of four", chunkFraction(1, 4), 0.25);
expect("megabytes", megabytes(37_744_925), "37.7");

// -- fetchModels against a stand-in network and cache ----------------------------

const FILES = { "./a.onnx": new Uint8Array(300).fill(1), "./b.onnx": new Uint8Array(700).fill(2) };
let downloads = 0;

globalThis.fetch = async (url) => {
  downloads += 1;
  const bytes = FILES[url];
  if (!bytes) return new Response("missing", { status: 404 });
  // Two pieces, so progress is reported in between.
  const half = bytes.length >> 1;
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(bytes.slice(0, half));
      controller.enqueue(bytes.slice(half));
      controller.close();
    },
  });
  return new Response(body, { headers: { "content-length": String(bytes.length) } });
};

const stores = new Map();
globalThis.caches = {
  keys: async () => [...stores.keys()],
  delete: async (name) => stores.delete(name),
  open: async (name) => {
    if (!stores.has(name)) stores.set(name, new Map());
    const store = stores.get(name);
    return {
      match: async (url) => store.get(url)?.clone(),
      put: async (url, response) => {
        store.set(url, response);
      },
    };
  },
};
stores.set("aztts-onnx-v0", new Map()); // an outdated cache that must go

const reports = [];
const first = await fetchModels(["./a.onnx", "./b.onnx"], (loaded, total) =>
  reports.push([loaded, total]),
);
expect("bytes come back in order", [first[0].length, first[1][0]], [300, 2]);
expect("two downloads", downloads, 2);
expect("last report is complete", reports.at(-1), [1000, 1000]);
expect(
  "total is 0 until every size is known",
  reports.every(([, total]) => total === 0 || total === 1000),
  true,
);
expect("loaded never decreases", reports.every(([l], i) => i === 0 || l >= reports[i - 1][0]), true);
expect("old cache deleted", [...stores.keys()], [CACHE_NAME]);

const second = await fetchModels(["./a.onnx", "./b.onnx"]);
expect("second visit reads the cache", downloads, 2);
expect("cached bytes identical", [second[0].length, second[1].length, second[1][0]], [300, 700, 2]);

let message = "";
try {
  await fetchModels(["./missing.onnx"]);
} catch (error) {
  message = error.message;
}
expect("HTTP errors are reported", message.includes("HTTP 404"), true);

// A compressed response: the header gives fewer bytes than the stream reads.
FILES["./gz.onnx"] = new Uint8Array(900);
const realFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
  const response = await realFetch(url);
  return new Response(response.body, { headers: { "content-length": "400" } });
};
const lying = [];
await fetchModels(["./gz.onnx"], (loaded, total) => lying.push([loaded, total]));
globalThis.fetch = realFetch;
expect("a wrong size becomes unknown", lying.some(([l, t]) => t > 0 && l > t), false);
expect("the end is still exact", lying.at(-1), [900, 900]);

delete globalThis.caches;
const uncached = await fetchModels(["./a.onnx"]);
expect("works without Cache Storage", uncached[0].length, 300);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  progress: ${checked} checks on the bar and the model download`);
