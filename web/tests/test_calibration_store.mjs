// The stored device check: reused while nothing it depends on has changed.
import { forgetStored, readStored, writeStored, MAX_AGE_MS } from "../js/ui/calibration-store.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

function memory() {
  const items = new Map();
  return {
    getItem: (key) => (items.has(key) ? items.get(key) : null),
    setItem: (key, value) => items.set(key, String(value)),
    removeItem: (key) => items.delete(key),
  };
}
const refusing = {
  getItem() {
    throw new Error("SecurityError");
  },
  setItem() {
    throw new Error("QuotaExceededError");
  },
  removeItem() {
    throw new Error("SecurityError");
  },
};

const KEY = { version: "61730f86976c96b3", userAgent: "UA", threads: 4, adapter: "ok:qualcomm:adreno-600" };
const VERDICT = {
  version: KEY.version,
  measuredAt: 1_000,
  chosen: { backend: "wasm", precision: "int8" },
  rtf: 0.8,
  sentenceSeconds: 3.2,
  tier: "usable",
  gpu: "no-adapter",
  plan: [{ backend: "wasm", precision: "int8" }, { backend: "wasm", precision: "fp32" }],
  trials: [],
};

{
  const storage = memory();
  expect("nothing stored", readStored(storage, KEY, 2_000), null);
  writeStored(storage, KEY, VERDICT);
  expect("the same device, soon after", readStored(storage, KEY, 2_000), VERDICT);
  expect("a month later it is measured again", readStored(storage, KEY, 1_000 + MAX_AGE_MS + 1), null);
  for (const field of Object.keys(KEY)) {
    const changed = { ...KEY, [field]: field === "threads" ? 1 : "other" };
    expect(`a different ${field} is measured again`, readStored(storage, changed, 2_000), null);
  }
  forgetStored(storage);
  expect("forgotten", readStored(storage, KEY, 2_000), null);
}

{
  const storage = memory();
  storage.setItem("aztts-device-check", "{not json");
  expect("a broken entry is ignored", readStored(storage, KEY, 2_000), null);
  storage.setItem("aztts-device-check", JSON.stringify({ key: KEY, verdict: { measuredAt: 1 } }));
  expect("an entry without a plan is ignored", readStored(storage, KEY, 2_000), null);
  storage.setItem("aztts-device-check", JSON.stringify({ key: KEY, verdict: { ...VERDICT, measuredAt: 5_000 } }));
  expect("an entry from the future is ignored", readStored(storage, KEY, 2_000), null);
}

// An entry of an older shape, or tampered with, is measured again rather than trusted.
{
  const storage = memory();
  const stored = (verdict) => {
    storage.setItem("aztts-device-check", JSON.stringify({ key: KEY, verdict }));
    return readStored(storage, KEY, 2_000);
  };
  expect("the whole shape passes", stored(VERDICT), VERDICT);
  expect("no trials", stored({ ...VERDICT, trials: undefined }), null);
  expect("no tier", stored({ ...VERDICT, tier: "warp" }), null);
  expect("a chosen step that is not a step", stored({ ...VERDICT, chosen: "webgpu" }), null);
  expect("an empty plan", stored({ ...VERDICT, plan: [] }), null);
  expect("a speed that is not a number", stored({ ...VERDICT, rtf: "fast" }), null);
  expect("a verdict for other graphs", stored({ ...VERDICT, version: "0000000000000000" }), null);
  expect("nothing measured is a valid verdict", stored({ ...VERDICT, chosen: null, rtf: null, sentenceSeconds: null, tier: "failed" })?.tier, "failed");
}

expect("storage that throws reads as nothing", readStored(refusing, KEY, 2_000), null);
writeStored(refusing, KEY, VERDICT);
forgetStored(refusing);
expect("and never throws", true, true);
expect("no storage at all", readStored(undefined, KEY, 2_000), null);

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) {
    console.log(`  ${f.name}: got ${JSON.stringify(f.got)}, want ${JSON.stringify(f.want)}`);
  }
  process.exit(1);
}
console.log(`ok  calibration-store: ${checked} checks on the stored device check`);
