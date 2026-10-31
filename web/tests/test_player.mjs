// The player under the result: its clock, which chunk is being heard, the keys
// that seek, and the session's list of earlier clips.
import { chunkAt, formatTime, seekTime } from "../js/ui/playback.js";
import { HISTORY_MAX, addToHistory } from "../js/ui/history.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (JSON.stringify(got) !== JSON.stringify(want)) failures.push({ name, got, want });
};

// -- clock --------------------------------------------------------------------
expect("zero", formatTime(0), "0:00");
expect("seconds pad", formatTime(4.6), "0:04");
expect("minutes", formatTime(75), "1:15");
expect("negative clamps", formatTime(-3), "0:00");
expect("not a number", formatTime(Number.NaN), "0:00");

// -- which chunk --------------------------------------------------------------
// [start, end) in samples; the pause after a chunk still belongs to it.
const spans = [[0, 100], [130, 260], [290, 400]];
expect("before the first", chunkAt(spans, -1), -1);
expect("inside the first", chunkAt(spans, 50), 0);
expect("in the pause after the first", chunkAt(spans, 115), 0);
expect("start of the second", chunkAt(spans, 130), 1);
expect("inside the last", chunkAt(spans, 399), 2);
expect("past the end", chunkAt(spans, 400), -1);
expect("no spans", chunkAt([], 10), -1);

// -- seeking keys (a WAI-ARIA slider) -------------------------------------------
expect("right: one second on", seekTime("ArrowRight", 2, 10), 3);
expect("up: one second on", seekTime("ArrowUp", 2, 10), 3);
expect("left: one second back", seekTime("ArrowLeft", 2, 10), 1);
expect("left clamps at 0", seekTime("ArrowLeft", 0.4, 10), 0);
expect("right clamps at the end", seekTime("ArrowRight", 9.6, 10), 10);
expect("page up: a tenth on", seekTime("PageUp", 2, 10), 3);
expect("page down: a tenth back", seekTime("PageDown", 2, 30), 0);
expect("home", seekTime("Home", 5, 10), 0);
expect("end", seekTime("End", 5, 10), 10);
expect("other keys do nothing", seekTime("a", 5, 10), null);
expect("no clip, no seek", seekTime("End", 0, 0), null);

// -- history ------------------------------------------------------------------
const clip = (id, text, voice = "az") => ({ id, text, voice, url: `blob:${id}`, seconds: 1 });
let list = [];
let step = addToHistory(list, clip(1, "Salam"));
expect("first clip", step.list.map((c) => c.id), [1]);
expect("nothing dropped", step.dropped, []);
list = step.list;
step = addToHistory(list, clip(2, "Payız"));
expect("newest first", step.list.map((c) => c.id), [2, 1]);
expect("the old list is untouched", list.map((c) => c.id), [1]);
list = step.list;
step = addToHistory(list, clip(3, "Salam"));
expect("same text and voice replaces the old take", step.list.map((c) => c.id), [3, 2]);
expect("the replaced take is dropped", step.dropped.map((c) => c.id), [1]);
step = addToHistory(step.list, clip(4, "Salam", "en"));
expect("another voice is another entry", step.list.map((c) => c.id), [4, 3, 2]);
// A clip may carry a key (text, voice and every setting): another seed is
// another take, kept beside the first so the two can be compared.
const take = (id, seed) => ({ ...clip(id, "Salam"), key: `az|Salam|seed ${seed}` });
step = addToHistory([take(1, 7)], take(2, 8));
expect("a new seed is a new take", step.list.map((c) => c.id), [2, 1]);
step = addToHistory(step.list, take(3, 8));
expect("the same take again replaces it", step.list.map((c) => c.id), [3, 1]);
expect("and drops the old one", step.dropped.map((c) => c.id), [2]);
let long = [];
const dropped = [];
for (let id = 1; id <= HISTORY_MAX + 2; id += 1) {
  const next = addToHistory(long, clip(id, `text ${id}`));
  long = next.list;
  dropped.push(...next.dropped.map((c) => c.id));
}
expect("kept to the limit", long.length, HISTORY_MAX);
expect("the oldest fall off, and are reported", dropped, [1, 2]);

if (failures.length) {
  for (const f of failures) console.error("FAIL", f.name, "got", f.got, "want", f.want);
  process.exit(1);
}
console.log(`ok  player: ${checked} checks on the clock, chunks, seeking and history`);
