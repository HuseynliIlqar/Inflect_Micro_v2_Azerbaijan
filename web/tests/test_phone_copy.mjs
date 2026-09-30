// What a phone visitor reads must be plain: no engine names, no model jargon,
// and it must say how to get full speed and quality.
import { label, LABELS } from "../js/i18n.js";

const PHONE_KEYS = [
  "device_title_phone", "device_text_phone", "device_note_phone",
  "phone_note", "phone_why", "mode_fast_title_phone", "mode_fast_text_phone",
  "mode_full_text_phone", "result_ready", "result_ready_badge",
];
const JARGON = /WebGPU|WASM|WebAssembly|int8|fp32|CPU|GPU|backend|thread|axın|\?backend/iu;

const failures = [];
let checked = 0;
const expect = (name, ok) => {
  checked += 1;
  if (!ok) failures.push(name);
};

for (const lang of Object.keys(LABELS)) {
  for (const key of PHONE_KEYS) {
    const text = label(lang, key);
    expect(`[${lang}] ${key} exists`, text !== key && text.trim().length > 0);
    expect(`[${lang}] ${key} has no jargon: "${text}"`, !JARGON.test(text));
  }
  // The advice itself: a computer, and which browsers.
  for (const key of ["device_note_phone", "phone_note"]) {
    const text = label(lang, key);
    expect(`[${lang}] ${key} names what to do (a computer, Chrome or Edge)`, /Chrome/u.test(text) && /(kompüter|computer)/iu.test(text));
  }
  // The cause is the browser, not the phone: an app on the same phone runs fine.
  for (const key of ["device_text_phone", "phone_note"]) {
    expect(`[${lang}] ${key} puts it on the browser`, /(brauzer|browser)/iu.test(label(lang, key)));
  }
  expect(`[${lang}] device_text_phone says an app runs normally`, /(tətbiq|app)/iu.test(label(lang, "device_text_phone")));
  // Short enough that the popup fits a small phone without scrolling
  // (checked in a browser at 320x568; these limits keep it that way).
  const limits = { device_title_phone: 45, device_text_phone: 240, device_note_phone: 90 };
  for (const [key, max] of Object.entries(limits)) {
    const length = label(lang, key).length;
    expect(`[${lang}] ${key} is at most ${max} characters (it is ${length})`, length <= max);
  }
  for (const key of ["tech_wasm", "tech_threads", "tech_gpu", "tech_int8", "tech_download", "tech_screen"]) {
    const length = label(lang, key, { threads: 4 }).length;
    expect(`[${lang}] ${key} is at most 95 characters (it is ${length})`, length <= 95);
  }
  for (const key of ["tech_title", "tech_back"]) {
    expect(`[${lang}] ${key} exists`, label(lang, key) !== key);
  }
  // Where the audio will appear, for people who lose it on a phone.
  expect(`[${lang}] phone_note says where the result appears`, label(lang, "phone_note").includes(label(lang, "audio_label")));
}

if (failures.length) {
  console.log(`FAIL ${failures.length} of ${checked}`);
  for (const f of failures) console.log(`  ${f}`);
  process.exit(1);
}
console.log(`ok  phone-copy: ${checked} checks that phone texts are plain and helpful`);
