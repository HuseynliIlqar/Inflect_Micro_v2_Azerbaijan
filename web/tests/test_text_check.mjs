// What the playground accepts, and that every message exists in both languages.
import { checkText, cleanText, charCount, MAX_CHARS, ERROR_KEYS } from "../js/text/text-check.js";
import { label, LABELS } from "../js/ui/i18n.js";

let checks = 0;
function check(condition, message) {
  checks += 1;
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

// -- cleaning and counting -------------------------------------------------------

check(cleanText("  Salam.\n ") === "Salam.", "trims both ends");
check(cleanText("Sa\u0000la​m\u0007") === "Salam", "drops control and zero-width characters");
check(cleanText("bir\niki\tüç") === "bir\niki\tüç", "keeps newlines and tabs");
check(cleanText(undefined) === "", "no value is empty text");
check(charCount("Şəhər") === 5, "an Azerbaijani letter is one character");
check(charCount("ok 👍") === 4, "an emoji is one character, not two code units");

// -- errors ------------------------------------------------------------------------

check(checkText("").error === "empty", "empty");
check(checkText("   \n\t ").error === "empty", "whitespace only is empty");
check(checkText("​​").error === "empty", "invisible characters only are empty");
check(checkText("!!! ... ???").error === "nothing_to_say", "punctuation only has nothing to say");
check(checkText("🙂🙂").error === "nothing_to_say", "emoji only has nothing to say");
check(checkText("25%").error === null, "a number is speakable");
check(checkText("Salam.").error === null, "a sentence passes");

const atLimit = "a".repeat(MAX_CHARS);
check(checkText(atLimit).error === null, "exactly the limit passes");
check(checkText(`${atLimit}b`).error === "too_long", "one over the limit is too long");
check(checkText(`  ${atLimit}  `).error === null, "surrounding spaces do not count");
check(checkText("ə".repeat(MAX_CHARS)).error === null, "the limit counts letters, not bytes");
check(checkText("abc", { max: 2 }).error === "too_long", "the limit can be set per call");
const long = checkText(`${atLimit}xyz`);
check(long.count === MAX_CHARS + 3 && long.max === MAX_CHARS, "reports the count and the limit");

check(!checkText("Salam.").near, "short text is not near the limit");
check(checkText("a".repeat(Math.ceil(MAX_CHARS * 0.9))).near, "90% of the limit is near it");

// -- warnings ----------------------------------------------------------------------

check(checkText("Salam, necəsən?", { voice: "az" }).warnings.length === 0, "Azerbaijani in the Azerbaijani voice");
check(checkText("Salam, necəsən?", { voice: "en" }).warnings.includes("warn_az_in_en"), "ə in the English voice warns");
check(checkText("İstanbul, ağac", { voice: "en" }).warnings.includes("warn_az_in_en"), "İ and ğ in the English voice warn");
check(checkText("A café in Zürich.", { voice: "en" }).warnings.length === 0, "English loanword accents do not warn");
check(checkText("Привет", { voice: "az" }).warnings.includes("warn_script"), "Cyrillic warns");
check(checkText("مرحبا salam").warnings.includes("warn_script"), "Arabic script warns");
check(checkText("Привет").error === null, "a warning does not block");

// -- every message in both languages -----------------------------------------------

const keys = [...Object.values(ERROR_KEYS).flat(), "warn_script", "warn_az_in_en", "char_count", "char_count_label"];
for (const lang of Object.keys(LABELS)) {
  for (const key of keys) {
    const text = label(lang, key);
    check(text !== key && text.trim().length > 0, `[${lang}] ${key} exists`);
  }
  const tooLong = label(lang, "error_too_long", { count: 612, max: MAX_CHARS });
  check(tooLong.includes("612") && tooLong.includes(String(MAX_CHARS)), `[${lang}] error_too_long names the count and the limit`);
  check(!/\{\w+\}/u.test(tooLong), `[${lang}] error_too_long fills every placeholder`);
  const counter = label(lang, "char_count_label", { count: 12, max: MAX_CHARS });
  check(counter.includes("12") && !/\{\w+\}/u.test(counter), `[${lang}] char_count_label fills every placeholder`);
}
check(label("az", "error_too_long") !== label("en", "error_too_long"), "the two languages are not the same text");

console.log(`ok  text check: ${checks} checks on what the playground accepts`);
