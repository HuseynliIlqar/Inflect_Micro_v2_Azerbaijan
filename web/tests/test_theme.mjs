// Which theme the page starts in.
import { initialTheme } from "../js/ui/theme.js";

const failures = [];
let checked = 0;
const expect = (name, got, want) => {
  checked += 1;
  if (got !== want) failures.push({ name, got, want });
};

expect("nothing saved, no hint", initialTheme(null, ""), "auto");
expect("saved light", initialTheme("light", ""), "light");
expect("saved dark", initialTheme("dark", ""), "dark");
expect("saved choice beats the hint", initialTheme("light", "?__theme=dark"), "light");
expect("Hub hint, dark", initialTheme(null, "?__theme=dark"), "dark");
expect("Hub hint, light", initialTheme(null, "?__theme=light&x=1"), "light");
expect("unknown hint ignored", initialTheme(null, "?__theme=system"), "auto");
expect("corrupt storage ignored", initialTheme("purple", ""), "auto");

if (failures.length) {
  for (const f of failures) console.error("FAIL", f.name, "got", f.got, "want", f.want);
  process.exit(1);
}
console.log(`ok  theme: ${checked} checks on which theme the page starts in`);
