// Which theme the page starts in, and where the one toggle button takes it.
import { initialTheme, toggledTheme } from "../js/ui/theme.js";

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

// The toggle always lands on the opposite of what is on screen.
expect("auto on a dark device -> light", toggledTheme("auto", true), "light");
expect("auto on a light device -> dark", toggledTheme("auto", false), "dark");
expect("dark -> light", toggledTheme("dark", false), "light");
expect("light -> dark", toggledTheme("light", true), "dark");
expect("unknown counts as auto", toggledTheme("purple", true), "light");

if (failures.length) {
  for (const f of failures) console.error("FAIL", f.name, "got", f.got, "want", f.want);
  process.exit(1);
}
console.log(`ok  theme: ${checked} checks on which theme the page starts in`);
