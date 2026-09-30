// After a clip is ready, the page brings the result into view only when needed.
import { needsReveal } from "../js/ui/reveal.js";

let checks = 0;
function check(condition, message) {
  checks += 1;
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

const VIEWPORT = 800;
check(needsReveal({ top: 1400, bottom: 1700 }, VIEWPORT), "below the screen: scroll to it");
check(needsReveal({ top: 620, bottom: 900 }, VIEWPORT), "only its heading peeks in at the bottom: scroll");
check(!needsReveal({ top: 120, bottom: 500 }, VIEWPORT), "already in view: leave the page alone");
check(!needsReveal({ top: 0, bottom: 300 }, VIEWPORT), "at the very top: in view");
check(needsReveal({ top: -300, bottom: -10 }, VIEWPORT), "scrolled past it: bring it back");
check(needsReveal({ top: -40, bottom: 400 }, VIEWPORT), "its heading is cut off above: scroll");
check(!needsReveal({ top: 100, bottom: 400 }, 0), "no viewport (a test page): do nothing");

console.log(`ok  reveal: ${checks} checks on when to scroll to the result`);
