// The device dialog closes on a tap on the backdrop -- only a tap that both
// starts and ends outside it, so dragging out of a text selection does not.
import { outsideRect } from "../js/ui/device-view.js";

let checks = 0;
function check(condition, message) {
  checks += 1;
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

const box = { left: 20, top: 100, right: 340, bottom: 500 };
check(!outsideRect(box, 180, 300), "the middle of the dialog is inside");
check(!outsideRect(box, 20, 100), "its top-left corner is inside");
check(!outsideRect(box, 340, 500), "its bottom-right corner is inside");
check(outsideRect(box, 10, 300), "left of it is outside");
check(outsideRect(box, 350, 300), "right of it is outside");
check(outsideRect(box, 180, 60), "above it is outside");
check(outsideRect(box, 180, 560), "below it is outside");

console.log(`ok  dialog: ${checks} checks on what counts as a tap on the backdrop`);
