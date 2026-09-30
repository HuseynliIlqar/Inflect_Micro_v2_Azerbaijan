// Cancelling a synthesis: the engine stops at the next chunk and says why.
import { CancelledError, Engine } from "../js/engine/engine.js";

let checks = 0;
function check(condition, message) {
  checks += 1;
  if (!condition) {
    console.log(`FAIL ${message}`);
    process.exit(1);
  }
}

function standIn() {
  const spoken = [];
  const engine = new Engine({}, { phonemize: async (text) => text });
  engine.decode = {}; // loaded
  engine.runWithFallback = (run) => run();
  engine.speakPhonemes = async (phonemes) => {
    spoken.push(phonemes);
    return new Float32Array(10);
  };
  return { engine, spoken };
}

async function rejection(promise) {
  try {
    await promise;
    return null;
  } catch (error) {
    return error;
  }
}

const THREE = "Bir. İki. Üç.";

// Cancelled after the first chunk: the second and third are never spoken.
{
  const { engine, spoken } = standIn();
  const error = await rejection(engine.speak(THREE, { isCancelled: () => spoken.length >= 1 }));
  check(error instanceof CancelledError, `a CancelledError, got ${error}`);
  check(error.name === "CancelledError", "its name survives a postMessage");
  check(spoken.length === 1, `one chunk spoken, got ${spoken.length}`);
}

// Cancelled before anything ran: nothing is spoken at all.
{
  const { engine, spoken } = standIn();
  const error = await rejection(engine.speak(THREE, { isCancelled: () => true }));
  check(error instanceof CancelledError, "cancelled up front");
  check(spoken.length === 0, "nothing spoken");
}

// Inside a chunk cut by a bleep, the check runs between the pieces too.
{
  const { engine, spoken } = standIn();
  const error = await rejection(
    engine.speak("Sən qəhbəsən, bildin?", { isCancelled: () => spoken.length >= 1 }),
  );
  check(error instanceof CancelledError, "cancelled between the pieces of one chunk");
  check(spoken.length === 1, `only the piece before the bleep, got ${spoken}`);
}

// Never cancelled: the whole text is spoken, as before.
{
  const { engine, spoken } = standIn();
  const { chunks } = await engine.speak(THREE, { isCancelled: () => false });
  check(spoken.length === chunks.length && chunks.length === 3, "all three chunks");
}

console.log(`ok  cancel: ${checks} checks on stopping a synthesis`);
