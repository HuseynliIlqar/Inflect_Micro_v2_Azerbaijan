/**
 * The one worker the page talks to, shared by Speak and the device check: the
 * check leaves the engine loaded on its winner, so both must use the same one.
 */

import { t } from "./context.js";
import { createWorkerClient } from "./worker-client.js";

export const client = createWorkerClient({
  url: new URL("../worker.js", import.meta.url),
  workerError: () => new Error(t("error_worker")),
});
