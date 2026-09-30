/**
 * Where the progress bar stands, as pure functions of what the worker reports.
 *
 * `null` means indeterminate: the work is running but its size is unknown, so
 * the bar animates instead of pretending to a number.
 */

/** Bytes downloaded so far as a fraction, or null before the size is known. */
export function downloadFraction(loaded, total) {
  if (!(total > 0)) return null;
  return Math.min(1, Math.max(0, loaded / total));
}

/**
 * Chunks finished as a fraction. A single chunk has no midpoint to report --
 * the model does not say how far through a chunk it is -- so it is
 * indeterminate rather than stuck at zero.
 */
export function chunkFraction(done, total) {
  if (!(total > 1)) return null;
  return Math.min(1, Math.max(0, done / total));
}

/** Bytes as megabytes with one decimal, for "12.3 / 37.7 MB". */
export function megabytes(bytes) {
  return (bytes / 1e6).toFixed(1);
}
