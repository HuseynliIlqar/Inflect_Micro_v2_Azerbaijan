/** Every element the page script touches, looked up once. */

const $ = (id) => document.getElementById(id);
const all = (selector) => [...document.querySelectorAll(selector)];

export const els = {
  // -- top bar
  themeToggle: $("theme-toggle"),
  // -- hero: the orb and the status line under it
  orb: $("orb"), orbBlob: $("orb-blob"), orbRingInner: $("orb-ring-inner"),
  orbRingOuter: $("orb-ring-outer"), orbProgress: $("orb-progress"),
  orbGlow: document.querySelector(".orb-glow"), orbHint: $("orb-hint"),
  status: $("status"), elapsed: $("elapsed"),
  // -- the composer
  text: $("text"), textCount: $("text-count"), textError: $("text-error"), textWarning: $("text-warning"),
  clearText: $("clear-text"), voiceInfo: $("voice-info"),
  preview: $("preview"), previewText: $("preview-text"),
  chips: all(".chip[data-panel]"),
  speed: $("speed"), speedValue: $("speed-value"),
  variation: $("variation"), variationValue: $("variation-value"),
  seed: $("seed"), seedValue: $("seed-value"), shuffle: $("shuffle"),
  normalise: $("normalise"), normaliseInfo: $("normalise-info"),
  maxWords: $("max-words"), maxWordsValue: $("max-words-value"),
  speak: $("speak"), speakLabel: $("speak-label"), speakKbd: $("speak-kbd"), cancel: $("cancel"),
  progress: $("progress"),
  slowNote: $("slow-note"),
  phoneNote: $("phone-note"), phoneNoteText: $("phone-note-text"), phoneWhy: $("phone-why"),
  examples: $("examples"),
  // -- the result
  result: $("result"), resultReady: $("result-ready"),
  modeBanner: $("mode-banner"), modeTitle: $("mode-title"), modeText: $("mode-text"),
  modeGpu: $("mode-gpu"), modeAction: $("mode-action"), modeBadge: $("mode-badge"),
  play: $("play"), wave: $("wave"), waveBox: document.querySelector(".wave"), waveEmpty: $("wave-empty"),
  time: $("time"), audio: $("audio"), chunks: $("chunks"),
  stats: $("stats"), regen: $("regen"), download: $("download"),
  copyDiagnostics: $("copy-diagnostics"), copyStatus: $("copy-status"), diagnostics: $("diagnostics"),
  history: $("history"), historyList: $("history-list"),
  // Every link to the About page; its target follows the interface language.
  aboutLinks: all("[data-about-link]"),
  // -- the device check: card, dialog, first-visit measurement
  deviceCard: $("device-card"), deviceMeter: document.querySelector("#device-card .meter"),
  deviceLabel: $("device-label"), deviceTitle: $("device-title"), deviceDetails: $("device-details"),
  dialog: $("device-dialog"), dialogMeter: document.querySelector("#device-dialog .meter"),
  dialogTitle: $("dialog-title"), dialogText: $("dialog-text"), dialogNote: $("dialog-note"),
  dialogDontShow: $("dialog-dont-show"), dialogDontShowLabel: $("dialog-dont-show-label"),
  dialogDirect: $("dialog-direct"), dialogOk: $("dialog-ok"),
  dialogMain: $("dialog-main"), dialogTech: $("dialog-tech"), dialogTechList: $("dialog-tech-list"),
  dialogTechOpen: $("dialog-tech-open"), dialogTechBack: $("dialog-tech-back"),
  dialogRemeasure: $("dialog-remeasure"),
  cal: $("calibration"), calMeasuring: $("cal-measuring"), calHeading: $("cal-heading"),
  calIntro: $("cal-intro"), calProgress: $("cal-progress"), calStage: $("cal-stage"),
  calSteps: $("cal-steps"), calResult: $("cal-result"), calResultTitle: $("cal-result-title"),
  calMeter: document.querySelector("#calibration .meter"), calSummary: $("cal-summary"),
  calRunsOn: $("cal-runs-on"), calWhy: $("cal-why"), calReasons: $("cal-reasons"),
  calNoteOnce: $("cal-note-once"), calError: $("cal-error"), calSkip: $("cal-skip"),
  calRetry: $("cal-retry"), calContinue: $("cal-continue"),
  toasts: $("toasts"),
};
