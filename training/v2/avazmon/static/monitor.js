// The monitor page: progress and tiles every 30 s; charts, validation clips and listening samples every 2 minutes.
import { api, audioUrl, AuthError, setToken } from "./api.js?v=4";
import { lineChart } from "./charts.js?v=5";

const $ = (id) => document.getElementById(id);
const STAGES = {
  posterior_warmup: "İsinmə (posterior)",
  linguistic_adaptation: "Dil uyğunlaşması",
  decoder_polish: "Səs (decoder) öyrənməsi",
  disc_warmup: "Polis isinməsi (yalnız polis öyrənir)",
};
let run = null;
let listenData = { sentences: [], versions: [] };
let listenChoice = null; // the version picked in the list, null = the newest
const shown = {}; // what each section last drew: redrawing an unchanged section would stop a clip mid-play

/** True when ``data`` differs from what ``section`` shows now; remembers it. */
function changed(section, data) {
  const key = JSON.stringify(data);
  if (shown[section] === key) return false;
  shown[section] = key;
  return true;
}
let lastStep = 0;
let range = 0; // 0 = the whole run, else the last N steps

/** @param {number|null|undefined} seconds */
function duration(seconds) {
  if (seconds == null) return "—";
  const h = Math.floor(seconds / 3600), m = Math.round((seconds % 3600) / 60);
  return h ? `${h} saat ${m} dəq` : `${m} dəq`;
}

/** "bu gün 20:01", "sabah 09:30" or "02.11 14:00" in the viewer's own time zone. */
function finishTime(date) {
  const two = (n) => String(n).padStart(2, "0");
  const clock = `${two(date.getHours())}:${two(date.getMinutes())}`;
  const today = new Date();
  const days = Math.round((new Date(date).setHours(0, 0, 0, 0) - today.setHours(0, 0, 0, 0)) / 86_400_000);
  if (days === 0) return `bu gün ${clock}`;
  if (days === 1) return `sabah ${clock}`;
  return `${two(date.getDate())}.${two(date.getMonth() + 1)} ${clock}`;
}

const num = (v, digits = 3) => (v == null ? "—" : Number(v).toFixed(digits));

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
}

/** A play button that fetches the clip with the token on first use. */
function player(query, label) {
  const button = el("button", { type: "button", class: "play", text: `▶ ${label}` });
  button.addEventListener("click", async () => {
    button.disabled = true;
    try {
      const audio = el("audio", { controls: "", src: await audioUrl(query), "aria-label": label });
      button.replaceWith(audio);
      audio.play().catch(() => {});
    } catch (error) {
      button.disabled = false;
      button.textContent = `⚠ ${error.message}`;
    }
  });
  return button;
}

function showStatus(s) {
  run = s.run;
  lastStep = s.step;
  const select = $("run");
  select.hidden = s.runs.length < 2;
  select.replaceChildren(...s.runs.map((name) => el("option", { value: name, text: name })));
  select.value = s.run;
  $("alive").textContent = s.alive ? "İşləyir" : "Dayanıb";
  $("alive").className = `pill ${s.alive ? "ok" : "stopped"}`;
  $("step").textContent = s.step.toLocaleString("az");
  $("max").textContent = s.max_steps.toLocaleString("az");
  $("bar").setAttribute("aria-valuenow", String(Math.round(s.percent)));
  $("bar-fill").style.width = `${Math.min(100, s.percent)}%`;
  $("stage").textContent = STAGES[s.stage] || s.stage || "—";
  $("epoch").textContent = !s.epoch ? "—" : s.total_epochs
    ? `${s.epoch} / ${Math.round(s.total_epochs)} (1 epoch = ${s.steps_per_epoch.toLocaleString("az")} addım)`
    : `${s.epoch} (cəmi hələ hesablanır)`;
  $("eta").textContent = duration(s.eta_seconds);
  $("finish").textContent = s.eta_seconds == null ? "—" : finishTime(new Date((s.now + s.eta_seconds) * 1000));
  $("speed").textContent = s.seconds_per_step == null ? "—" : `${s.seconds_per_step.toFixed(2)} s/addım`;
  $("cost").textContent = `$${s.cost_so_far.toFixed(2)} ($${s.price_per_hour}/saat)`;
  $("balance").textContent = s.balance ? `$${Number(s.balance.balance).toFixed(2)}` : "—";
  const latest = s.latest || {};
  $("t-mel").textContent = num(latest.loss_mel);
  $("t-kl").textContent = num(latest.loss_kl);
  $("t-dur").textContent = num(latest.loss_duration);
  $("t-vram").textContent = latest.peak_reserved_gb == null ? "—" : `${latest.peak_reserved_gb.toFixed(1)} GB`;
  $("t-gpu").textContent = s.gpu ? `${s.gpu.utilization}%` : "—";
  $("t-temp").textContent = s.gpu ? `${s.gpu.temperature}°C` : "—";
  $("updated").textContent = `Yeniləndi: ${new Date().toLocaleTimeString("az")} · ${s.run}`;
}

const LOSS_CHARTS = [
  ["c-mel", { key: "loss_mel", label: "Mel itkisi", digits: 3 }],
  ["c-kl", { key: "loss_kl", label: "KL", digits: 3 }],
  ["c-dur", { key: "loss_duration", label: "Müddət itkisi", digits: 3 }],
  ["c-g", { key: "loss_g", label: "Generator itkisi", digits: 2 }],
  ["c-d", { key: "loss_d", label: "Diskriminator itkisi", digits: 3 }],
  ["c-speed", { key: "seconds_per_step", label: "Sürət", digits: 2, unit: " s", band: false }],
];

const POLICE_CHARTS = [
  ["c-dmpd", { key: "d_mpd", label: "Dövr polisi", digits: 3 }],
  ["c-dmrd", { key: "d_mrd", label: "Spektr polisi", digits: 3 }],
  ["c-dcqtd", { key: "d_cqtd", label: "Ton polisi", digits: 3 }],
  ["c-melms", { key: "loss_mel_ms", label: "Çoxdəqiqlikli mel", digits: 3 }],
  ["c-stft", { key: "loss_stft", label: "Xətti spektr", digits: 3 }],
];

function showHistory(points) {
  for (const [id, spec] of LOSS_CHARTS) lineChart($(id), points, spec);
  // Only the decoder run logs one loss per discriminator family.
  const police = points.some((p) => Number.isFinite(p.d_mpd));
  $("police").hidden = !police;
  if (police) for (const [id, spec] of POLICE_CHARTS) lineChart($(id), points, spec);
}

const clipQuery = (clip) => `run=${encodeURIComponent(run)}&file=${encodeURIComponent(clip.file)}`;
const stepTitle = (step) => `${step.toLocaleString("az")}-cü addım`;

/** A labelled player for one validation clip. */
const labelled = (label, file) => el("div", { class: "side" }, [
  el("span", { class: "side-label", text: label }),
  player(clipQuery({ file }), "Dinlə"),
]);

/** The validation sentence with the clips to compare against, then every validation step, newest first. */
function showValidation(data) {
  if (!changed("validation", data)) return;
  const { steps, text, before, target } = data;
  const box = $("validation");
  const reference = [
    ...(text ? [el("p", { class: "val-text", text: `Cümlə: “${text}”` })] : []),
    ...(before || target ? [el("div", { class: "players" }, [
      ...(before ? [labelled("Train-dən əvvəl", before)] : []),
      ...(target ? [labelled("Hədəf (datasetdəki səs)", target)] : []),
    ])] : []),
  ];
  if (!steps.length) {
    box.replaceChildren(...reference, el("p", { class: "muted", text: "İlk validasiya 2,500-cü addımda olacaq." }));
    return;
  }
  const [newest, ...older] = steps;
  box.replaceChildren(
    ...reference,
    el("div", { class: "val-step" }, [
      el("h3", { text: `${stepTitle(newest.step)} (ən yeni)` }),
      el("div", { class: "players" }, newest.clips.map((c) => player(clipQuery(c), "Dinlə"))),
    ]),
    ...(older.length ? [el("h3", { text: "Köhnə addımlar" }), el("div", { class: "val-list" }, older.map((step) =>
      el("div", { class: "val-row" }, [
        el("span", { class: "val-step-name", text: stepTitle(step.step) }),
        ...step.clips.map((c) => player(clipQuery(c), "Dinlə")),
      ])))] : []),
  );
}

const versionLabel = (v) => (v.name === "base" ? "Train-dən əvvəl" : stepTitle(v.step));
const listenQuery = (version, file) =>
  `run=${encodeURIComponent(run)}&listen=${encodeURIComponent(version)}&file=${encodeURIComponent(file)}`;

/** The measurement table: every version that has metrics.json, then the dataset (the target). */
function showMeasurements(data) {
  const table = $("measure-table");
  const rows = data.versions.filter((v) => v.metrics);
  table.hidden = !rows.length;
  if (!rows.length) return;
  const cells = (label, m) => el("tr", {}, [
    el("th", { scope: "row", text: label }),
    el("td", { text: num(m.h2_4k, 2) }),
    el("td", { text: m.ring_share == null ? "—" : `${Math.round(m.ring_share * 100)}%` }),
    el("td", { text: m.ring_hz == null ? "—" : `${(m.ring_hz / 1000).toFixed(1)} kHz` }),
  ]);
  table.tBodies[0].replaceChildren(
    ...rows.map((v) => cells(versionLabel(v), v.metrics)),
    ...(data.reference ? [cells("Hədəf (dataset)", data.reference)] : []),
  );
}

/** The fixed sentences: each with the old Avaz and the chosen checkpoint side by side. */
function showListen(data) {
  listenData = data;
  showMeasurements(data);
  const steps = data.versions.filter((v) => v.name !== "base");
  const base = data.versions.find((v) => v.name === "base");
  // Until a version is picked the newest one is shown, so a new checkpoint appears by itself.
  const picked = steps.some((v) => v.name === listenChoice) ? listenChoice : (steps[0]?.name ?? null);
  if (!changed("listen", { data, picked })) return;
  const select = $("listen-version");
  select.replaceChildren(...steps.map((v) => el("option", { value: v.name, text: versionLabel(v) })));
  select.disabled = !steps.length;
  if (picked) select.value = picked;
  const chosenVersion = steps.find((v) => v.name === picked);
  const box = $("listen");
  if (!data.sentences.length) {
    box.replaceChildren(el("li", { class: "muted", text: "Nümunələr ilk checkpoint-dən (5,000-ci addım) sonra çıxacaq." }));
    return;
  }
  box.replaceChildren(...data.sentences.map((text, index) => {
    const file = `${String(index).padStart(2, "0")}.wav`;
    const side = (version, label) => el("div", { class: "side" }, [
      el("span", { class: "side-label", text: label }),
      version && version.files.includes(file)
        ? player(listenQuery(version.name, file), "Dinlə")
        : el("span", { class: "muted", text: "hələ yoxdur" }),
    ]);
    return el("li", { class: "sample" }, [
      el("p", { text }),
      el("div", { class: "players" }, [side(base, "Train-dən əvvəl"), side(chosenVersion, chosenVersion ? versionLabel(chosenVersion) : "Yeni")]),
    ]);
  }));
}

async function guarded(task) {
  try {
    await task();
    $("token-form").hidden = true;
  } catch (error) {
    if (error instanceof AuthError) $("token-form").hidden = false;
    else $("updated").textContent = `Xəta: ${error.message}`;
  }
}

const runQuery = () => (run ? `run=${encodeURIComponent(run)}` : "");
const refreshFast = () => guarded(async () => showStatus(await api(`status?${runQuery()}`)));
const historyQuery = () => `history?points=300&from_step=${range ? Math.max(0, lastStep - range) : 0}&${runQuery()}`;
const refreshCharts = () => guarded(async () => showHistory(await api(historyQuery())));
const refreshSlow = () => guarded(async () => {
  showHistory(await api(historyQuery()));
  showValidation(await api(`validation?limit=1000&${runQuery()}`));
  showListen(await api(`listen?${runQuery()}`));
});

$("token-form").addEventListener("submit", (event) => {
  event.preventDefault();
  setToken($("token").value.trim());
  refreshFast().then(refreshSlow);
});
for (const button of document.querySelectorAll("[data-range]")) {
  button.addEventListener("click", () => {
    range = Number(button.dataset.range);
    for (const other of document.querySelectorAll("[data-range]")) {
      other.setAttribute("aria-pressed", String(other === button));
    }
    refreshCharts();
  });
}
$("listen-version").addEventListener("change", (event) => {
  // Picking the newest again means "follow the newest" from now on.
  const newest = listenData.versions.find((v) => v.name !== "base");
  listenChoice = newest && event.target.value === newest.name ? null : event.target.value;
  showListen(listenData);
});
$("run").addEventListener("change", (event) => {
  run = event.target.value;
  refreshFast().then(refreshSlow);
});

refreshFast().then(refreshSlow);
setInterval(refreshFast, 30_000);
setInterval(refreshSlow, 120_000);
