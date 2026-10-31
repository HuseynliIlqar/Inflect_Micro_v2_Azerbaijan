// What search engines and AI crawlers read: most of them run no JavaScript, so
// everything here must be in the static files. The FAQ's structured data must
// say exactly what the page shows -- Google ignores (and may penalise) markup
// that does not match the visible text.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { LABELS } from "../js/ui/i18n.js";

const WEB = new URL("../", import.meta.url);
const REPO = new URL("../../", import.meta.url);
const read = (path, base = WEB) => readFileSync(new URL(path, base), "utf-8");

// The Hub page ranks (huggingface.co's authority); the static origin is where
// the files live. The playground points its canonical at the Hub page.
const SPACE_URL = "https://huggingface.co/spaces/ilqarrrr/Inflect_Micro_v2_Azerbaijan";
const ORIGIN = "https://ilqarrrr-inflect-micro-v2-azerbaijan.static.hf.space";
const ABOUT = { az: "about.html", en: "about-en.html" };
const MAX_PAGE_LINES = 400; // CLAUDE.md: a file stays under ~400 lines

const failures = [];
let checked = 0;
const expect = (name, ok) => {
  checked += 1;
  if (!ok) failures.push(name);
};

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', "#39": "'", nbsp: " " };
const plain = (html) =>
  html
    .replace(/<[^>]+>/g, "")
    .replace(/&(amp|lt|gt|quot|#39|nbsp);/g, (_, name) => ENTITIES[name])
    .replace(/\s+/g, " ")
    .trim();

const attr = (html, pattern) => html.match(pattern)?.[1] ?? null;
const meta = (html, key) =>
  attr(html, new RegExp(`<meta\\s+(?:name|property)="${key}"\\s+content="([^"]*)"`, "u"));
const canonical = (html) => attr(html, /<link rel="canonical" href="([^"]+)"/u);
const jsonLd = (html) =>
  [...html.matchAll(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/gu)].map(
    ([, body]) => {
      try {
        return JSON.parse(body);
      } catch {
        return null;
      }
    },
  );

function checkHead(name, html, { lang, wantCanonical }) {
  expect(`${name}: <html lang="${lang}">`, new RegExp(`<html lang="${lang}"`, "u").test(html));
  const title = plain(attr(html, /<title>([^<]*)<\/title>/u) ?? "");
  expect(`${name}: a title of 20-70 characters (${title.length})`, title.length >= 20 && title.length <= 70);
  const description = meta(html, "description") ?? "";
  expect(`${name}: a description of 70-170 characters (${description.length})`,
    description.length >= 70 && description.length <= 170);
  expect(`${name}: canonical is ${wantCanonical}`, canonical(html) === wantCanonical);
  expect(`${name}: og:url matches the canonical`, meta(html, "og:url") === wantCanonical);
  for (const key of ["og:title", "og:description", "og:image", "og:type"]) {
    expect(`${name}: has ${key}`, Boolean(meta(html, key)));
  }
  expect(`${name}: every JSON-LD block parses`, jsonLd(html).every((block) => block !== null));
  // The saved theme must land before the first paint: a blocking classic
  // script, ahead of every stylesheet. defer, async or a module would flash.
  const boot = attr(html, /(<script[^>]*src="\.\/js\/page\/theme-boot\.js"[^>]*>)/u) ?? "";
  expect(`${name}: links the theme boot script`, boot.length > 0);
  expect(`${name}: the theme boot script blocks`, !/\b(defer|async|type="module")/u.test(boot));
  expect(`${name}: the theme boot script comes before the stylesheets`,
    html.indexOf(boot) < html.indexOf('rel="stylesheet"'));
  expect(`${name}: links the favicon file`, html.includes('href="./favicon.svg"'));
  // Every relative link and script must reach a file that ships.
  for (const [, path] of html.matchAll(/(?:href|src)="\.\/([^"#?]+)/gu)) {
    expect(`${name}: ./${path} exists`, existsSync(new URL(path, WEB)));
  }
}

// -- the playground ----------------------------------------------------------
const index = read("index.html");
checkHead("index.html", index, { lang: "az", wantCanonical: SPACE_URL });
expect("index.html: description is Azerbaijani", /Azərbaycan/u.test(meta(index, "description") ?? ""));
expect(`index.html: at most ${MAX_PAGE_LINES} lines`, index.split("\n").length <= MAX_PAGE_LINES);
// The script sets document.title from the label; the static one must agree.
expect("index.html: <title> is the Azerbaijani page_title label",
  plain(attr(index, /<title>([^<]*)<\/title>/u) ?? "") === LABELS.az.page_title);
expect("index.html: og:title is the page title", meta(index, "og:title") === LABELS.az.page_title);
const app = jsonLd(index).find((block) => block?.["@type"] === "WebApplication");
expect("index.html: WebApplication structured data", Boolean(app));
expect("index.html: the app's url is the Hub page", app?.url === SPACE_URL);
expect("index.html: the app is in Azerbaijani", [app?.inLanguage].flat().includes("az"));
expect("index.html: the app is free", app?.isAccessibleForFree === true && app?.offers?.price === "0");
expect("index.html: <noscript> points to the About page",
  /<noscript>[\s\S]*href="\.\/about\.html"[\s\S]*<\/noscript>/u.test(index));
// The static text is what the script writes on load, so nothing jumps.
const staticText = (id) =>
  plain(index.match(new RegExp(`<(\\w+)\\b[^>]*\\bid="${id}"[^>]*>([\\s\\S]*?)</\\1>`, "u"))?.[2] ?? "");
for (const [id, key] of [["subtitle", "subtitle"], ["offline-note", "offline_note"],
  ["footer", "footer"], ["about-link", "about_link"]]) {
  expect(`index.html: #${id} holds the Azerbaijani "${key}" label`, staticText(id) === LABELS.az[key]);
}
expect("index.html: the footer links the About page", /id="about-link"\s+href="\.\/about\.html"/u.test(index));
for (const language of Object.keys(LABELS)) {
  for (const key of ["about_link", "page_title"]) {
    expect(`[${language}] ${key} label exists`, Boolean(LABELS[language][key]));
  }
}

// -- the About pages ---------------------------------------------------------
function faqOf(html) {
  const section = attr(html, /<section id="faq"[^>]*>([\s\S]*?)<\/section>/u) ?? "";
  return [...section.matchAll(/<h3[^>]*>([\s\S]*?)<\/h3>\s*<p>([\s\S]*?)<\/p>/gu)].map(
    ([, question, answer]) => ({ question: plain(question), answer: plain(answer) }),
  );
}

const faqCounts = {};
for (const [lang, file] of Object.entries(ABOUT)) {
  const html = read(file);
  checkHead(file, html, { lang, wantCanonical: `${ORIGIN}/${file}` });
  expect(`${file}: at most ${MAX_PAGE_LINES} lines`, html.split("\n").length <= MAX_PAGE_LINES);
  for (const [other, otherFile] of Object.entries(ABOUT)) {
    expect(`${file}: hreflang ${other}`,
      html.includes(`<link rel="alternate" hreflang="${other}" href="${ORIGIN}/${otherFile}" />`));
  }
  expect(`${file}: hreflang x-default`,
    html.includes(`<link rel="alternate" hreflang="x-default" href="${ORIGIN}/${ABOUT.az}" />`));
  expect(`${file}: links back to the playground`, html.includes('href="./index.html"'));
  expect(`${file}: credits the training dataset`, html.includes("ughurabbasov/azerbaijani-tts-dataset"));
  expect(`${file}: names the GPL phonemiser`, /GPL-3\.0/u.test(html));

  const visible = faqOf(html);
  faqCounts[lang] = visible.length;
  expect(`${file}: at least six questions (${visible.length})`, visible.length >= 6);
  const faq = jsonLd(html).find((block) => block?.["@type"] === "FAQPage");
  const marked = (faq?.mainEntity ?? []).map((item) => ({
    question: item?.name, answer: item?.acceptedAnswer?.text,
  }));
  expect(`${file}: FAQPage has as many questions as the page shows`, marked.length === visible.length);
  visible.forEach((item, i) => {
    expect(`${file}: question ${i + 1} matches its markup`, marked[i]?.question === item.question);
    expect(`${file}: answer ${i + 1} matches its markup`, marked[i]?.answer === item.answer);
  });
}
expect("both About pages ask the same number of questions", faqCounts.az === faqCounts.en);

// -- robots, sitemap, llms.txt -----------------------------------------------
const robots = read("robots.txt");
expect("robots.txt names the sitemap", robots.includes(`Sitemap: ${ORIGIN}/sitemap.xml`));
expect("robots.txt blocks nothing site-wide", !/^Disallow:\s*\/\s*$/mu.test(robots));

const sitemap = read("sitemap.xml");
const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/gu)].map(([, loc]) => loc);
for (const file of Object.values(ABOUT)) {
  expect(`sitemap lists ${file}`, locs.includes(`${ORIGIN}/${file}`));
}
for (const loc of locs) {
  const file = loc.slice(ORIGIN.length + 1);
  expect(`sitemap: ${loc} is on the static origin`, loc.startsWith(`${ORIGIN}/`));
  expect(`sitemap: ${file} exists`, existsSync(new URL(file, WEB)));
  // A sitemap lists canonical URLs only; the playground's canonical is the Hub page.
  expect(`sitemap: ${file} is its own canonical`, canonical(read(file)) === loc);
}
for (const [, date] of sitemap.matchAll(/<lastmod>([^<]+)<\/lastmod>/gu)) {
  expect(`sitemap: lastmod ${date} is a date`, /^\d{4}-\d{2}-\d{2}$/u.test(date));
}

const llms = read("llms.txt");
expect("llms.txt starts with a title", llms.startsWith("# "));
expect("llms.txt has a summary line", /^> \S/mu.test(llms));
const llmsLinks = [...llms.matchAll(/\]\(([^)]+)\)/gu)].map(([, url]) => url);
expect("llms.txt links are absolute", llmsLinks.length > 0 && llmsLinks.every((url) => url.startsWith("https://")));
for (const url of [SPACE_URL, `${ORIGIN}/${ABOUT.en}`, `${ORIGIN}/${ABOUT.az}`]) {
  expect(`llms.txt links ${url}`, llmsLinks.includes(url));
}

// -- the Space's README: its YAML is the Hub page's title and tags -------------
const spaceReadme = read("packaging/huggingface/STATIC_SPACE_README.md", REPO);
const front = attr(spaceReadme, /^---\n([\s\S]*?)\n---/u) ?? "";
expect("Space README: sdk static, app_file index.html",
  /^sdk: static$/mu.test(front) && /^app_file: index\.html$/mu.test(front));
expect("Space README: the title names the task", /^title: .*Text-to-Speech/mu.test(front));
const short = attr(front, /^short_description: (.+)$/mu) ?? "";
expect(`Space README: short_description fits the Hub's 60 (${short.length})`, short.length > 0 && short.length <= 60);
for (const tag of ["text-to-speech", "azerbaijani"]) {
  expect(`Space README: tag ${tag}`, new RegExp(`^  - ${tag}$`, "mu").test(front));
}
// The headers that give the page its threads, exactly as web/README.md documents them.
const headers = attr(read("README.md"), /```yaml\n(custom_headers:[\s\S]*?)```/u) ?? "";
expect("web/README.md documents custom_headers", headers.length > 0);
for (const line of headers.split("\n").filter(Boolean)) {
  expect(`Space README: "${line.trim()}"`, front.split("\n").includes(line));
}

// -- the deploy recipe ships every top-level file ------------------------------
// The checks above read the local tree; the Space only gets what the recipe
// copies, so a page missing from it would 404 there.
const publishing = read("packaging/PUBLISHING.md", REPO);
for (const file of readdirSync(WEB).filter((name) => /\.(html|txt|xml|svg)$/u.test(name))) {
  expect(`PUBLISHING.md copies web/${file} to the Space`, publishing.includes(`../web/${file}`));
}
expect("PUBLISHING.md copies the Space README",
  publishing.includes("cp ../packaging/huggingface/STATIC_SPACE_README.md README.md"));

if (failures.length) {
  for (const f of failures) console.error("FAIL", f);
  console.error(`${failures.length} of ${checked} checks failed`);
  process.exit(1);
}
console.log(`ok  seo: ${checked} checks on what crawlers read`);
