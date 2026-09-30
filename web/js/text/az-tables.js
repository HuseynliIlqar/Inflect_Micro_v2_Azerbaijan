/**
 * The Azerbaijani language data behind `normalizeAz`: letter names, months,
 * abbreviations, units, currencies, acronyms. A port of the same tables in
 * `aztts/az_text.py`. These strings are data -- never translate them.
 */


export const LETTER_NAMES = {
  a: "a", b: "be", c: "ce", ç: "çe", d: "de", e: "e",
  ə: "ə", f: "fe", g: "ge", ğ: "ğe", h: "he", x: "xe",
  ı: "ı", i: "i", j: "je", k: "ka", q: "qe", l: "el",
  m: "em", n: "en", o: "o", ö: "ö", p: "pe", r: "er",
  s: "se", ş: "şe", t: "te", u: "u", ü: "ü", v: "ve",
  y: "ye", z: "ze",
};

export const MONTHS = [
  "yanvar", "fevral", "mart", "aprel", "may", "iyun",
  "iyul", "avqust", "sentyabr", "oktyabr", "noyabr", "dekabr",
];

// Written abbreviation -> spoken form. Applied before anything else so their
// full stops never look like sentence boundaries to the chunker.
export const ABBREVIATIONS = {
  "və s.": "və sairə",
  "və b.": "və başqaları",
  "və i.a.": "və iləaxır",
  "məs.": "məsələn",
  "e.ə.": "eramızdan əvvəl",
  "b.e.ə.": "bizim eradan əvvəl",
  "prof.": "professor",
  "dos.": "dosent",
  "akad.": "akademik",
  "dr.": "doktor",
  "küç.": "küçə",
  "şəh.": "şəhər",
  "r-nu": "rayonu",
  "mkr.": "mikrorayon",
  "səh.": "səhifə",
  "bax:": "bax,",
};

export const UNITS_TABLE = {
  kq: "kiloqram", q: "qram", t: "ton", mq: "milliqram",
  km: "kilometr", m: "metr", sm: "santimetr", mm: "millimetr",
  l: "litr", ml: "millilitr",
  san: "saniyə", dəq: "dəqiqə", st: "ədəd",
  "kv.m": "kvadrat metr", "kub.m": "kub metr", ha: "hektar",
  kvt: "kilovatt", vt: "vatt", hz: "herts",
  gb: "giqabayt", mb: "meqabayt", kb: "kilobayt", tb: "terabayt",
  "°c": "dərəcə selsi", "°f": "dərəcə farenheyt",
  "km/saat": "kilometr saatda", "km/s": "kilometr saniyədə",
  "m/s": "metr saniyədə",
  mln: "milyon", mlrd: "milyard", trln: "trilyon",
};

export const CURRENCIES = {
  azn: "manat", "₼": "manat",
  usd: "dollar", $: "dollar",
  eur: "avro", "€": "avro",
  gbp: "funt sterlinq", "£": "funt sterlinq",
  rub: "rubl", "₽": "rubl",
  try: "Türkiyə lirəsi", "₺": "Türkiyə lirəsi",
};

// The coin an amount's two decimal places count: "19,99 AZN" is said
// "on doqquz manat doxsan doqquz qəpik".
export const CURRENCY_SUBUNITS = {
  manat: "qəpik", dollar: "sent", avro: "sent", rubl: "qəpik",
  "funt sterlinq": "pens", "Türkiyə lirəsi": "quruş",
};

export const SYMBOLS = {
  "&": " və ", "№": " nömrə ", "§": " paraqraf ",
  "×": " vurulsun ", "÷": " bölünsün ", "±": " artı mənfi ",
  "≈": " təxminən ", "≤": " kiçik və ya bərabər ",
  "≥": " böyük və ya bərabər ", "=": " bərabərdir ",
  "°": " dərəcə ",
};

// Acronyms with a settled spoken form; anything else short and upper-case is
// spelled out letter by letter.
export const ACRONYMS = {
  AMEA: "a em e a",
  ADA: "ada",
  ASAN: "asan",
  NATO: "nato",
  UNESCO: "yunesko",
  UNICEF: "yunisef",
  COVID: "kovid",
  PIN: "pin",
  SIM: "sim",
};

// Nouns after which a lone I, V or X is a numeral: "V əsr", "X sinif". A lone
// letter is otherwise left alone -- it is far more often just a letter.
export const ROMAN_NOUNS = [
  "əsr", "sinif", "kurs", "hissə", "fəsil", "bölmə", "cild", "maddə",
  "dərəcə", "qrup", "rüb", "yarımil", "mərtəbə",
];

export const ORDINAL_SUFFIXES = new Set([
  "cı", "ci", "cu", "cü",
  "ncı", "nci", "ncu", "ncü",
  "ıncı", "inci", "uncu", "üncü",
]);
