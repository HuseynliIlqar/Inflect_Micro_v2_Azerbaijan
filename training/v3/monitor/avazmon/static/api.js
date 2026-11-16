// Token handling and API calls. The token comes once in the link (#token=...), is kept in this browser, and is
// sent as a Bearer header; audio is fetched the same way and played from a blob URL.

const KEY = "monitor-token";

function stored() {
  try { return localStorage.getItem(KEY) || ""; } catch { return ""; }
}

let token = new URLSearchParams(location.hash.slice(1)).get("token") || stored();
if (location.hash.includes("token=")) {
  try { localStorage.setItem(KEY, token); } catch { /* private mode: this page only */ }
  history.replaceState(null, "", location.pathname + location.search);
}

export function setToken(value) {
  token = value;
  try { localStorage.setItem(KEY, value); } catch { /* private mode */ }
}

export class AuthError extends Error {}

export async function api(path) {
  const response = await fetch(`api/${path}`, { headers: { Authorization: `Bearer ${token}` } });
  if (response.status === 401) throw new AuthError("Token yanlışdır.");
  const body = await response.json();
  if (!response.ok || !body.success) throw new Error(body.detail || body.error || `Xəta ${response.status}`);
  return body.data;
}

const audioCache = new Map();

export async function audioUrl(query) {
  if (!audioCache.has(query)) {
    const response = await fetch(`api/audio?${query}`, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new Error("Səs tapılmadı.");
    audioCache.set(query, URL.createObjectURL(await response.blob()));
  }
  return audioCache.get(query);
}
