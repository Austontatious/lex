// frontend/src/services/api.ts
// ———————————————————————————————————————————————————————————
// Robust backend URL resolution (env + runtime + auto-discover) with
// self-healing failover, timeouts, and a unified API surface for App.tsx.
// ———————————————————————————————————————————————————————————

type AnyDict = Record<string, any>;

function mergeEnv(target: AnyDict, source: unknown) {
  if (!source || typeof source !== "object") return;
  Object.assign(target, source as AnyDict);
}

function readEnv(): AnyDict {
  const merged: AnyDict = {};
  const globalAny = globalThis as AnyDict;
  try { mergeEnv(merged, globalAny?.process?.env); } catch {}
  try { mergeEnv(merged, globalAny?.__ENV); } catch {}
  if (typeof window !== "undefined") {
    try { mergeEnv(merged, (window as AnyDict).__ENV); } catch {}
  }
  return merged;
}

const ENV = readEnv();

// Debug toggle (optional)
const DEBUG = ENV.VITE_API_DEBUG === "1" || ENV.REACT_APP_API_DEBUG === "1";

function log(...args: any[]) { if (DEBUG) console.log("[api]", ...args); }
function safeStr(x: any | undefined): string | undefined {
  return typeof x === "string" && x.trim() ? x.trim() : undefined;
}

// ==== Inputs: env, runtime globals, meta tag, persisted, heuristics ====
const envBase = safeStr(
  ENV.VITE_BACKEND ||
  ENV.VITE_BACKEND_URL ||
  ENV.VITE_API_URL ||
  ENV.REACT_APP_API_BASE ||
  ENV.REACT_APP_BACKEND
);

const runtimeGlobal =
  (typeof window !== "undefined" && (safeStr((window as any).__BACKEND__) || safeStr((window as any).__LEX_API_BASE))) ||
  undefined;

const metaTag =
  (typeof document !== "undefined" &&
    safeStr(document.querySelector<HTMLMetaElement>('meta[name="lexi-backend"]')?.content)) ||
  undefined;

const persisted =
  (typeof window !== "undefined" && safeStr(window.localStorage?.getItem("LEX_API_BASE") || undefined)) || undefined;

/** Heuristics from current page (LAN-friendly) */
function heuristicBases(): string[] {
  if (typeof window === "undefined") return [];
  const { protocol, hostname, port } = window.location;
  const hp = `${protocol}//${hostname}`;
  const host8000 = `${hp}:8000`;
  const sameOrigin = port ? `${hp}:${port}` : hp;
  const devMap = port === "3000" ? host8000 : undefined;
  return Array.from(new Set([devMap, host8000, sameOrigin].filter(Boolean))) as string[];
}

/** Drop localhost targets when the app itself isn’t on localhost */
function stripBadLocalhosts(cands: string[]): string[] {
  if (typeof window === "undefined") return cands;
  const h = window.location.hostname;
  const onLocal = h === "localhost" || h === "127.0.0.1";
  if (onLocal) return cands;
  return cands.filter((b) => !/^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/i.test(b));
}

/** Build ordered candidate list (strongest → weakest) */
function candidateBases(): string[] {
  const cands = [
    envBase,
    runtimeGlobal,
    metaTag,
    persisted,
    ...heuristicBases(),
    "http://localhost:8000",
    "http://127.0.0.1:8000",
  ].filter(Boolean) as string[];
  return stripBadLocalhosts(Array.from(new Set(cands)));
}

function joinUrl(base: string, path: string): string {
  if (!path.startsWith("/")) path = `/${path}`;
  return `${base.replace(/\/+$/, "")}${path}`;
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  ms: number,
  label: string
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  const externalSignal = init?.signal;
  let abortHandler: (() => void) | undefined;
  if (externalSignal) {
    if (externalSignal.aborted) {
      clearTimeout(timer);
      throw externalSignal.reason ?? Object.assign(new Error("Aborted"), { name: "AbortError" });
    }
    abortHandler = () => controller.abort();
    externalSignal.addEventListener("abort", abortHandler, { once: true });
  }

  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } catch (err: any) {
    if (err?.name === "AbortError") {
      throw Object.assign(new Error(`Timeout after ${ms}ms: ${label}`), { name: "TimeoutError" });
    }
    throw err;
  } finally {
    clearTimeout(timer);
    if (externalSignal && abortHandler) {
      externalSignal.removeEventListener("abort", abortHandler);
    }
  }
}

async function probe(base: string, ms = 1500): Promise<boolean> {
  try {
    const url = joinUrl(base, "/lexi/health");
    const res = await fetchWithTimeout(
      url,
      { method: "GET", cache: "no-store" },
      ms,
      `probe ${url}`
    );
    const ok = res.ok;
    log("probe:", base, ok ? "OK" : res.status);
    return ok;
  } catch (e) {
    log("probe failed:", base, (e as any)?.message || e);
    return false;
  }
}

// Live backend base: always use page origin when available
let _API_BASE: string = (typeof window !== "undefined" && window.location?.origin)
  ? window.location.origin
  : (candidateBases()[0] || "http://localhost:8000");

function publishBase(base: string) {
  _API_BASE = base;
  if (typeof window !== "undefined") {
    try {
      window.localStorage?.setItem("LEX_API_BASE", base);
      (window as any).__BACKEND__ = base;
      (window as any).__LEX_API_BASE = base;
    } catch {}
  }
  log("API_BASE set →", base);
}

async function resolveFirstHealthy(): Promise<string> {
  const cands = candidateBases();
  for (const c of cands) if (await probe(c)) return c;
  return cands[0]!;
}

/** Optional warmup: call once on app start (non-blocking) */
export async function initBackend() {
  const healthy = await resolveFirstHealthy();
  publishBase(healthy);
}
// Fire-and-forget warmup
void initBackend();

function API(): string { return _API_BASE; }

// Back-compat exports (value updated via publishBase)
export const API_BASE: string = ((): string => _API_BASE)() as unknown as string;
export const BACKEND = API_BASE;

// ==== Session header wiring ====
let _SESSION_ID: string | null = null;

export function setSessionId(id: string | null) {
  _SESSION_ID = id;
  if (typeof window !== "undefined") {
    try {
      if (id) {
        window.sessionStorage?.setItem("LEX_SESSION_ID", id);
      } else {
        window.sessionStorage?.removeItem("LEX_SESSION_ID");
      }
    } catch {
      // sessionStorage may be unavailable; ignore
    }
  }
}

export function getSessionId(): string | null {
  if (_SESSION_ID) return _SESSION_ID;
  if (typeof window !== "undefined") {
    try {
      const stored = window.sessionStorage?.getItem("LEX_SESSION_ID");
      if (stored) {
        _SESSION_ID = stored;
        return stored;
      }
    } catch {
      return _SESSION_ID;
    }
  }
  return _SESSION_ID;
}

// Resilient fetch: try alt bases on network failure once
type LexiRequestInit = RequestInit & { _path?: string; timeoutMs?: number };

async function resilientFetch(input: RequestInfo | URL, init?: LexiRequestInit): Promise<Response> {
  const { _path, timeoutMs, ...rest } = init ?? {};
  const rawUrl =
    typeof input === "string"
      ? input
      : input instanceof URL
      ? input.toString()
      : (input as Request).url;
  let derivedPath = _path;
  if (!derivedPath) {
    try {
      const u = new URL(rawUrl);
      derivedPath = (u.pathname || "/") + (u.search || "");
    } catch {
      derivedPath = rawUrl.startsWith("http") ? "/" : rawUrl || "/";
    }
  }

  const method = (rest.method || "GET").toUpperCase();
  const label = `${method} ${derivedPath}`;
  const ms = timeoutMs ?? 10000;

  const sessionHeader = getSessionId();
  const baseHeaders = (() => {
    const existing = rest.headers;
    if (!existing) return new Headers();
    if (existing instanceof Headers) return new Headers(existing);
    if (Array.isArray(existing)) return new Headers(existing);
    return new Headers(existing as Record<string, string>);
  })();
  if (sessionHeader) {
    baseHeaders.set("X-Lexi-Session", sessionHeader);
  }

  const attemptInit: RequestInit = {
    credentials: "same-origin",
    keepalive: true,
    ...rest,
    headers: baseHeaders,
  };

  try {
    return await fetchWithTimeout(input, attemptInit, ms, label);
  } catch (error: any) {
    const current = API();
    const candidates = candidateBases().filter((b) => b !== current);

    for (const cand of candidates) {
      log("rescue attempt via", cand, "for", derivedPath, "after", error?.name || error);
      const ok = await probe(cand, Math.min(ms, 1000));
      if (!ok) continue;
      publishBase(cand);
      const rescueUrl = joinUrl(cand, derivedPath);
      return fetchWithTimeout(rescueUrl, attemptInit, ms, label);
    }

    // Last-chance recovery: force a fresh discovery if every candidate failed.
    try {
      const fresh = await resolveFirstHealthy();
      if (fresh && fresh !== current) {
        publishBase(fresh);
        return fetchWithTimeout(joinUrl(fresh, derivedPath), attemptInit, ms, label);
      }
    } catch (reprobeError) {
      log("rehydrate after failure failed:", reprobeError);
    }

    throw error;
  }
}

async function jsonFetch<T>(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<T> {
  const base = API();
  const url = joinUrl(base, path);
  const timeoutMs = init?.timeoutMs ?? 10000;

  // Decide method + body presence
  const method = (init?.method || "GET").toUpperCase();
  const hasBody = !!(init && "body" in init && init.body != null);

  // Build headers without forcing Content-Type on GET
  const hdrs: Record<string, string> = {
    Accept: "application/json",
    ...(init?.headers as Record<string, string> | undefined),
  };
  if (hasBody && !hdrs["content-type"] && method !== "GET") {
    hdrs["content-type"] = "application/json";
  }

  const res = await resilientFetch(url, {
    ...init,
    method,
    headers: hdrs,
    _path: path,
    timeoutMs,
  });

  const ctype = res.headers.get("content-type") || "";
  if (!res.ok) {
    const body = ctype.includes("application/json")
      ? await res.json().catch(() => ({}))
      : await res.text().catch(() => "");
    throw new Error(
      `HTTP ${res.status} ${res.statusText} @ ${path}: ${typeof body === "string" ? body : JSON.stringify(body)}`
    );
  }
  if (ctype.includes("application/json")) return (await res.json()) as T;
  return ({ text: await res.text() } as unknown) as T;
}

// ——— Types ———
export type Persona = {
  mode?: string;
  traits?: Record<string, string>;
  image_path?: string;
  [k: string]: any;
};
export type TraitResponse = { narration?: string; ready?: boolean; prompt?: string; [k: string]: any };
export type ChatResponse = { cleaned?: string; raw?: string; text?: string; meta?: Record<string, unknown>; [k: string]: any };

// ——— API ———
export async function fetchPersona(): Promise<Persona> {
  return jsonFetch<Persona>("/lexi/persona", { method: "GET" });
}
export async function addTrait(text: string): Promise<TraitResponse> {
  return jsonFetch<TraitResponse>("/lexi/persona/add_trait", { method: "POST", body: JSON.stringify({ text }) });
}
export async function generateAvatar(
  body:
    | string
    | { prompt?: string; mode?: "txt2img" | "img2img"; changes?: string; denoise?: number; fresh_base?: boolean }
): Promise<{ image?: string; image_url?: string; path?: string; url?: string; file?: string; [k: string]: any }> {
  const payload = typeof body === "string" ? { prompt: body } : body;
  return jsonFetch("/lexi/persona/generate_avatar", {
    method: "POST",
    body: JSON.stringify(payload),
    // Comfy runs can take a while; align with backend poll window
    timeoutMs: 240000,
  });
}
export async function classifyIntent(text: string): Promise<{ intent: string; [k: string]: any }> {
  return jsonFetch<{ intent: string }>("/lexi/intent", { method: "POST", body: JSON.stringify({ text }) });
}
export async function sendPrompt(input: string | { prompt: string }): Promise<ChatResponse> {
  const prompt = typeof input === "string" ? input : input.prompt;
  return jsonFetch<ChatResponse>("/lexi/process", { method: "POST", body: JSON.stringify({ prompt }) });
}
export async function health(): Promise<string> {
  try {
    const r = await resilientFetch(joinUrl(API(), "/lexi/health"), { _path: "/lexi/health", timeoutMs: 5000 });
    return r.ok ? "ok" : `bad (${r.status})`;
  } catch (e: any) {
    return `err: ${e?.message || e}`;
  }
}

// ——— Alpha onboarding helpers ———
type SessionStartResponse = {
  session_id: string;
  consent: boolean;
  variant: string;
  alpha_strict: boolean;
};

export async function startAlphaSession(payload?: {
  consent?: boolean;
  userId?: string | null;
  variant?: string | null;
  tags?: string[];
}): Promise<SessionStartResponse> {
  const body = JSON.stringify({
    consent: payload?.consent ?? true,
    user_id: payload?.userId ?? null,
    variant: payload?.variant ?? null,
    tags: payload?.tags ?? ["alpha"],
  });
  const data = await jsonFetch<SessionStartResponse>("/lexi/alpha/session/start", { method: "POST", body });
  setSessionId(data.session_id);
  return data;
}

export async function updateAlphaConsent(consent: boolean): Promise<{ consent: boolean }> {
  const res = await jsonFetch<{ consent: boolean }>("/lexi/alpha/session/consent", {
    method: "POST",
    body: JSON.stringify({ consent }),
  });
  return res;
}

export async function endAlphaSession(): Promise<{ archived: boolean; archive_path?: string }> {
  const res = await jsonFetch<{ archived: boolean; archive_path?: string }>("/lexi/alpha/session/end", {
    method: "POST",
    body: "{}",
  });
  setSessionId(null);
  return res;
}

export async function fetchAlphaTourScript(): Promise<{ steps: Array<{ slug: string; prompt: string; narration: string }> }> {
  return jsonFetch("/lexi/alpha/tour/script", { method: "GET" });
}

export async function requestTourAvatarPreview(prompt: string): Promise<{ preview_url: string; alpha_strict: boolean }> {
  return jsonFetch("/lexi/alpha/tour/avatar-preview", {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export async function submitTourNowTopic(topic: string): Promise<{ now_topic: string }> {
  return jsonFetch("/lexi/alpha/tour/now-topic", {
    method: "POST",
    body: JSON.stringify({ topic }),
  });
}

export async function submitTourMemoryNote(note: string): Promise<{ ack: boolean }> {
  return jsonFetch("/lexi/alpha/tour/memory-note", {
    method: "POST",
    body: JSON.stringify({ note }),
  });
}

export async function sendTourFeedback(helpful: boolean, comment?: string): Promise<{ ok: boolean }> {
  return jsonFetch("/lexi/alpha/tour/feedback", {
    method: "POST",
    body: JSON.stringify({ helpful, comment }),
  });
}

export async function postAlphaMetric(event: string, detail?: Record<string, unknown>): Promise<{ ok: boolean }> {
  return jsonFetch("/lexi/alpha/session/metrics", {
    method: "POST",
    body: JSON.stringify({ event, detail }),
  });
}

export async function downloadSessionMemory(): Promise<{ blob: Blob; filename: string }> {
  const path = "/lexi/alpha/session/memory";
  const res = await resilientFetch(joinUrl(API(), path), { method: "GET", _path: path });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`download failed: ${res.status} ${res.statusText} ${text}`);
  }
  const disposition = res.headers.get("content-disposition") || "";
  const match = disposition.match(/filename\*=UTF-8''([^;]+)$|filename="?([^";]+)"?/i);
  const filename = decodeURIComponent(match?.[1] || match?.[2] || "session_memory.jsonl");
  const blob = await res.blob();
  return { blob, filename };
}
