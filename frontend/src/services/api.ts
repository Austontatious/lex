const SAME_ORIGIN_BASE = "/lexi";
const API_OVERRIDE_PARAM = "apiBase";
const SESSION_STORAGE_KEY = "LEXI_ALPHA_SESSION";
const USER_ID_STORAGE_KEY = "LEXI_USER_ID";
const USER_ID_ENABLED = (() => {
  try {
    let viteFlag = false;
    try {
      viteFlag = (import.meta as any)?.env?.VITE_USER_ID_ENABLED === "1";
    } catch {
      viteFlag = false;
    }
    const reactFlag = typeof process !== "undefined" && process.env.REACT_APP_USER_ID_ENABLED === "1";
    const viteFromProcess =
      typeof process !== "undefined" && process.env.VITE_USER_ID_ENABLED === "1";
    const runtimeFlag =
      typeof window !== "undefined" && (window as any).__LEX_USER_ID_ENABLED === true;
    return Boolean(viteFlag || reactFlag || viteFromProcess || runtimeFlag);
  } catch {
    return false;
  }
})();

function resolveApiBase(): string {
  if (typeof window === "undefined") {
    return SAME_ORIGIN_BASE;
  }
  try {
    const current = new URL(window.location.href);
    const override = current.searchParams.get(API_OVERRIDE_PARAM);
    if (override) {
      return override.replace(/\/+$/, "");
    }
  } catch {
    // ignore URL parsing issues, fall through to default
  }
  return SAME_ORIGIN_BASE;
}

let API_BASE = resolveApiBase();
let configLoaded = false;
let cachedSessionId: string | null = null;
let cachedUserId: string | null = null;

export function setSessionId(id: string | null) {
  cachedSessionId = id;
  if (typeof window !== "undefined") {
    try {
      if (id) {
        window.sessionStorage?.setItem(SESSION_STORAGE_KEY, id);
      } else {
        window.sessionStorage?.removeItem(SESSION_STORAGE_KEY);
      }
    } catch {
      // sessionStorage might be blocked; ignore
    }
  }
}

export function getSessionId(): string | null {
  if (cachedSessionId) {
    return cachedSessionId;
  }
  if (typeof window !== "undefined") {
    try {
      const stored = window.sessionStorage?.getItem(SESSION_STORAGE_KEY);
      if (stored) {
        cachedSessionId = stored;
        return stored;
      }
    } catch {
      // ignore
    }
  }
  return cachedSessionId;
}

export function setUserId(id: string | null) {
  cachedUserId = id;
  if (typeof window !== "undefined") {
    try {
      if (id) {
        window.localStorage?.setItem(USER_ID_STORAGE_KEY, id);
      } else {
        window.localStorage?.removeItem(USER_ID_STORAGE_KEY);
      }
    } catch {
      // ignore storage issues
    }
  }
}

export function getUserId(): string | null {
  if (cachedUserId) {
    return cachedUserId;
  }
  if (typeof window !== "undefined") {
    try {
      const stored = window.localStorage?.getItem(USER_ID_STORAGE_KEY);
      if (stored) {
        cachedUserId = stored;
        return stored;
      }
    } catch {
      // ignore
    }
  }
  return cachedUserId;
}

export async function loadApiConfig() {
  if (configLoaded) return API_BASE;
  API_BASE = resolveApiBase();
  configLoaded = true;
  console.info("[API] Using API_BASE:", API_BASE);
  return API_BASE;
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  await loadApiConfig();

  const headers = new Headers(init.headers || undefined);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const sessionHeader = getSessionId();
  if (sessionHeader && !headers.has("X-Lexi-Session")) {
    headers.set("X-Lexi-Session", sessionHeader);
  }
  const userHeader = getUserId();
  if ((USER_ID_ENABLED || userHeader) && userHeader && !headers.has("X-Lexi-User")) {
    headers.set("X-Lexi-User", userHeader);
  }

  const url = path.startsWith("http") ? path : `${API_BASE}${path}`;
  return fetch(url, { ...init, headers, credentials: "include" });
}

export const PERSONA_PREFIX = "/persona";

export type Traits = Record<string, string>;

export interface PersonaState {
  traits: Traits;
  certainty: number;
  image_path: string;
}

export interface IntentResult {
  intent: "avatar_flow" | "avatar_edit" | "new_look" | "describe_avatar" | "chat" | (string & {});
}

export interface TraitResponse {
  ready: boolean;
  narration?: string;
  prompt: string;
  negative?: string;
  persona: {
    traits: Record<string, string>;
    certainty: number;
    image_path: string;
  };
  added?: boolean;
}

export interface AvatarStepResponse {
  ready: boolean;
  persona: PersonaState;
  prompt: string;
  negative: string;
  narration: string;
}

export interface DebugTraitsResponse {
  file_traits: Traits;
  persona_traits: Traits;
}

export type AvatarGenMode = "txt2img" | "img2img";

export type LexiverseStyle = "off" | "soft" | "full" | "promo";

export interface AvatarGenRequestPayload {
  prompt: string;
  negative_prompt?: string;
  sd_mode: AvatarGenMode;
  lexiverse_style: LexiverseStyle;
  seed?: number | null;
  strength?: number;
  flux_pipeline?: string;
}

export interface AvatarGenResponse {
  avatar_url?: string;
  url?: string;
  image?: string;
  image_url?: string;
  filename?: string;
  prompt_id?: string;
  status?: string;
  code?: string;
  error?: string;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await apiFetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`[API] ${init?.method ?? "GET"} ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

/** POST /intent */
export function detectIntent(text: string): Promise<IntentResult> {
  return jsonFetch<IntentResult>(`/intent`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

/** Alias for detectIntent for old code */
export const classifyIntent = detectIntent;

/** POST /persona/add_trait */
export function addTrait(text: string): Promise<TraitResponse> {
  return jsonFetch<TraitResponse>(`${PERSONA_PREFIX}/add_trait`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

/** POST /persona/avatar_step */
export function avatarStep(input: { traits?: Traits; reply?: string }): Promise<AvatarStepResponse> {
  return jsonFetch<AvatarStepResponse>(`${PERSONA_PREFIX}/avatar_step`, {
    method: "POST",
    body: JSON.stringify(input ?? {}),
  });
}

/** Alias for avatarStep for old code expecting generateAvatar */
export async function generateAvatar(prompt: string) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // allow slow renders
  try {
    const res = await apiFetch(`/persona/generate_avatar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`generate_avatar failed (${res.status}): ${text}`);
    }
    return res.json();
  } finally {
    clearTimeout(timeout);
  }
}

/** POST /lexi/gen/avatar */
export async function requestAvatarGeneration(
  payload: AvatarGenRequestPayload
): Promise<AvatarGenResponse> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000);
  try {
    // API_BASE already includes "/lexi", so keep path relative.
    const res = await apiFetch(`/gen/avatar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        ...payload,
        seed: payload.seed ?? null,
        flux_pipeline: payload.flux_pipeline ?? "flux_v2",
        ...(payload.sd_mode === "img2img" ? { strength: payload.strength ?? 0.35 } : {}),
      }),
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`[API] gen/avatar failed (${res.status}): ${text}`);
    }
    return res.json() as Promise<AvatarGenResponse>;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchAvatarGenerationStatus(
  promptId: string
): Promise<AvatarGenResponse & { status: string }> {
  const res = await apiFetch(`/gen/avatar/status/${promptId}`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`[API] GET gen/avatar/status failed (${res.status}): ${text}`);
  }
  return res.json();
}

/** GET /persona */
export function getPersona(): Promise<PersonaState> {
  return jsonFetch<PersonaState>(PERSONA_PREFIX);
}

/** Alias for getPersona for old code expecting fetchPersona */
export const fetchPersona = getPersona;

/** GET /persona/debug/traits */
export function debugTraits(): Promise<DebugTraitsResponse> {
  return jsonFetch<DebugTraitsResponse>(`${PERSONA_PREFIX}/debug/traits`);
}

export interface SendPromptOptions {
  prompt: string;
  intent?: string;
  onChunk?: (delta: string) => void;
}

export async function sendPrompt({ prompt, intent, onChunk }: SendPromptOptions) {
  const body = intent ? { prompt, intent } : { prompt };
  const response = await apiFetch(`/process`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: onChunk ? "application/x-ndjson" : "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errText = await response.text().catch(() => "");
    throw new Error(
      `[API] POST /process failed (${response.status}): ${errText || response.statusText}`
    );
  }

  const reader = response.body?.getReader?.();
  if (!reader) {
    const payload = await response.json();
    if (payload?.cleaned && typeof payload.cleaned === "string") {
      onChunk?.(payload.cleaned);
    }
    return payload;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let finalPayload: Record<string, any> | null = null;
  let aggregated = "";

  const processLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let data: Record<string, any>;
    try {
      data = JSON.parse(trimmed);
    } catch {
      throw new Error("Malformed stream chunk");
    }
    if (typeof data.delta === "string") {
      aggregated += data.delta;
      onChunk?.(data.delta);
      return;
    }
    if (data.error) {
      throw new Error(typeof data.error === "string" ? data.error : "stream error");
    }
    if (data.done) {
      finalPayload = data;
      return;
    }
    finalPayload = data;
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 1);
      processLine(line);
    }
  }

  buffer += decoder.decode();
  if (buffer) {
    processLine(buffer);
  }

  if (!finalPayload) {
    finalPayload = {
      cleaned: aggregated,
      raw: aggregated,
      choices: aggregated ? [{ text: aggregated }] : [],
      mode: null,
    };
  }

  if (finalPayload.done) {
    const { done, ...rest } = finalPayload;
    return rest;
  }

  return finalPayload;
}

export type LexiEventPayload = {
  type: "system_onboarding";
  mode: "tour" | "skip";
  flags?: Record<string, any>;
};

export interface LexiEventResponse {
  ok: boolean;
  message?: { role?: string; content: string };
  fallback?: boolean;
  tool_used?: boolean;
  tools_contract?: Record<string, any>;
  legal_available?: boolean;
  legal_path?: string;
  skip_message?: string;
}

export function sendLexiEvent(payload: LexiEventPayload): Promise<LexiEventResponse> {
  return jsonFetch<LexiEventResponse>(`/onboarding/boot`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchTourLegal(): Promise<{ ok?: boolean; text: string }> {
  return jsonFetch<{ ok?: boolean; text: string }>(`/tour/legal`);
}

export type EntryMode = "new" | "returning";

export interface AccountBootstrapReq {
  identifier: string;
  entry_mode: EntryMode;
  attempt_count?: number;
}

export type AccountBootstrapStatus = "CREATED_NEW" | "FOUND_EXISTING" | "EXISTS_CONFLICT" | "NOT_FOUND";

export interface AccountBootstrapResp {
  status: AccountBootstrapStatus;
  user_id?: string;
  display_name?: string;
  has_seen_disclaimer?: boolean;
}

export async function apiAccountBootstrap(payload: AccountBootstrapReq): Promise<AccountBootstrapResp> {
  const res = await apiFetch(`/account/bootstrap`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error("Account bootstrap failed");
  }
  return res.json();
}

export async function apiDisclaimerPreload(entry_mode: EntryMode): Promise<void> {
  const res = await apiFetch(`/onboarding/disclaimer_preload`, {
    method: "POST",
    body: JSON.stringify({ entry_mode }),
  });
  if (!res.ok) {
    throw new Error("Disclaimer preload failed");
  }
}

export interface DisclaimerCachedResp {
  status: "OK" | "EMPTY" | "NO_SESSION";
  disclaimer?: string;
}

export async function apiDisclaimerCached(): Promise<DisclaimerCachedResp> {
  const res = await apiFetch(`/onboarding/disclaimer_cached`);
  if (!res.ok) {
    throw new Error("Disclaimer cached fetch failed");
  }
  return res.json();
}

export async function apiDisclaimerAck(user_id: string, accepted: boolean, version = "v1"): Promise<void> {
  const res = await apiFetch(`/account/disclaimer_ack`, {
    method: "POST",
    body: JSON.stringify({ user_id, accepted, version }),
  });
  if (!res.ok) {
    throw new Error("Disclaimer ack failed");
  }
}

export type AlphaTourStep = {
  slug: string;
  prompt: string;
  narration: string;
};

export interface AlphaTourScriptResponse {
  steps: AlphaTourStep[];
  onboarding?: Record<string, any>;
  alpha_strict?: boolean;
}

export interface AlphaSessionResponse {
  session_id: string;
  consent: boolean;
  variant?: string | null;
  alpha_strict: boolean;
}

export async function startAlphaSession(payload?: {
  consent?: boolean;
  userId?: string | null;
  variant?: string | null;
  tags?: string[];
}): Promise<AlphaSessionResponse> {
  const body = JSON.stringify({
    consent: payload?.consent ?? true,
    user_id: payload?.userId ?? null,
    variant: payload?.variant ?? null,
    tags: payload?.tags ?? ["alpha"],
  });
  const data = await jsonFetch<AlphaSessionResponse>(`/alpha/session/start`, {
    method: "POST",
    body,
  });
  setSessionId(data.session_id);
  if (payload?.userId) {
    setUserId(payload.userId);
  }
  return data;
}

export async function updateAlphaConsent(consent: boolean): Promise<{ consent: boolean }> {
  return jsonFetch<{ consent: boolean }>(`/alpha/session/consent`, {
    method: "POST",
    body: JSON.stringify({ consent }),
  });
}

export async function endAlphaSession(): Promise<{ archived: boolean; archive_path?: string }> {
  const res = await jsonFetch<{ archived: boolean; archive_path?: string }>(`/alpha/session/end`, {
    method: "POST",
    body: "{}",
  });
  setSessionId(null);
  return res;
}

export async function fetchAlphaTourScript(): Promise<AlphaTourScriptResponse> {
  return jsonFetch<AlphaTourScriptResponse>(`/alpha/tour/script`);
}

export async function requestTourAvatarPreview(
  prompt: string
): Promise<{ preview_url?: string; url?: string; alpha_strict?: boolean }> {
  return jsonFetch(`/alpha/tour/avatar-preview`, {
    method: "POST",
    body: JSON.stringify({ prompt }),
  });
}

export async function submitTourNowTopic(topic: string): Promise<{ now_topic: string }> {
  return jsonFetch(`/alpha/tour/now-topic`, {
    method: "POST",
    body: JSON.stringify({ topic }),
  });
}

export async function submitTourMemoryNote(note: string): Promise<{ ack: boolean }> {
  return jsonFetch(`/alpha/tour/memory-note`, {
    method: "POST",
    body: JSON.stringify({ note }),
  });
}

export async function postAlphaTourMemory(body: { note: string }): Promise<{ ok: boolean; ack?: boolean }> {
  return jsonFetch(`/alpha/tour/memory`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function sendTourFeedback(helpful: boolean, comment?: string): Promise<{ ok: boolean }> {
  return jsonFetch(`/alpha/tour/feedback`, {
    method: "POST",
    body: JSON.stringify({ helpful, comment }),
  });
}

export async function postAlphaMetric(
  event: string,
  detail?: Record<string, unknown>
): Promise<{ ok: boolean }> {
  return jsonFetch(`/alpha/session/metrics`, {
    method: "POST",
    body: JSON.stringify({ event, detail }),
  });
}

export async function downloadSessionMemory(): Promise<{ blob: Blob; filename: string }> {
  const res = await apiFetch(`/alpha/session/memory`, {
    method: "GET",
    headers: { Accept: "application/jsonl" },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`[API] GET /alpha/session/memory failed (${res.status}): ${text}`);
  }
  const disposition = res.headers.get("content-disposition") || "";
  const match = disposition.match(/filename\*=UTF-8''([^;]+)|filename="?([^";]+)"?/i);
  const filename = decodeURIComponent(match?.[1] || match?.[2] || "memory.jsonl");
  const blob = await res.blob();
  return { blob, filename };
}

export type Persona = PersonaState;

export const API = {
  detectIntent,
  classifyIntent,
  addTrait,
  avatarStep,
  generateAvatar,
  getPersona,
  fetchPersona,
  debugTraits,
  requestAvatarGeneration,
  fetchAvatarGenerationStatus,
  sendPrompt,
  apiFetch,
  startAlphaSession,
  updateAlphaConsent,
  endAlphaSession,
  fetchAlphaTourScript,
  requestTourAvatarPreview,
  submitTourNowTopic,
  submitTourMemoryNote,
  postAlphaTourMemory,
  sendTourFeedback,
  postAlphaMetric,
  downloadSessionMemory,
  sendLexiEvent,
  fetchTourLegal,
};

export { API_BASE as BACKEND };
export default API;
