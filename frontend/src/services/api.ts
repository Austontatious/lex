const PROD_FALLBACK = "https://api.lexicompanion.com/lexi";
const DEFAULT_FALLBACK = "/lexi";
const isProdHost =
  typeof window !== "undefined" && window.location.hostname.endsWith("lexicompanion.com");

let API_BASE = isProdHost ? PROD_FALLBACK : DEFAULT_FALLBACK;
let configLoaded = false;

export async function loadApiConfig() {
  if (configLoaded) return API_BASE;
  try {
    const r = await fetch("/config.json", { cache: "no-store" });
    if (!r.ok) {
      configLoaded = true;
      console.warn("[API] /config.json responded", r.status);
      return API_BASE;
    }
    const cfg = await r.json();
    if (cfg?.API_BASE && typeof cfg.API_BASE === "string") {
      const candidate = cfg.API_BASE.trim();
      if (candidate) {
        const normalized = candidate.replace(/\/+$/, "");
        const isLocalhost = /^https?:\/\/(localhost|127\.0\.0\.1)/i.test(normalized);
        if (isProdHost && isLocalhost) {
          console.warn(
            "[API] Ignoring localhost API_BASE from config on production host; keeping fallback"
          );
        } else {
          API_BASE = normalized;
        }
      }
    }
  } catch {
    // ignore, fallback remains
  }
  configLoaded = true;
  console.log("[API] Using API_BASE:", API_BASE);
  return API_BASE;
}

export async function apiFetch(path: string, init: RequestInit = {}) {
  await loadApiConfig();

  const headers = new Headers(init.headers || undefined);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
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
export function addTrait(trait: string): Promise<TraitResponse> {
  return jsonFetch<TraitResponse>(`${PERSONA_PREFIX}/add_trait`, {
    method: "POST",
    body: JSON.stringify({ trait }),
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
  return apiFetch(`/generate_avatar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  }).then((res) => res.json());
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
  sendPrompt,
  apiFetch,
};

export { API_BASE as BACKEND };
export default API;
