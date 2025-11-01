// src/services/api.ts (fully patched to match old imports and lex_persona endpoints)

const runtimeApiBase = (typeof window !== "undefined" && (window as any)?.RUNTIME_CONFIG?.API_BASE) || undefined;

export const BACKEND =
  (typeof runtimeApiBase === "string" && runtimeApiBase.trim()) ||
  (window as any)?.__LEX_API_BASE ||
  (import.meta && import.meta.env && import.meta.env.VITE_BACKEND_URL) ||
  "http://localhost:8000";

export const PERSONA_PREFIX =
  (window as any)?.__LEX_PERSONA_PREFIX ||
  (import.meta && import.meta.env && import.meta.env.VITE_PERSONA_PREFIX) ||
  "/persona";

console.info("[API] Using BACKEND:", BACKEND, "PERSONA_PREFIX:", PERSONA_PREFIX);

export type Traits = Record<string, string>;

export interface PersonaState {
  traits: Traits;
  certainty: number;
  image_path: string;
}

export interface IntentResult {
  intent: "avatar_flow" | "describe_avatar" | "chat" | (string & {});
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

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`[API] ${init?.method ?? "GET"} ${url} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

/** POST /persona/intent */
export function detectIntent(text: string): Promise<IntentResult> {
  return jsonFetch<IntentResult>(`${BACKEND}${PERSONA_PREFIX}/intent`, {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}

/** Alias for detectIntent for old code */
export const classifyIntent = detectIntent;

/** POST /persona/add_trait */
export function addTrait(trait: string): Promise<TraitResponse> {
  return jsonFetch<TraitResponse>(`${BACKEND}${PERSONA_PREFIX}/add_trait`, {
    method: "POST",
    body: JSON.stringify({ trait }),
  });
}

/** POST /persona/avatar_step */
export function avatarStep(input: { traits?: Traits; reply?: string }): Promise<AvatarStepResponse> {
  return jsonFetch<AvatarStepResponse>(`${BACKEND}${PERSONA_PREFIX}/avatar_step`, {
    method: "POST",
    body: JSON.stringify(input ?? {}),
  });
}

/** Alias for avatarStep for old code expecting generateAvatar */
export async function generateAvatar(prompt: string) {
  return fetch(`${BACKEND}/generate_avatar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  }).then((res) => res.json());
}


/** GET /persona/get */
export function getPersona(): Promise<PersonaState> {
  return jsonFetch<PersonaState>(`${BACKEND}${PERSONA_PREFIX}/get`);
}

/** Alias for getPersona for old code expecting fetchPersona */
export const fetchPersona = getPersona;

/** GET /persona/debug/traits */
export function debugTraits(): Promise<DebugTraitsResponse> {
  return jsonFetch<DebugTraitsResponse>(`${BACKEND}${PERSONA_PREFIX}/debug/traits`);
}

export async function sendPrompt({ prompt }: { prompt: string }) {
  return fetch(`${BACKEND}/lex/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  }).then((res) => res.json());
}


// Old type name aliases
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
};

export default API;
