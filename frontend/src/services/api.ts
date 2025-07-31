// src/services/api.ts  
// ── backend origin autodetect ─────────────────────────────────────────
function resolveBackend(): string {
  const viteEnv =
    typeof import.meta !== "undefined" &&
    (import.meta as any).env?.VITE_BACKEND_URL;
  const craEnv = (process.env as any)?.REACT_APP_BACKEND_URL;
  const runtimeGlobal = (window as any).__LEX_BACKEND__;
  return viteEnv || craEnv || runtimeGlobal || "http://localhost:8000";
}
export const BACKEND = resolveBackend();

/** Simple JSON API wrapper with error handling */
async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BACKEND}${path}`, init);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<T>;
}

// ── Interfaces ─────────────────────────────────────────────────────────
export interface Persona {
  style: string[];
  traits: string[];
  certainty: number;
  image_path?: string;
}

export interface TraitResponse {
  avatar_url: string;
  traits: Record<string, string>;
  mode: string;
  ask?: string;
  ready?: boolean;
  narration?: string;
  prompt?: string;
}



export interface AvatarResponse {
  image: string;
}

// ── Chat / persona API calls ───────────────────────────────────────────

/**
 * Send a free‑form chat prompt.
 */
export async function sendPrompt(body: { prompt: string }) {
  const res = await fetch(`${BACKEND}/lex/process`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  let json: any = {};
  try {
    json = JSON.parse(text);
  } catch {
    return { cleaned: "[invalid JSON]" };
  }
  let cleaned = json.cleaned;
  if (!cleaned?.trim() || cleaned.trim() === "[no response]") {
    cleaned = json.choices?.[0]?.text?.trim() || "[no response]";
  }
  return { ...json, cleaned };
}

/**
 * Decide whether to go into chat or avatar flow.
 */
export async function classifyIntent(
  text: string
): Promise<{ intent: string }> {
  const res = await fetch(`${BACKEND}/lex/persona/intent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error("Intent API failed");
  return res.json();
}

/**
 * Load the current persona (traits + image path).
 */
export async function fetchPersona(): Promise<Persona | null> {
  try {
    return await api<Persona>("/lex/persona/get");
  } catch {
    return null;
  }
}

export async function avatarStep(payload: {
  traits: Record<string, string>,
  reply: string
}): Promise<TraitResponse> {
  return api<TraitResponse>("/lex/persona/avatar_step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function addTrait(trait: string): Promise<TraitResponse> {
  return api<TraitResponse>("/lex/persona/add_trait", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trait }),
  });
}

/**
 * Generate (or regenerate) the avatar image once all traits are set.
 */
export async function generateAvatar(
  traits: Record<string, string>
): Promise<AvatarResponse> {
  return api<AvatarResponse>("/lex/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ traits }),
  });
}



