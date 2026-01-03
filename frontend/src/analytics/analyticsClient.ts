import { v4 as uuidv4 } from "uuid";
import { apiFetch } from "../services/api";

const STORAGE_KEY = "lexi_visitor_id";
const HEARTBEAT_INTERVAL_MS = 30_000;

let activeTimer: number | null = null;
let memoryVisitorId: string | null = null;

function generateVisitorId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return uuidv4();
}

function getOrCreateVisitorId(): string | null {
  if (typeof window === "undefined") return null;
  if (memoryVisitorId) return memoryVisitorId;
  try {
    const stored = window.localStorage?.getItem(STORAGE_KEY);
    if (stored) {
      memoryVisitorId = stored;
      return stored;
    }
  } catch {
    // ignore storage errors
  }
  const created = generateVisitorId();
  memoryVisitorId = created;
  try {
    window.localStorage?.setItem(STORAGE_KEY, created);
  } catch {
    // ignore storage errors
  }
  return created;
}

async function postAnalytics(event: "visit" | "heartbeat", visitorId: string) {
  try {
    const res = await apiFetch(`/analytics/${event}`, {
      method: "POST",
      body: JSON.stringify({ visitor_id: visitorId }),
      keepalive: true,
    });
    if (!res.ok) {
      return;
    }
  } catch {
    // swallow analytics errors
  }
}

export function startAnalyticsTracking(intervalMs = HEARTBEAT_INTERVAL_MS) {
  if (typeof window === "undefined") return () => {};
  if (activeTimer !== null) return () => {};
  const visitorId = getOrCreateVisitorId();
  if (!visitorId) return () => {};

  void postAnalytics("visit", visitorId);
  activeTimer = window.setInterval(() => {
    void postAnalytics("heartbeat", visitorId);
  }, intervalMs);

  return () => {
    if (activeTimer !== null) {
      window.clearInterval(activeTimer);
      activeTimer = null;
    }
  };
}
