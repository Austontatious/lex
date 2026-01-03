const TOUR_VERSION = "v1";

type TourFlags = {
  completed: boolean;
  dontShowAgain: boolean;
  legalAck: boolean;
  version: string | null;
  completedAt: string | null;
};

const ANON_KEYS = {
  completed: "lexi.tour.completed",
  dontShowAgain: "lexi.tour.dont_show_again",
  legalAck: "lexi.tour.legal_ack",
  version: "lexi.tour.version",
  completedAt: "lexi.tour.completed_at",
};

const SCOPED_SUFFIX = {
  completed: "completed",
  dontShowAgain: "dont_show_again",
  legalAck: "legal_ack",
  version: "version",
  completedAt: "completed_at",
};

const scopedKey = (userId: string, suffix: keyof typeof SCOPED_SUFFIX) =>
  `lexi.tour.${userId}.${SCOPED_SUFFIX[suffix]}`;

const readFlag = (key: string): boolean => {
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(key) === "true";
  } catch {
    return false;
  }
};

const readValue = (key: string): string | null => {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

const writeFlag = (key: string, value: boolean) => {
  if (typeof window === "undefined") return;
  try {
    if (value) {
      window.localStorage.setItem(key, "true");
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    return;
  }
};

const writeValue = (key: string, value: string | null) => {
  if (typeof window === "undefined") return;
  try {
    if (value) {
      window.localStorage.setItem(key, value);
    } else {
      window.localStorage.removeItem(key);
    }
  } catch {
    return;
  }
};

export const getTourFlags = (userId?: string | null): TourFlags => {
  if (userId) {
    const completed = readFlag(scopedKey(userId, "completed"));
    const dontShowAgain = readFlag(scopedKey(userId, "dontShowAgain"));
    const legalAck = readFlag(scopedKey(userId, "legalAck"));
    const version = readValue(scopedKey(userId, "version"));
    const completedAt = readValue(scopedKey(userId, "completedAt"));
    if (!completed && readFlag(ANON_KEYS.completed)) {
      writeFlag(scopedKey(userId, "completed"), true);
    }
    if (!dontShowAgain && readFlag(ANON_KEYS.dontShowAgain)) {
      writeFlag(scopedKey(userId, "dontShowAgain"), true);
    }
    if (!legalAck && readFlag(ANON_KEYS.legalAck)) {
      writeFlag(scopedKey(userId, "legalAck"), true);
    }
    if (!version && readValue(ANON_KEYS.version)) {
      writeValue(scopedKey(userId, "version"), readValue(ANON_KEYS.version));
    }
    if (!completedAt && readValue(ANON_KEYS.completedAt)) {
      writeValue(scopedKey(userId, "completedAt"), readValue(ANON_KEYS.completedAt));
    }
    if (completed && !readValue(scopedKey(userId, "version"))) {
      writeValue(scopedKey(userId, "version"), TOUR_VERSION);
    }
    if (completed && !readValue(scopedKey(userId, "completedAt"))) {
      writeValue(scopedKey(userId, "completedAt"), new Date().toISOString());
    }
    return {
      completed: readFlag(scopedKey(userId, "completed")),
      dontShowAgain: readFlag(scopedKey(userId, "dontShowAgain")),
      legalAck: readFlag(scopedKey(userId, "legalAck")),
      version: readValue(scopedKey(userId, "version")),
      completedAt: readValue(scopedKey(userId, "completedAt")),
    };
  }

  const completed = readFlag(ANON_KEYS.completed);
  if (completed && !readValue(ANON_KEYS.version)) {
    writeValue(ANON_KEYS.version, TOUR_VERSION);
  }
  if (completed && !readValue(ANON_KEYS.completedAt)) {
    writeValue(ANON_KEYS.completedAt, new Date().toISOString());
  }
  return {
    completed: readFlag(ANON_KEYS.completed),
    dontShowAgain: readFlag(ANON_KEYS.dontShowAgain),
    legalAck: readFlag(ANON_KEYS.legalAck),
    version: readValue(ANON_KEYS.version),
    completedAt: readValue(ANON_KEYS.completedAt),
  };
};

export const shouldSkipSplash = (userId?: string | null) => {
  const flags = getTourFlags(userId);
  return flags.dontShowAgain && flags.version === TOUR_VERSION;
};

export const shouldSkipTourCards = (userId?: string | null) => {
  const flags = getTourFlags(userId);
  return flags.completed && flags.version === TOUR_VERSION;
};

export const markTourCompleted = (userId?: string | null) => {
  const completedKey = userId ? scopedKey(userId, "completed") : ANON_KEYS.completed;
  writeFlag(completedKey, true);
  writeValue(userId ? scopedKey(userId, "version") : ANON_KEYS.version, TOUR_VERSION);
  writeValue(userId ? scopedKey(userId, "completedAt") : ANON_KEYS.completedAt, new Date().toISOString());
};

export const markLegalAck = (userId?: string | null) => {
  writeFlag(userId ? scopedKey(userId, "legalAck") : ANON_KEYS.legalAck, true);
};

export const setDontShowAgain = (value: boolean, userId?: string | null) => {
  writeFlag(userId ? scopedKey(userId, "dontShowAgain") : ANON_KEYS.dontShowAgain, value);
};

export const syncTourFlags = (userId?: string | null) => {
  if (!userId) return;
  const anon = getTourFlags(null);
  if (anon.completed) {
    writeFlag(scopedKey(userId, "completed"), true);
  }
  if (anon.dontShowAgain) {
    writeFlag(scopedKey(userId, "dontShowAgain"), true);
  }
  if (anon.legalAck) {
    writeFlag(scopedKey(userId, "legalAck"), true);
  }
  if (anon.version) {
    writeValue(scopedKey(userId, "version"), anon.version);
  }
  if (anon.completedAt) {
    writeValue(scopedKey(userId, "completedAt"), anon.completedAt);
  }
};

const CHAT_PREFILL_KEY = "lexi.chat.prefill";
const CHAT_AUTOSTART_KEY = "lexi.chat.autostart";

export type ChatAutostartMode = "voice" | "direct";

export const setChatPrefill = (text: string) => {
  writeValue(CHAT_PREFILL_KEY, text);
};

export const consumeChatPrefill = () => {
  const value = readValue(CHAT_PREFILL_KEY);
  writeValue(CHAT_PREFILL_KEY, null);
  return value;
};

export const setChatAutostart = (mode: ChatAutostartMode) => {
  writeValue(CHAT_AUTOSTART_KEY, mode);
};

export const consumeChatAutostart = (): ChatAutostartMode | null => {
  const value = readValue(CHAT_AUTOSTART_KEY);
  writeValue(CHAT_AUTOSTART_KEY, null);
  if (value === "voice" || value === "direct") {
    return value;
  }
  return null;
};
