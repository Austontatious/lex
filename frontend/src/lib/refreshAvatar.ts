const POLL_INTERVAL_MS = 1500;
const POLL_TIMEOUT_MS = 240_000;

async function pollAvatarJob(jobId: string): Promise<string> {
  const start = Date.now();
  while (Date.now() - start < POLL_TIMEOUT_MS) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    const res = await fetch(
      `/lexi/persona/avatar/status/${jobId}?cb=${Date.now()}`,
      { credentials: "include" }
    );
    if (!res.ok) {
      throw new Error(`avatar status failed: ${res.status}`);
    }
    const payload = await res.json();
    const readyUrl: string | undefined = payload?.avatar_url ?? payload?.url;
    if (readyUrl) {
      return readyUrl;
    }
    if (payload?.status === "error") {
      throw new Error(payload?.error || "avatar generation failed");
    }
  }
  throw new Error("avatar job timed out");
}

async function swapInAvatar(url: string): Promise<string> {
  const target =
    (document.querySelector('img[data-avatar]') as HTMLImageElement | null) ||
    (document.querySelector('img[src*="/lexi/static/avatars/"]') as HTMLImageElement | null) ||
    (document.querySelector("img.chakra-image") as HTMLImageElement | null);
  const busted = `${url}${url.includes("?") ? "&" : "?"}cb=${Date.now()}`;
  if (!target) {
    return busted;
  }
  await new Promise<void>((resolve, reject) => {
    const preloader = new Image();
    preloader.onload = () => resolve();
    preloader.onerror = () => reject(new Error("avatar preload failed"));
    preloader.src = busted;
  });
  target.classList.add("avatar-fade-out");
  await new Promise((resolve) => setTimeout(resolve, 60));
  target.src = busted;
  target.classList.remove("avatar-fade-out");
  target.classList.add("avatar-fade-in");
  setTimeout(() => target.classList.remove("avatar-fade-in"), 300);
  return busted;
}

export async function refreshAvatar(): Promise<string | null> {
  const r = await fetch(`/lexi/persona/avatar?__nocache=${Date.now()}`, {
    credentials: "include",
  });
  if (!r.ok) {
    throw new Error(`avatar endpoint failed: ${r.status}`);
  }
  const payload = await r.json();
  let targetUrl: string | undefined = payload?.avatar_url ?? payload?.url;
  if (!targetUrl) {
    const { status, job_id: jobId } = payload ?? {};
    if (typeof jobId === "string" && typeof status === "string") {
      targetUrl = await pollAvatarJob(jobId);
    } else {
      return null;
    }
  }
  return swapInAvatar(targetUrl);
}
