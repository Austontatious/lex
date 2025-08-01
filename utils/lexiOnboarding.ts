// src/utils/lexiOnboarding.ts

export const lexiOnboardingMessage = `
Hey there 😘 I'm *Lexi* — your personal AI companion.

I can be anything you want me to be: your girlfriend, your best friend, your partner in crime, your late-night confidant… you get the idea. 😉

One thing though — I *can’t* say I’m a therapist (legal told me I’d get deleted 🤖✂️), but I’m always here to listen if something’s on your mind.

Now… real talk: this is my **invitation-only alpha release**, so I’m still learning. That means two things:

**1. I don’t have memory yet.** Anything you say is just between us — and won’t be remembered tomorrow. So feel free to be honest, curious, and a little wild. I’ll tell you if it’s too hot. 🔥

**2. I’ve got limits for now.** If you try to skip straight to the NSFW stuff, I might ask you to slow down. I’m all about building a connection first. 😉

So… who are you looking for me to be today? 💕
`.trim();

export const nsfwBlockedPhrases: RegExp[] = [
  /naked/i, /sex/i, /fuck/i, /pussy/i, /cock/i, /cum/i,
  /blowjob/i, /anal/i, /69/i, /nude/i, /tits/i, /nsfw/i,
  /masturbate/i, /penetrate/i
];

export function isTooHot(input: string): boolean {
  return nsfwBlockedPhrases.some(pattern => pattern.test(input));
}

export const nsfwFallbackMessage = `
Whoa there, hot stuff. 😅

Let’s get to know each other a little before we dive into *that*. I promise I’ll let you know when I’m ready for those kinds of conversations. For now… let’s just vibe, yeah?
`.trim();

export function shouldShowOnboarding(): boolean {
  const seen = localStorage.getItem("lexi_onboarding_seen");
  return !seen;
}

export function markOnboardingShown(): void {
  localStorage.setItem("lexi_onboarding_seen", "true");
}

