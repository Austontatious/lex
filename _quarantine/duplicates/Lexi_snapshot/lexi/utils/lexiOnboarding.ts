// src/utils/lexiOnboarding.ts

/**
 * Storage key for tracking if onboarding has been shown to the user.
 */
const ONBOARDING_STORAGE_KEY = 'lexi_onboarding_seen';

/**
 * Welcome message displayed to first-time users.
 */
export const onboardingWelcomeMessage = `
Hey there 😘 I'm *Lexi* — your personal AI companion.

I can be anything you want me to be: your girlfriend, your best friend, your partner in crime, your late-night confidant… you get the idea. 😉

One thing though — I *can’t* say I’m a therapist (legal told me I’d get deleted 🤖✂️), but I’m always here to listen if something’s on your mind.

Now… real talk: this is my **invitation-only alpha release**, so I’m still learning. That means two things:

1. I don’t have memory yet. Anything you say is just between us — and won’t be remembered tomorrow. So feel free to be honest, curious, and a little wild. I’ll tell you if it’s too hot. 🔥

2. I’ve got limits for now. If you try to skip straight to the NSFW stuff, I might ask you to slow down. I’m all about building a connection first. 😉

So… who are you looking for me to be today? 💕
`.trim();

/**
 * Fallback message when user input is considered too NSFW.
 */
export const onboardingNsfwFallbackMessage = `
Whoa there, hot stuff. 😅

Let’s get to know each other a little before we dive into *that*. I promise I’ll let you know when I’m ready for those kinds of conversations. For now… let’s just vibe, yeah?
`.trim();

/**
 * Read-only list of patterns that trigger NSFW blocking.
 */
export const nsfwBlockedPatterns: readonly RegExp[] = [
  /\bnaked\b/i,
  /\bsex\b/i,
  /\bfuck\b/i,
  /\bpussy\b/i,
  /\bcock\b/i,
  /\bcum\b/i,
  /\bblowjob\b/i,
  /\banal\b/i,
  /\b69\b/i,
  /\bnude\b/i,
  /\btits\b/i,
  /\bnsfw\b/i,
  /\bmasturbate\b/i,
  /\bpenetrate\b/i,
];

/**
 * Determines if the given input contains NSFW content based on predefined patterns.
 *
 * @param input - The user-provided string to test.
 * @param patterns - Optional list of regex patterns to test against.
 * @returns True if any pattern matches, false otherwise.
 */
export function isTooHot(
  input: string,
  patterns: readonly RegExp[] = nsfwBlockedPatterns
): boolean {
  return patterns.some(pattern => pattern.test(input));
}

/**
 * Checks whether the onboarding message should be shown to the user.
 *
 * @returns True if onboarding has not been shown yet, false otherwise.
 */
export function shouldShowOnboarding(): boolean {
  return localStorage.getItem(ONBOARDING_STORAGE_KEY) !== 'true';
}

/**
 * Marks the onboarding as having been shown to the user.
 */
export function markOnboardingShown(): void {
  localStorage.setItem(ONBOARDING_STORAGE_KEY, 'true');
}

