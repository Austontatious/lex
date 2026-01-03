export type TourCard =
  | { kind: "intro"; title: string; body: string }
  | { kind: "demo"; title: string; body: string; prompt: string; ctaLabel: string; allowEdit?: boolean }
  | { kind: "feature"; title: string; body: string }
  | {
      kind: "cta";
      title: string;
      body: string;
      primaryLabel: string;
      primaryTo: string;
      secondaryLabel?: string;
      secondaryTo?: string;
    };

export const tourCards: TourCard[] = [
  {
    kind: "intro",
    title: "Meet Lexi",
    body:
      "Lexi is a personal AI companion - warm, thoughtful, and built for conversations that actually feel good.",
  },
  {
    kind: "demo",
    title: "A calm start",
    body: "Tap once and Lexi will introduce herself in a way that helps you relax.",
    ctaLabel: "Try it",
    prompt:
      "We're just meeting. How would you explain what it's like to talk with you, in a way that helps someone relax?",
  },
  {
    kind: "demo",
    title: "The part people don't say out loud",
    body: "Lexi is good at noticing the quiet stuff - what you mean, not just what you type.",
    ctaLabel: "Ask Lexi",
    prompt:
      "Before we talk about anything else - what do you think most people hope an AI like you can help with, but don't usually say out loud?",
  },
  {
    kind: "cta",
    title: "Ready?",
    body: "You can jump in now, or finish setup in one last step.",
    primaryLabel: "Continue",
    primaryTo: "/tour/legal",
    secondaryLabel: "Talk to Lexi",
    secondaryTo: "/chat",
  },
];
