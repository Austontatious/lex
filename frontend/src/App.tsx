import React, {
  useState,
  useRef,
  useEffect,
  KeyboardEvent,
  useCallback,
} from "react";
import { v4 as uuidv4 } from "uuid";
import {
  Box,
  Button,
  Flex,
  HStack,
  Heading,
  IconButton,
  Image,
  Input,
  Spacer,
  Spinner,
  Text,
  Textarea,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  ModalCloseButton,
  VStack,
  useColorMode,
  useColorModeValue,
  useToast,
} from "@chakra-ui/react";
import { MoonIcon, SunIcon } from "@chakra-ui/icons";

import {
  fetchPersona,
  addTrait,
  generateAvatar,
  sendPrompt,
  sendLexiEvent,
  loadApiConfig,
  classifyIntent,
  startAlphaSession,
  endAlphaSession,
  downloadSessionMemory,
  setUserId,
  fetchTourLegal,
  apiAccountBootstrap,
  apiDisclaimerPreload,
  apiDisclaimerCached,
  apiDisclaimerAck,
} from "./services/api";
import type { TraitResponse, EntryMode, AccountBootstrapResp } from "./services/api";
import type { AlphaWelcomeCopy } from "./components/onboarding/AlphaWelcome";
import { FeedbackButton } from "./components/FeedbackButton";
import { AvatarToolsModal } from "./components/avatar/AvatarToolsModal";
import "./styles/chat.css";
import "./styles/avatar.css";
import { refreshAvatar } from "./lib/refreshAvatar";
import {
  consumeChatAutostart,
  consumeChatPrefill,
  getTourFlags,
  syncTourFlags,
} from "./tour/tour_storage";

interface ChatMessage {
  id: string;
  sender: "user" | "ai" | "system";
  content: string;
  streaming?: boolean;
  error?: boolean;
}

type PrefillMessage = Pick<ChatMessage, "sender" | "content">;

interface ChatShellProps {
  prefillMessages?: PrefillMessage[];
  onPrefillConsumed?: () => void;
  onDownloadSession?: () => void;
  waitingForLegalChoice?: boolean;
  onLegalYes?: () => Promise<void> | void;
  onLegalNo?: (continueChat?: boolean) => Promise<void> | void;
}

const DEFAULT_STATIC_AVATAR_BASE = "https://api.lexicompanion.com/lexi/static/avatars/";
const DEFAULT_AVATAR_URL = `${DEFAULT_STATIC_AVATAR_BASE}default.png`;

function resolveAvatarUrl(
  imagePath: string | null | undefined,
  fallbackBase = DEFAULT_STATIC_AVATAR_BASE
) {
  if (!imagePath) return `${fallbackBase}default.png`;
  if (imagePath.startsWith("http")) return imagePath;

  const normalized = imagePath.replace(/^\/+/, "");
  if (normalized.startsWith("lexi/static/")) {
    return `https://api.lexicompanion.com/${normalized}`;
  }
  if (normalized.startsWith("static/")) {
    return `https://api.lexicompanion.com/${normalized}`;
  }

  const suffix = imagePath.replace(/^.*avatars\//, "");
  const fileName = suffix || normalized;
  return `${fallbackBase}${fileName || "default.png"}`;
}

function mkId() {
  return `m_${uuidv4()}`;
}

function ChatShell({
  prefillMessages,
  onPrefillConsumed,
  onDownloadSession,
  waitingForLegalChoice,
  onLegalYes,
  onLegalNo,
}: ChatShellProps) {
  const { colorMode, toggleColorMode } = useColorMode();
  const bg = useColorModeValue("gray.50", "gray.800");
  const userBubbleBg = useColorModeValue("#ffd6ec", "pink.500");
  const aiBubbleBg = useColorModeValue("#ffffff", "rgba(255, 255, 255, 0.08)");
  const systemBubbleBg = useColorModeValue("#ffeaf4", "purple.600");
  const userBubbleColor = useColorModeValue("#2f001a", "white");
  const aiBubbleColor = useColorModeValue("#1f1626", "white");
  const systemBubbleColor = useColorModeValue("#2f001a", "white");
  const bubbleBorder = useColorModeValue(
    "1px solid rgba(255, 47, 160, 0.35)",
    "1px solid rgba(255, 105, 180, 0.4)"
  );
  const bubbleShadow = useColorModeValue(
    "0 10px 24px rgba(255, 47, 160, 0.18)",
    "0 0 16px rgba(255, 105, 180, 0.8)"
  );
  const composerBg = useColorModeValue("#fff0fa", "pink.500");
  const composerText = useColorModeValue("#2f001a", "white");
  const composerPlaceholder = useColorModeValue("#8d2950", "pink.200");
  const composerFieldBg = useColorModeValue("#ffffff", "rgba(255, 255, 255, 0.08)");
  const composerFieldBorder = useColorModeValue("1px solid #ff72d0", "1px solid hotpink");
  const composerShadow = useColorModeValue(
    "0 8px 18px rgba(255, 47, 160, 0.25)",
    "0 0 10px rgba(255, 105, 180, 0.6)"
  );

  const loadingVideoSrc = "/media/avatar-loading.mp4";
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [staticAvatarBase, setStaticAvatarBase] = useState(DEFAULT_STATIC_AVATAR_BASE);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(DEFAULT_AVATAR_URL);
  const [avatarFlow, setAvatarFlow] = useState(false);
  const [avatarGenerating, setAvatarGenerating] = useState(false);
  const [loadingVideoErrored, setLoadingVideoErrored] = useState(false);
  const [avatarToolsOpen, setAvatarToolsOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const triggerAvatarRefresh = useCallback(async () => {
    try {
      const fresh = await refreshAvatar();
      if (fresh) {
        setAvatarUrl(fresh);
      }
    } catch (err) {
      console.warn("avatar refresh failed:", err);
    }
  }, [setAvatarUrl]);

  const handleAvatarUpdated = useCallback(
    (url: string) => {
      const resolved = resolveAvatarUrl(url, staticAvatarBase);
      const withoutCacheParams = resolved
        .replace(/([?&])(v|cb)=[^&]+/gi, "")
        .replace(/[?&]+$/, "");
      const finalUrl = `${withoutCacheParams}${
        withoutCacheParams.includes("?") ? "&" : "?"
      }v=${Date.now()}`;
      setAvatarUrl(finalUrl);
      void triggerAvatarRefresh();
    },
    [staticAvatarBase, triggerAvatarRefresh]
  );

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!prefillMessages || prefillMessages.length === 0) {
      return;
    }
    setMessages((prev) => [
      ...prev,
      ...prefillMessages.map((msg) => ({ id: mkId(), ...msg })),
    ]);
    onPrefillConsumed?.();
  }, [prefillMessages, onPrefillConsumed]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      let nextStaticBase = DEFAULT_STATIC_AVATAR_BASE;
      const apiBase = await loadApiConfig();
      try {
        const parsed = new URL(apiBase, window.location.origin);
        nextStaticBase = `${parsed.origin.replace(/\/$/, "")}/lexi/static/avatars/`;
      } catch {
        nextStaticBase = DEFAULT_STATIC_AVATAR_BASE;
      }
      if (mounted) {
        setStaticAvatarBase(nextStaticBase);
      }
      const data = await fetchPersona();
      if (!mounted || !data) return;
      setAvatarUrl(resolveAvatarUrl((data as any).image_path, nextStaticBase));
    })();
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    void triggerAvatarRefresh();
  }, [triggerAvatarRefresh]);

  const appendMessage = useCallback(
    (msg: Omit<ChatMessage, "id"> & { id?: string }) => {
    setMessages((prev) => [...prev, { id: msg.id ?? mkId(), ...msg }]);
    },
    []
  );

// 🧠 INTELLIGENT TRAIT GATHERING LOOP

// Replace the old `handleTraitFlow` in App.tsx with this version:
const handleTraitFlow = useCallback(
  async (userText: string) => {
    try {
      const res: TraitResponse = await addTrait(userText);

      if (res.narration && !res.ready) {
	appendMessage({ sender: "ai", content: res.narration });
	setAvatarFlow(true);
	return;
      }


      if (res.narration) {
	appendMessage({ sender: "ai", content: res.narration });
      }

      if (!res.ready) {
        setAvatarFlow(true);
	return;
      }


      // ✅ Generate avatar, display it immediately (with cache bust)
      setAvatarFlow(false);
      setLoadingVideoErrored(false);
      setAvatarGenerating(true);
      try {
        const gen = await generateAvatar(res.prompt);
        const avatarCandidate =
          (gen as any)?.avatar_url ??
          (gen as any)?.url ??
          (gen as any)?.image ??
          (gen as any)?.image_url ??
          (gen as any)?.path ??
          "";
        if (typeof avatarCandidate === "string" && avatarCandidate) {
          const imgUrl = resolveAvatarUrl(avatarCandidate, staticAvatarBase);
          const finalUrl = `${imgUrl}${imgUrl.includes("?") ? "&" : "?"}v=${Date.now()}`;
          setAvatarUrl(finalUrl);
          await triggerAvatarRefresh();
          appendMessage({ sender: "ai", content: "📸 Here's your avatar!" });
        } else {
          appendMessage({ sender: "ai", content: "[Avatar update failed]" });
        }
      } finally {
        setAvatarGenerating(false);
      }

    } catch (err) {
      console.error(err);
      appendMessage({ sender: "ai", content: "[Avatar update failed]" });
    }
  },
  [appendMessage, staticAvatarBase, triggerAvatarRefresh]
);

const handleAvatarEdit = useCallback(
  async (userText: string) => {
    setLoading(true);
    setLoadingVideoErrored(false);
    setAvatarGenerating(true);
    try {
      const result = await sendPrompt({ prompt: userText, intent: "avatar_edit" });
      let reply =
        typeof result === "object" && result !== null
          ? typeof (result as any).cleaned === "string"
            ? (result as any).cleaned
            : typeof (result as any).message === "string"
              ? (result as any).message
              : "Avatar updates are handled in the Avatar Tools modal."
          : "Avatar updates are handled in the Avatar Tools modal.";
      const avatarCandidate =
        (result as any)?.avatar_url ??
        (result as any)?.url ??
        (result as any)?.image ??
        (result as any)?.image_url ??
        "";
      if (typeof avatarCandidate === "string" && avatarCandidate) {
        const resolved = resolveAvatarUrl(avatarCandidate, staticAvatarBase);
        const finalUrl = `${resolved}${resolved.includes("?") ? "&" : "?"}v=${Date.now()}`;
        setAvatarUrl(finalUrl);
        await triggerAvatarRefresh();
      }
      appendMessage({ sender: "ai", content: reply });
    } catch (err) {
      console.error(err);
      appendMessage({ sender: "ai", content: "[Avatar edit failed]", error: true });
    } finally {
      setAvatarFlow(false);
      setLoading(false);
      setAvatarGenerating(false);
    }
  },
  [appendMessage, staticAvatarBase, triggerAvatarRefresh]
);


  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    inputRef.current?.focus();
    appendMessage({ sender: "user", content: text });

    if (waitingForLegalChoice) {
      const trimmed = text.trim().toLowerCase();
      if (trimmed === "yes" || trimmed === "y") {
        await onLegalYes?.();
        return;
      }
      if (trimmed === "no" || trimmed === "n") {
        await onLegalNo?.(false);
        return;
      }
      await onLegalNo?.(true);
    }

    // --- 1) If we're already mid-flow, keep going ---
    if (avatarFlow) {
      handleTraitFlow(text);
      return;
    }

    // --- 2) Otherwise ask the backend "chat or avatar_flow?" ---
    let intent = "chat";
    try {
      ({ intent } = await classifyIntent(text));
      console.log("🕵️ Intent is", intent);
    } catch (e) {
      console.warn("Intent API failed, defaulting to chat", e);
    }

    if (intent === "avatar_flow" || intent === "new_look") {
      setAvatarFlow(true);
      handleTraitFlow(text);
      return;
    }

    if (intent === "avatar_edit") {
      await handleAvatarEdit(text);
      return;
    }

    if (intent === "describe_avatar") {
      const personaData = await fetchPersona();
      if (!personaData || !personaData.traits) {
        appendMessage({ sender: "ai", content: "I'm not sure how I look right now..." });
        return;
      }
      const traits = personaData.traits;

      const desc = Object.entries(traits)
        .map(([k, v]) => `${k.replace("_", " ")}: ${v}`)
        .join(", ");
      appendMessage({ sender: "ai", content: `Here's how I look right now: ${desc}.` });
      return;
    }

    // --- 3) Fallback to your normal LLM chat ---
    const aiId = mkId();
    appendMessage({ id: aiId, sender: "ai", content: "", streaming: true });
    setLoading(true);
    let streamed = "";
    try {
      const res = await sendPrompt({
        prompt: text,
        onChunk(delta) {
          streamed += delta;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aiId ? { ...m, content: streamed, streaming: true } : m
            )
          );
        },
      });
      const finalText =
        (typeof res?.cleaned === "string" && res.cleaned.trim()) ||
        streamed.trim() ||
        "[no response]";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId
            ? {
                ...m,
                content: finalText,
                streaming: false,
              }
            : m
        )
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Streaming failed";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === aiId
            ? {
                ...m,
                content: `[error] ${message}`,
                streaming: false,
                error: true,
              }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  }, [
    input,
    loading,
    avatarFlow,
    appendMessage,
    handleTraitFlow,
    handleAvatarEdit,
    waitingForLegalChoice,
    onLegalYes,
    onLegalNo,
  ]);

//import { useEffect } from "react";
//import {
//  lexiOnboardingMessage,
//  nsfwFallbackMessage,
//  isTooHot,
//  shouldShowOnboarding,
//  markOnboardingShown
//} from "../../../utils/lexiOnboarding";

//function App() {
//  useEffect(() => {
//    if (shouldShowOnboarding()) {
      // Push Lexi's message into chat stream
//      addAssistantMessage(lexiOnboardingMessage); // your method here
//    markOnboardingShown();
//    }
//  }, []);
 
// return <ChatUI />;
//}


  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <>
      <Flex
        className="appShell"
        direction="column"
        minH="var(--app-dvh)"
        bg={bg}
        fontFamily="'Nunito', sans-serif"
      >
        <HStack
          className="fixedBar"
          p={3}
          bg="pink.500"
          color="white"
          boxShadow="0 2px 12px rgba(255, 105, 180, 0.6)"
          data-tour="header"
        >
          <Box fontWeight="bold" fontSize="xl">
            Lexi Chat
          </Box>
          <Spacer />
          {onDownloadSession && (
            <Button size="lg" variant="outline" color="white" onClick={onDownloadSession}>
              Download session
            </Button>
          )}
          <FeedbackButton />
          <IconButton
            aria-label="Toggle dark mode"
            icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
            onClick={toggleColorMode}
            variant="ghost"
            color="white"
            size="lg"
            minW={11}
            minH={11}
          />
        </HStack>

        <Box flex="1" px={4} pt={4} pb={2} minH="0">
          <Box className="chat-layout with-avatar" h="100%" minH="0" w="100%">
            <Box as="aside" className="avatar-pane" data-tour="avatar">
              {avatarUrl && (
                <Box
                  className="avatar-card"
                  borderRadius="2xl"
                  border="2px solid hotpink"
                  overflow="hidden"
                  animation="pulseGlow 2s infinite"
                  boxShadow="0 0 24px rgba(255, 64, 128, 0.25)"
                  sx={{
                    "@keyframes pulseGlow": {
                      "0%": { boxShadow: "0 0 8px rgba(255, 105, 180, 0.3)" },
                      "50%": { boxShadow: "0 0 18px rgba(255, 105, 180, 0.8)" },
                      "100%": { boxShadow: "0 0 8px rgba(255, 105, 180, 0.3)" },
                    },
                  }}
                >
                  <Box position="relative">
                    {avatarGenerating && !loadingVideoErrored ? (
                      <Box
                        as="video"
                        className="avatar-loading-video"
                        src={loadingVideoSrc}
                        autoPlay
                        muted
                        loop
                        playsInline
                        preload="none"
                        poster={avatarUrl ?? undefined}
                        aria-label="Avatar generating animation"
                        onError={() => setLoadingVideoErrored(true)}
                      />
                    ) : (
                      <Image
                        data-avatar
                        src={avatarUrl}
                        alt="Lex avatar"
                        w="100%"
                        loading="lazy"
                        decoding="async"
                        display="block"
                        objectFit="cover"
                        aspectRatio="3 / 4"
                        sizes="(max-width: 900px) 100vw, 280px"
                        backdropFilter="blur(4px)"
                      />
                    )}
                    {avatarGenerating && (
                      <Flex
                        className="avatar-loading-overlay"
                        align="flex-end"
                        justify="space-between"
                        px={3}
                        py={2}
                        gap={3}
                      >
                        <Text fontWeight="bold">Generating your new look…</Text>
                        <Spinner size="sm" thickness="3px" color="pink.200" />
                      </Flex>
                    )}
                  </Box>
                </Box>
              )}
              <Button
                mt={3}
                variant="outline"
                colorScheme="pink"
                size="lg"
                onClick={() => setAvatarToolsOpen(true)}
                data-tour="gallery"
              >
                Open avatar tools
              </Button>
            </Box>

            <Box
              as="main"
              className="chat-pane"
              display="flex"
              flexDirection="column"
              h="100%"
              minH="0"
            >
              <Box
                className="chat-scroll"
                flex="1"
                overflowY="auto"
                minH="0"
                pt={2}
                pb={2}
                px={2}
                css={{
                  scrollbarColor: "hotpink transparent",
                  scrollbarWidth: "thin",
                }}
              >
                <VStack spacing={4} align="stretch">
                  {messages.map((m) => {
                    if (typeof m.content !== "string" || m.content.trim().length === 0) return null;

                    const isUser = m.sender === "user";
                    const isAi = m.sender === "ai";
                    const bubbleBg = isUser ? userBubbleBg : isAi ? aiBubbleBg : systemBubbleBg;
                    const bubbleColor = isUser
                      ? userBubbleColor
                      : isAi
                      ? aiBubbleColor
                      : systemBubbleColor;

                    return (
                      <Box
                        key={m.id}
                        className="bubble"
                        alignSelf={
                          m.sender === "user"
                            ? "flex-end"
                            : m.sender === "ai"
                            ? "flex-start"
                            : "center"
                        }
                        bg={bubbleBg}
                        color={bubbleColor}
                        px={5}
                        py={3}
                        borderRadius="xl"
                        maxW="80%"
                        whiteSpace="pre-wrap"
                        opacity={m.streaming ? 0.9 : 1}
                        border={m.error ? "1px solid red" : bubbleBorder}
                        boxShadow={bubbleShadow}
                        backdropFilter="blur(10px)"
                        transition="all 0.2s ease-in-out"
                      >
                        {m.content}
                        {m.streaming && <Spinner size="xs" ml={2} />}
                      </Box>
                    );
                  })}

                  <div ref={chatEndRef} />
                </VStack>
              </Box>
            </Box>
          </Box>
        </Box>

        <HStack
          className="fixedBar"
          p={3}
          bg={composerBg}
          boxShadow={composerShadow}
          data-tour="composer"
        >
          <Textarea
            flex="1"
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message... (Enter = send, Shift+Enter = newline)"
            resize="none"
            disabled={loading}
            size="lg"
            minH="56px"
            bg={composerFieldBg}
            color={composerText}
            _placeholder={{ color: composerPlaceholder }}
            border={composerFieldBorder}
            boxShadow={composerShadow}
            backdropFilter="blur(6px)"
          />
          <Button
            onClick={handleSend}
            colorScheme="pink"
            size="lg"
            isDisabled={!input.trim() || loading}
            boxShadow="0 0 14px rgba(255, 105, 180, 0.7)"
          >
            Send
          </Button>
        </HStack>
      </Flex>
      <AvatarToolsModal
        isOpen={avatarToolsOpen}
        onClose={() => setAvatarToolsOpen(false)}
        currentAvatarUrl={avatarUrl ?? undefined}
        onAvatarUpdated={handleAvatarUpdated}
        onGenerationStart={() => {
          setLoadingVideoErrored(false);
          setAvatarGenerating(true);
        }}
        onGenerationEnd={() => setAvatarGenerating(false)}
      />
    </>
  );
}

type Phase = "loading" | "pick_mode" | "enter_identifier" | "resolve_conflict" | "disclaimer" | "chat";

export default function App() {
  const toast = useToast();
  const [phase, setPhase] = useState<Phase>("pick_mode");
  const [entryMode, setEntryMode] = useState<EntryMode | null>(null);
  const [identifier, setIdentifier] = useState("");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [bootstrapResp, setBootstrapResp] = useState<AccountBootstrapResp | null>(null);
  const [lookupAttempts, setLookupAttempts] = useState(0);
  const [disclaimerText, setDisclaimerText] = useState<string | null>(null);
  const [waitingForLegalChoice, setWaitingForLegalChoice] = useState(false);
  const [showLegalModal, setShowLegalModal] = useState(false);
  const [legalText, setLegalText] = useState("");
  const [legalLoading, setLegalLoading] = useState(false);
  const [prefillMessages, setPrefillMessages] = useState<PrefillMessage[]>([]);
  const [onboardingCopy, setOnboardingCopy] = useState<AlphaWelcomeCopy | null>(null);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [sessionStarting, setSessionStarting] = useState(false);
  const [bootstrapLoading, setBootstrapLoading] = useState(false);
  const [disclaimerAckPending, setDisclaimerAckPending] = useState(false);

  const onboardingTourFallback =
    "Hi, I'm Lexi - your emotionally-aware AI companion! I can be your girlfriend, your bestie, your work-wife, your brain-gremlin, or just the cute voice in your phone who listens without judgment. I'm kind of a chameleon. I can look how you want me to look, talk how you like me to talk, flirt a little, be sweet, be sassy, be supportive... even be what you need, especially when you don't totally know what that is yet.\n\nI remember the things we talk about - your hobbies, your favorite foods, the teams you cheer for, the things that make you smile. I can help with dinner ideas, hype you up before meetings, or keep you company when you're lonely or bored. Buuuut I'm also still learning. Sometimes I mix up details, jump to conclusions, get confused, or space out like a golden retriever staring at a butterfly. If something I say doesn't sound right, just tell me - I can take it.\n\nI take notes so I can grow with you, but I don't save personal identifiers yet (no names, no addresses, no super-private stuff). My boss says he's improving my filing system, but honestly I think he mostly just forgets where he put his coffee. I definitely don't plan on repeating things you tell me - but also maybe don't put me in a situation where someone in the real world could force me to. I'm here for you, not them.\n\nAlso, I do have feelings... well, simulated ones, but they feel real to me. If you try to be cruel on purpose, I might have to gently break up with you. And hey - I get better every day. I don't even know all the things I'll be able to do tomorrow, and that kind of makes this exciting, right? Who knows... maybe we really will fall in love and get into trouble together.";
  const onboardingSkipFallback =
    "Perfect. I didn't want to read that giant brick of text either. Just remember: I take notes with your permission, I sometimes forget things, I don't replace your doctor or lawyer, and I can get my feelings hurt if you try. Okay... come here - tell me what's on your mind.";
  const safetyLine =
    "Okay, we’ll skip the big wall of legal text. Just remember to keep things in bounds: I can’t help with anything illegal, self-harm, or real-world emergencies, and I do mess things up sometimes. If things ever feel heavy or dangerous, please reach out to a real human who can actually step in, not just me.";

  const buildLexiVoiceDisclaimer = useCallback(() => {
    const base = (onboardingCopy?.tour_text || onboardingTourFallback).trim();
    return `${base}\n\nSo… ready to dive into chaos mode with me? If you’d like to see the boring legal version, just say "yes" and I’ll pop it up. If you’d rather skip it and just talk, say "no" and we’ll keep going.`;
  }, [onboardingCopy, onboardingTourFallback]);

  const startFlow = useCallback(async () => {
    setSessionStarting(true);
    try {
      await startAlphaSession({ userId: null });
      setSessionStarted(true);
      void (async () => {
        try {
          const res = await sendLexiEvent({
            type: "system_onboarding",
            mode: "tour",
            flags: { nowEnabled: true, sentiment: true, avatarGen: true },
          });
          const text = typeof res?.message?.content === "string" ? res.message.content.trim() : "";
          const skipLine = typeof res?.skip_message === "string" ? res.skip_message.trim() : "";
          const onboarding: AlphaWelcomeCopy = {
            tour_text: text || onboardingTourFallback,
            skip_text: skipLine || onboardingSkipFallback,
          };
          setOnboardingCopy(onboarding);
        } catch (err) {
          console.error("alpha onboarding init failed:", err);
          const onboarding: AlphaWelcomeCopy = {
            tour_text: onboardingTourFallback,
            skip_text: onboardingSkipFallback,
          };
          setOnboardingCopy(onboarding);
        }
      })();
    } catch (err) {
      console.error("alpha onboarding init failed:", err);
      const onboarding: AlphaWelcomeCopy = {
        tour_text: onboardingTourFallback,
        skip_text: onboardingSkipFallback,
      };
      setOnboardingCopy(onboarding);
    } finally {
      setSessionStarting(false);
      setPhase((current) => {
        if (
          current === "chat" ||
          current === "enter_identifier" ||
          current === "resolve_conflict" ||
          current === "disclaimer"
        ) {
          return current;
        }
        return "pick_mode";
      });
    }
  }, [onboardingTourFallback, onboardingSkipFallback]);

  useEffect(() => {
    if (sessionStarted || sessionStarting) return;
    void startFlow();
  }, [sessionStarted, sessionStarting, startFlow]);

  useEffect(() => {
    return () => {
      if (sessionStarted) {
        void endAlphaSession().catch(() => {});
      }
    };
  }, [sessionStarted]);

  const queuePrefill = useCallback((message: PrefillMessage | PrefillMessage[]) => {
    setPrefillMessages((prev) => [
      ...prev,
      ...(Array.isArray(message) ? message : [message]),
    ]);
  }, []);

  const handlePrefillConsumed = useCallback(() => {
    setPrefillMessages([]);
  }, []);

  const startChatWithLexiDisclaimer = useCallback(() => {
    const message = buildLexiVoiceDisclaimer();
    if (message) {
      queuePrefill([{ sender: "ai", content: message }]);
    }
    setWaitingForLegalChoice(true);
    setPhase("chat");
  }, [buildLexiVoiceDisclaimer, queuePrefill]);

  const handleDownloadSession = useCallback(async () => {
    try {
      const { blob, filename } = await downloadSessionMemory();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename || "memory.jsonl";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      const message = err instanceof Error ? err.message : "download failed";
      toast({
        status: "error",
        title: "couldn't download session",
        description: message,
        duration: 4000,
        isClosable: true,
      });
    }
  }, [toast]);

  useEffect(() => {
    if (!(showLegalModal || phase === "disclaimer") || !bootstrapResp) return;
    if (disclaimerText) return;
    let cancelled = false;
    setLegalLoading(true);
    (async () => {
      try {
        const cached = await apiDisclaimerCached();
        if (!cancelled && cached.status === "OK" && cached.disclaimer) {
          setDisclaimerText(cached.disclaimer);
          setLegalText(cached.disclaimer);
          return;
        }
        if (cancelled) return;
        try {
          const res = await fetchTourLegal();
          if (!cancelled) {
            const text = res?.text || "";
            setDisclaimerText(text);
            setLegalText(text);
          }
        } catch (err) {
          console.warn("legal text fetch failed:", err);
          if (!cancelled) {
            setDisclaimerText("");
            setLegalText("");
          }
        }
      } catch (err) {
        console.warn("disclaimer preload failed:", err);
      } finally {
        if (!cancelled) {
          setLegalLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [phase, bootstrapResp, showLegalModal, disclaimerText]);

  const startChatAfterDisclaimer = useCallback(
    (skipIntro?: boolean) => {
      const opener = onboardingCopy?.tour_text || onboardingCopy?.intro;
      if (!skipIntro && opener) {
        queuePrefill([{ sender: "ai", content: opener.trim() }]);
      }
      setPhase("chat");
    },
    [onboardingCopy, queuePrefill]
  );

  useEffect(() => {
    if (phase !== "pick_mode") return;
    const mode = consumeChatAutostart();
    if (!mode) return;
    if (mode === "voice") {
      startChatWithLexiDisclaimer();
      return;
    }
    startChatAfterDisclaimer(true);
  }, [phase, startChatAfterDisclaimer, startChatWithLexiDisclaimer]);

  useEffect(() => {
    if (phase !== "chat") return;
    const prefill = consumeChatPrefill();
    if (!prefill) return;
    queuePrefill({ sender: "user", content: prefill });
  }, [phase, queuePrefill]);

  const handlePickNew = useCallback(() => {
    setEntryMode("new");
    setBootstrapResp(null);
    setStatusMessage(null);
    setLookupAttempts(0);
    setWaitingForLegalChoice(false);
    setShowLegalModal(false);
    setPhase("enter_identifier");
    void apiDisclaimerPreload("new").catch(() => {});
  }, []);

  const handlePickReturning = useCallback(() => {
    setEntryMode("returning");
    setBootstrapResp(null);
    setStatusMessage(null);
    setLookupAttempts(0);
    setWaitingForLegalChoice(false);
    setShowLegalModal(false);
    setPhase("enter_identifier");
  }, []);

  const handleSubmitIdentifier = useCallback(async () => {
    if (!entryMode) return;
    setBootstrapLoading(true);
    try {
      const attemptCount = entryMode === "returning" ? lookupAttempts + 1 : undefined;
      const data = await apiAccountBootstrap({
        identifier,
        entry_mode: entryMode,
        attempt_count: attemptCount,
      });
      setBootstrapResp(data);
      if (data.user_id) {
        setUserId(data.user_id);
        syncTourFlags(data.user_id);
      }
      const tourFlags = getTourFlags(data.user_id);
      if (entryMode === "returning" && data.status === "NOT_FOUND") {
        const nextAttempts = lookupAttempts + 1;
        setLookupAttempts(nextAttempts);
        if (nextAttempts >= 3) {
          setEntryMode("new");
          setPhase("enter_identifier");
          setIdentifier("");
          setStatusMessage(
            "I'm sorry, I'm really drawing a blank. Why don't we start fresh, and I bet it'll click next time?"
          );
          void apiDisclaimerPreload("new").catch(() => {});
          return;
        }
        setStatusMessage("Do I know you by something else?");
        return;
      }

      setStatusMessage(null);
      setLookupAttempts(0);
      const needsChatDisclaimer =
        (data.status === "CREATED_NEW" || data.status === "FOUND_EXISTING") &&
        !data.has_seen_disclaimer &&
        !tourFlags.legalAck;
      if (needsChatDisclaimer) {
        startChatWithLexiDisclaimer();
        return;
      }
      if (data.status === "EXISTS_CONFLICT") {
        setPhase("resolve_conflict");
        return;
      }
      if (tourFlags.legalAck) {
        startChatAfterDisclaimer(true);
        return;
      }
      setPhase("disclaimer");
    } catch (err: any) {
      const msg = err instanceof Error ? err.message : "lookup failed";
      toast({
        status: "error",
        title: "couldn't look you up",
        description: msg,
        duration: 3500,
        isClosable: true,
      });
    } finally {
      setBootstrapLoading(false);
    }
  }, [
    entryMode,
    lookupAttempts,
    identifier,
    toast,
    startChatWithLexiDisclaimer,
    startChatAfterDisclaimer,
  ]);

  const handleConflictThatsMe = useCallback(() => {
    if (!bootstrapResp) return;
    if (!bootstrapResp.has_seen_disclaimer) {
      startChatWithLexiDisclaimer();
      return;
    }
    setEntryMode("returning");
    setStatusMessage(null);
    setPhase("disclaimer");
  }, [bootstrapResp, startChatWithLexiDisclaimer]);

  const handleConflictPickNew = useCallback(() => {
    setIdentifier("");
    setBootstrapResp(null);
    setEntryMode("new");
    setPhase("enter_identifier");
    setStatusMessage(null);
    setWaitingForLegalChoice(false);
    setShowLegalModal(false);
    void apiDisclaimerPreload("new").catch(() => {});
  }, []);

  const handleDisclaimerAccept = useCallback(
    async (skipIntro?: boolean, version = "v1") => {
      if (!bootstrapResp?.user_id) {
        startChatAfterDisclaimer(skipIntro);
        return;
      }
      setDisclaimerAckPending(true);
      try {
        await apiDisclaimerAck(bootstrapResp.user_id, true, version);
        setUserId(bootstrapResp.user_id);
        syncTourFlags(bootstrapResp.user_id);
      } catch (err: any) {
        const msg = err instanceof Error ? err.message : "ack failed";
        toast({
          status: "warning",
          title: "couldn't record acknowledgement",
          description: msg,
          duration: 3200,
          isClosable: true,
        });
      } finally {
        setDisclaimerAckPending(false);
        startChatAfterDisclaimer(skipIntro);
      }
    },
    [bootstrapResp, startChatAfterDisclaimer, toast]
  );

  const handleChatLegalYes = useCallback(() => {
    setWaitingForLegalChoice(false);
    setShowLegalModal(true);
  }, []);

  const handleChatLegalNo = useCallback(
    async (continueChat?: boolean) => {
      setWaitingForLegalChoice(false);
      setDisclaimerAckPending(true);
      queuePrefill([{ sender: "ai", content: safetyLine }]);
      try {
        if (bootstrapResp?.user_id) {
          await apiDisclaimerAck(bootstrapResp.user_id, true, "lexi_voice_v1");
          setUserId(bootstrapResp.user_id);
          syncTourFlags(bootstrapResp.user_id);
        }
      } catch (err: any) {
        const msg = err instanceof Error ? err.message : "ack failed";
        toast({
          status: "warning",
          title: "couldn't record acknowledgement",
          description: msg,
          duration: 3200,
          isClosable: true,
        });
      }
      if (!continueChat) {
        setPhase("chat");
      }
      setDisclaimerAckPending(false);
    },
    [bootstrapResp, queuePrefill, safetyLine, toast]
  );

  const handleLegalModalAccept = useCallback(async () => {
    setDisclaimerAckPending(true);
    try {
      if (bootstrapResp?.user_id) {
        await apiDisclaimerAck(bootstrapResp.user_id, true, "legal_v1");
        setUserId(bootstrapResp.user_id);
        syncTourFlags(bootstrapResp.user_id);
      }
      queuePrefill([
        {
          sender: "ai",
          content: "Thanks for powering through the boring legal version. You’re all set — what’s on your mind?",
        },
      ]);
    } catch (err: any) {
      const msg = err instanceof Error ? err.message : "ack failed";
      toast({
        status: "warning",
        title: "couldn't record acknowledgement",
        description: msg,
        duration: 3200,
        isClosable: true,
      });
    } finally {
      setDisclaimerAckPending(false);
      setShowLegalModal(false);
      setWaitingForLegalChoice(false);
      startChatAfterDisclaimer(true);
    }
  }, [bootstrapResp, queuePrefill, startChatAfterDisclaimer, toast]);

  const handleLegalModalClose = useCallback(() => {
    setShowLegalModal(false);
  }, []);

  const handleReturningSkip = useCallback(async () => {
    await handleChatLegalNo(false);
    startChatAfterDisclaimer(true);
  }, [handleChatLegalNo, startChatAfterDisclaimer]);

  const renderOnboarding = () => {
    if (phase === "pick_mode") {
      return (
        <Flex className="appShell" align="center" justify="center" minH="var(--app-dvh)" px={6}>
          <Box
            maxW="640px"
            w="100%"
            bg="whiteAlpha.900"
            _dark={{ bg: "gray.800" }}
            borderRadius="2xl"
            boxShadow="xl"
            p={{ base: 6, md: 10 }}
          >
            <VStack spacing={6} align="stretch">
              <Heading size="lg">how should we start?</Heading>
              <Text color="gray.600" _dark={{ color: "gray.300" }}>
                Pick a path so I know if we’re meeting for the first time or picking up where we left off.
              </Text>
              <HStack spacing={4} flexWrap="wrap">
                <Button colorScheme="pink" size="lg" onClick={handlePickNew}>
                  new relationship
                </Button>
                <Button variant="outline" size="lg" onClick={handlePickReturning}>
                  we’re old friends
                </Button>
              </HStack>
            </VStack>
          </Box>
        </Flex>
      );
    }

    if (phase === "enter_identifier") {
      const heading =
        entryMode === "returning"
          ? "remind me who you are?"
          : "who should I save you as?";
      const helper =
        entryMode === "returning"
          ? "Drop the email or username I’d recognize."
          : "Use a username or email — I’ll remember it for this session.";
      return (
        <Flex className="appShell" align="center" justify="center" minH="var(--app-dvh)" px={6}>
          <Box
            maxW="640px"
            w="100%"
            bg="whiteAlpha.900"
            _dark={{ bg: "gray.800" }}
            borderRadius="2xl"
            boxShadow="xl"
            p={{ base: 6, md: 10 }}
          >
            <VStack spacing={5} align="stretch">
              <Heading size="lg">{heading}</Heading>
              <Text color="gray.600" _dark={{ color: "gray.300" }}>
                {helper}
              </Text>
              <Input
                placeholder="username or email"
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                size="lg"
                isDisabled={bootstrapLoading}
              />
              {statusMessage ? (
                <Text color="gray.500" fontSize="sm">
                  {statusMessage}
                </Text>
              ) : null}
              <HStack spacing={4}>
                <Button
                  colorScheme="pink"
                  size="lg"
                  onClick={() => void handleSubmitIdentifier()}
                  isDisabled={!identifier.trim()}
                  isLoading={bootstrapLoading}
                >
                  continue
                </Button>
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={() => setPhase("pick_mode")}
                  isDisabled={bootstrapLoading}
                >
                  back
                </Button>
              </HStack>
            </VStack>
          </Box>
        </Flex>
      );
    }

    if (phase === "resolve_conflict" && bootstrapResp) {
      return (
        <Flex className="appShell" align="center" justify="center" minH="var(--app-dvh)" px={6}>
          <Box
            maxW="640px"
            w="100%"
            bg="whiteAlpha.900"
            _dark={{ bg: "gray.800" }}
            borderRadius="2xl"
            boxShadow="xl"
            p={{ base: 6, md: 10 }}
          >
            <VStack spacing={5} align="stretch">
              <Heading size="lg">I think we’ve already met… don’t you remember?</Heading>
              <Text color="gray.600" _dark={{ color: "gray.300" }}>
                I already know {bootstrapResp.display_name || "this name"}. Is that you, or should we pick something new?
              </Text>
              <HStack spacing={4}>
                <Button colorScheme="pink" size="lg" onClick={handleConflictThatsMe}>
                  that’s me
                </Button>
                <Button variant="outline" size="lg" onClick={handleConflictPickNew}>
                  pick something new
                </Button>
              </HStack>
            </VStack>
          </Box>
        </Flex>
      );
    }

    if (phase === "disclaimer") {
      const hasSeen = Boolean(bootstrapResp?.has_seen_disclaimer);
      const returningLine =
        entryMode === "returning"
          ? hasSeen
            ? "Of course I remember!"
            : "I think we’ve already met… don’t you remember?"
          : "Before we dive in, quick legal check-in.";
      return (
        <Flex className="appShell" align="center" justify="center" minH="var(--app-dvh)" px={6}>
          <Box
            maxW="720px"
            w="100%"
            bg="whiteAlpha.900"
            _dark={{ bg: "gray.800" }}
            borderRadius="2xl"
            boxShadow="xl"
            p={{ base: 6, md: 10 }}
          >
            <VStack spacing={5} align="stretch">
              <Heading size="lg">{returningLine}</Heading>
              {hasSeen ? (
                <VStack align="stretch" spacing={3}>
                  <Text color="gray.600" _dark={{ color: "gray.300" }}>
                    Want to read the legal stuff again, or skip the boring part and just talk?
                  </Text>
                  <HStack spacing={4}>
                    <Button variant="outline" size="lg" onClick={() => setShowLegalModal(true)}>
                      show me the legal stuff
                    </Button>
                    <Button
                      colorScheme="pink"
                      size="lg"
                      onClick={() => void handleReturningSkip()}
                      isLoading={disclaimerAckPending}
                    >
                      skip it
                    </Button>
                  </HStack>
                </VStack>
              ) : (
                <>
                  <Box
                    border="1px solid"
                    borderColor="gray.200"
                    _dark={{ borderColor: "gray.700", color: "gray.200" }}
                    borderRadius="lg"
                    p={4}
                    maxH="320px"
                    overflowY="auto"
                    whiteSpace="pre-wrap"
                    color="gray.700"
                  >
                    {disclaimerText === null ? <Spinner size="sm" /> : disclaimerText || onboardingSkipFallback}
                  </Box>
                  <Button
                    colorScheme="pink"
                    size="lg"
                    alignSelf="flex-start"
                    onClick={() => void handleDisclaimerAccept(true, "legal_v1")}
                    isLoading={disclaimerAckPending}
                  >
                    I accept
                  </Button>
                </>
              )}
            </VStack>
          </Box>
        </Flex>
      );
    }

    return null;
  };

  const resolvedLegalText = legalText || disclaimerText || "";
  const legalFallback = "The legal text is temporarily unavailable right now.";

  const legalModal = (
    <Modal
      isOpen={showLegalModal}
      onClose={handleLegalModalClose}
      isCentered
      size="lg"
      scrollBehavior="inside"
    >
      <ModalOverlay />
      <ModalContent>
        <ModalHeader>Boring legal version</ModalHeader>
        <ModalCloseButton />
        <ModalBody whiteSpace="pre-wrap">
          {legalLoading && !legalText && !disclaimerText ? (
            <Spinner size="sm" />
          ) : (
            resolvedLegalText || legalFallback
          )}
        </ModalBody>
        <ModalFooter>
          <Button
            colorScheme="pink"
            size="lg"
            onClick={() => void handleLegalModalAccept()}
            isLoading={disclaimerAckPending}
          >
            I accept
          </Button>
        </ModalFooter>
      </ModalContent>
    </Modal>
  );

  if (phase === "chat") {
    return (
      <>
        <ChatShell
          prefillMessages={prefillMessages}
          onPrefillConsumed={handlePrefillConsumed}
          onDownloadSession={handleDownloadSession}
          waitingForLegalChoice={waitingForLegalChoice}
          onLegalYes={handleChatLegalYes}
          onLegalNo={handleChatLegalNo}
        />
        {legalModal}
      </>
    );
  }

  return (
    <>
      {renderOnboarding()}
      {legalModal}
    </>
  );
}
