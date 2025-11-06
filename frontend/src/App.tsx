import React, {
  useState,
  useRef,
  useEffect,
  KeyboardEvent,
  useCallback,
  useContext,
} from "react";
import { v4 as uuidv4 } from "uuid";
import {
  Box,
  Button,
  Flex,
  HStack,
  IconButton,
  Image,
  Spacer,
  Textarea,
  VStack,
  useColorMode,
  useColorModeValue,
  Spinner,
} from "@chakra-ui/react";
import { MoonIcon, SunIcon } from "@chakra-ui/icons";

import {
  fetchPersona,
  addTrait,
  generateAvatar,
  sendPrompt,
  loadApiConfig,
  classifyIntent
} from "./services/api";
import type { TraitResponse, Persona } from "./services/api";
import { TourContext } from "./tour/TourProvider";
import "./styles/chat.css";

interface ChatMessage {
  id: string;
  sender: "user" | "ai" | "system";
  content: string;
  streaming?: boolean;
  error?: boolean;
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

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const bg = useColorModeValue("gray.50", "gray.800");

  const { start: startTour } = useContext(TourContext);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [staticAvatarBase, setStaticAvatarBase] = useState(DEFAULT_STATIC_AVATAR_BASE);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(DEFAULT_AVATAR_URL);
  const [avatarFlow, setAvatarFlow] = useState(false);
  const [persona, setPersona] = useState<Persona | null>(null);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
      setPersona(data);
      setMessages((prev) => [
        ...prev,
        {
          id: mkId(),
          sender: "system",
          content: `👋 ${(data as any).mode ?? "default"} persona loaded!`,
        },
      ]);
      setAvatarUrl(resolveAvatarUrl((data as any).image_path, nextStaticBase));
    })();
    return () => {
      mounted = false;
    };
  }, []);


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
      const gen = await generateAvatar(res.prompt);
      let img = (gen as any).image || (gen as any).image_url || (gen as any).path || "";
      const imgUrl = resolveAvatarUrl(img, staticAvatarBase);
      const finalUrl = `${imgUrl}${imgUrl.includes("?") ? "&" : "?"}v=${Date.now()}`;
      setAvatarUrl(finalUrl);
      appendMessage({ sender: "ai", content: "📸 Here's your avatar!" });

      // Now update persona, but do **NOT** set avatarUrl from it!
      const updated = await fetchPersona();
      setPersona(updated);

    } catch (err) {
      console.error(err);
      appendMessage({ sender: "ai", content: "[Avatar update failed]" });
    }
  },
  [appendMessage, staticAvatarBase]
);

const handleAvatarEdit = useCallback(
  async (userText: string) => {
    setLoading(true);
    try {
      const result = await sendPrompt({ prompt: userText, intent: "avatar_edit" });
      let reply =
        typeof result === "object" &&
        result !== null &&
        typeof (result as any).cleaned === "string"
          ? (result as any).cleaned
          : "Got it, updating her look! 💄";
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
      }
      appendMessage({ sender: "ai", content: reply });
      try {
        const updated = await fetchPersona();
        if (updated) {
          setPersona(updated);
        }
      } catch (personaErr) {
        console.warn("Failed to refresh persona after avatar edit", personaErr);
      }
    } catch (err) {
      console.error(err);
      appendMessage({ sender: "ai", content: "[Avatar edit failed]", error: true });
    } finally {
      setAvatarFlow(false);
      setLoading(false);
    }
  },
  [appendMessage, staticAvatarBase]
);


const handleSend = useCallback(async () => {
  const text = input.trim();
  if (!text || loading) return;
  setInput("");
  appendMessage({ sender: "user", content: text });

  // ——— 1) If we’re already mid‑flow, keep going —
  if (avatarFlow) {
    handleTraitFlow(text);
    return;
  }

  // ——— 2) Otherwise ask the backend “chat or avatar_flow?” —
  let intent = "chat";
  try {
    ({ intent } = await classifyIntent(text));
    console.log("🕵️ Intent is", intent);
  } catch (e) {
    console.warn("Intent API failed, defaulting to chat", e);
  }

  if (intent === "avatar_flow" || intent === "new_look") {
    // flip your FSM
    setAvatarFlow(true);
    // send this first message into the trait collector
    handleTraitFlow(text);
    return;
  }

  if (intent === "avatar_edit") {
    await handleAvatarEdit(text);
    return;
  }

  if (intent === "describe_avatar") {
  // 🎭 Don't enter avatar flow. Just describe her current traits
    const personaData = await fetchPersona();
    if (!personaData || !personaData.traits) {
      appendMessage({ sender: "ai", content: "I'm not sure how I look right now…" });
      return;
    }
    const traits = personaData.traits;

    const desc = Object.entries(traits)
      .map(([k, v]) => `${k.replace("_", " ")}: ${v}`)
      .join(", ");
    appendMessage({ sender: "ai", content: `Here's how I look right now: ${desc}.` });
    return;
  }

  // ——— 3) Fallback to your normal LLM chat —
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
}, [input, loading, avatarFlow, appendMessage, handleTraitFlow, handleAvatarEdit]);

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
    <Flex direction="column" h="100vh" bg={bg} fontFamily="'Nunito', sans-serif">
      <HStack
        p={3}
        bg="pink.500"
        color="white"
        boxShadow="0 2px 12px rgba(255, 105, 180, 0.6)"
        data-tour="header"
      >
        <Box fontWeight="bold" fontSize="xl">
          Lexi Chat
        </Box>
        <Box fontSize="sm" color="pink.100" ml={2} data-tour="modes">
          mode: {(persona as any)?.mode ?? "loading…"}
        </Box>
        <Spacer />
        <Button size="sm" variant="ghost" color="white" onClick={startTour}>
          Take a tour
        </Button>
        <IconButton
          aria-label="Toggle dark mode"
          icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
          onClick={toggleColorMode}
          variant="ghost"
          color="white"
        />
      </HStack>

      <Box flex="1" px={4} pt={4} pb={2}>
        <Box className="chat-layout with-avatar" h="100%">
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
                <Image
                  src={avatarUrl}
                  alt="Lex avatar"
                  w="100%"
                  display="block"
                  backdropFilter="blur(4px)"
                />
              </Box>
            )}
            <Button
              mt={3}
              variant="outline"
              colorScheme="pink"
              size="sm"
              onClick={() => setAvatarFlow(true)}
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
          >
            <Box
              className="chat-scroll"
              flex="1"
              overflowY="auto"
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
                      bg={
                        m.sender === "user"
                          ? "pink.400"
                          : m.sender === "ai"
                          ? "rgba(100, 100, 100, 0.3)"
                          : "purple.600"
                      }
                      color="white"
                      px={5}
                      py={3}
                      borderRadius="xl"
                      maxW="80%"
                      whiteSpace="pre-wrap"
                      opacity={m.streaming ? 0.9 : 1}
                      border={m.error ? "1px solid red" : "none"}
                      boxShadow="0 0 16px rgba(255, 105, 180, 0.8)"
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
        p={3}
        bg="pink.500"
        boxShadow="0 -2px 12px rgba(255, 105, 180, 0.5)"
        data-tour="composer"
      >
        <Textarea
          flex="1"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message… (Enter = send, Shift+Enter = newline)"
          resize="none"
          disabled={loading}
          bg="rgba(255, 255, 255, 0.05)"
          color="white"
          _placeholder={{ color: "pink.200" }}
          border="1px solid hotpink"
          boxShadow="0 0 10px rgba(255, 105, 180, 0.6)"
          backdropFilter="blur(6px)"
        />
        <Button
          onClick={handleSend}
          colorScheme="pink"
          isDisabled={!input.trim() || loading}
          boxShadow="0 0 14px rgba(255, 105, 180, 0.7)"
        >
          Send
        </Button>
      </HStack>
    </Flex>
  );
}

export default App;
