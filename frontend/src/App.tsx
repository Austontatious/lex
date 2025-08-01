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
  // apiStream, // ❌ Streaming disabled
  sendPrompt,
  BACKEND,
  classifyIntent
} from "./services/api";
import type { TraitResponse, Persona } from "./services/api";

interface ChatMessage {
  id: string;
  sender: "user" | "ai" | "system";
  content: string;
  streaming?: boolean;
  error?: boolean;
}

function mkId() {
  return `m_${uuidv4()}`;
}

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const bg = useColorModeValue("gray.50", "gray.800");

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
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
      if ((data as any).image_path) {
        setAvatarUrl(`${BACKEND}${(data as any).image_path}`);
      }
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

      if (res.ask) {
        appendMessage({ sender: "ai", content: res.ask });
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
      const gen = await generateAvatar({});
      let img = (gen as any).image || (gen as any).image_url || (gen as any).path || "";
      let normImg = img.startsWith("/static/lex/avatars/")
        ? img
        : `/static/lex/avatars/${img.split("/").pop()}`;
      const imgUrl = `${BACKEND}${normImg}${normImg.includes("?") ? "&" : "?"}v=${Date.now()}`;
      setAvatarUrl(imgUrl);
      appendMessage({ sender: "ai", content: "📸 Here's your avatar!" });

      // Now update persona, but do **NOT** set avatarUrl from it!
      const updated = await fetchPersona();
      setPersona(updated);

    } catch (err) {
      console.error(err);
      appendMessage({ sender: "ai", content: "[Avatar update failed]" });
    }
  },
  [appendMessage]
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

  if (intent === "avatar_flow") {
    // flip your FSM
    setAvatarFlow(true);
    // send this first message into the trait collector
    handleTraitFlow(text);
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
  appendMessage({ id: aiId, sender: "ai", content: "…" });
  setLoading(true);
  try {
    const res = await sendPrompt({ prompt: text });
    setMessages((prev) =>
      prev.map((m) =>
        m.id === aiId
          ? {
              ...m,
              content: res.cleaned?.trim() || "[no response]"
            }
          : m
      )
    );
  } catch (err) {
  } finally {
    setLoading(false);
  }
}, [input, loading, avatarFlow, appendMessage, handleTraitFlow]);

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
      <HStack p={3} bg="pink.500" color="white" boxShadow="0 2px 12px rgba(255, 105, 180, 0.6)">
        <Box fontWeight="bold" fontSize="xl">
          Lexi Chat
        </Box>
        {persona && (
          <Box fontSize="sm" color="pink.100" ml={2}>
            mode: {(persona as any).mode ?? "default"}
          </Box>
        )}
        <Spacer />
        <IconButton
          aria-label="Toggle dark mode"
          icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
          onClick={toggleColorMode}
          variant="ghost"
          color="white"
        />
      </HStack>

      <Box flex="1" position="relative" overflow="hidden">
        {/* Floating Avatar */}
        {avatarUrl && (
          <Box
            position="absolute"
            top="16px"
            left="16px"
            zIndex="10"
            animation="pulseGlow 2s infinite"
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
              maxW="250px"
              borderRadius="2xl"
              border="2px solid hotpink"
              backdropFilter="blur(4px)"
            />
          </Box>
        )}

        {/* Chat Scroll Window */}
        <Box
          h="100%"
          overflowY="auto"
          px={4}
          pt={4}
          pb={2}
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

      <HStack p={3} bg="pink.500" boxShadow="0 -2px 12px rgba(255, 105, 180, 0.5)">
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

