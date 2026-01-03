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
  useToast,
  Text,
  Badge,
  Heading,
} from "@chakra-ui/react";
import { MoonIcon, SunIcon, DownloadIcon } from "@chakra-ui/icons";

import {
  fetchPersona,
  addTrait,
  generateAvatar,
  sendPrompt,
  BACKEND,
  classifyIntent,
  startAlphaSession,
  updateAlphaConsent,
  endAlphaSession,
  fetchAlphaTourScript,
  requestTourAvatarPreview,
  submitTourNowTopic,
  submitTourMemoryNote,
  sendTourFeedback,
  postAlphaMetric,
  downloadSessionMemory,
} from "./services/api";
import type { TraitResponse, Persona } from "./services/api";
import AlphaWelcome from "./components/onboarding/AlphaWelcome";
import AlphaTour, { TourStep } from "./components/onboarding/AlphaTour";

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

type OnboardingView = "loading" | "welcome" | "tour" | "chat";

function App() {
  const { colorMode, toggleColorMode } = useColorMode();
  const bg = useColorModeValue("gray.50", "gray.800");
  const bannerBg = useColorModeValue("purple.50", "purple.900");
  const bannerColor = useColorModeValue("purple.700", "purple.100");
  const toast = useToast();

  const [view, setView] = useState<OnboardingView>("loading");
  const [sessionId, setSessionIdState] = useState<string | null>(null);
  const [consent, setConsent] = useState(true);
  const [alphaStrict, setAlphaStrict] = useState(false);
  const [tourSteps, setTourSteps] = useState<TourStep[]>([]);
  const [tourInitialized, setTourInitialized] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [downloadingLog, setDownloadingLog] = useState(false);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [avatarFlow, setAvatarFlow] = useState(false);
  const [persona, setPersona] = useState<Persona | null>(null);
  const [loading, setLoading] = useState(false);

  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let mounted = true;
    let started = false;
    (async () => {
      try {
        const data = await startAlphaSession();
        if (!mounted) return;
        started = true;
        setSessionIdState(data.session_id);
        setConsent(data.consent);
        setAlphaStrict(data.alpha_strict);
        setView("welcome");
        await postAlphaMetric("session_ready", { variant: data.variant });
        try {
          const script = await fetchAlphaTourScript();
          if (mounted && script?.steps) {
            setTourSteps(script.steps);
          }
        } catch (scriptErr) {
          console.warn("tour script fetch failed:", scriptErr);
        } finally {
          if (mounted) setTourInitialized(true);
        }
      } catch (err: any) {
        console.error("startAlphaSession failed:", err);
        if (!mounted) return;
        setSessionError(err?.message || "session initialization failed");
        setView("loading");
      }
    })();
    return () => {
      mounted = false;
      if (started) {
        void endAlphaSession().catch(() => {});
      }
    };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    const handleUnload = () => {
      void endAlphaSession().catch(() => {});
    };
    window.addEventListener("beforeunload", handleUnload);
    return () => {
      window.removeEventListener("beforeunload", handleUnload);
    };
  }, [sessionId]);

  const appendMessage = useCallback(
    (msg: Omit<ChatMessage, "id"> & { id?: string }) => {
      setMessages((prev) => [...prev, { id: msg.id ?? mkId(), ...msg }]);
    },
    []
  );

  const handleConsentToggle = useCallback(
    async (next: boolean) => {
      const previous = consent;
      setConsent(next);
      try {
        await updateAlphaConsent(next);
        void postAlphaMetric("consent_updated", { consent: next }).catch(() => {});
      } catch (err: any) {
        setConsent(previous);
        toast({
          status: "error",
          title: "couldn’t update consent",
          description: err?.message || "give it another shot in a moment.",
          duration: 4000,
          isClosable: true,
        });
      }
    },
    [consent, toast]
  );

  const handleOnboardingChoice = useCallback((choice: "tour" | "chat") => {
    if (choice === "tour") {
      setView("tour");
      void postAlphaMetric("tour_selected").catch(() => {});
    } else {
      setView("chat");
      void postAlphaMetric("chat_selected").catch(() => {});
    }
  }, []);

  const handleTourPreview = useCallback(async (prompt: string) => {
    try {
      const res = await requestTourAvatarPreview(prompt);
      void postAlphaMetric("tour_avatar_preview", { prompt }).catch(() => {});
      const url = res.preview_url || "";
      if (!url) throw new Error("no preview returned");
      return url.startsWith("http") ? url : `${BACKEND}${url}`;
    } catch (err: any) {
      const message =
        typeof err?.message === "string" && err.message.includes("429")
          ? "tour preview limit hit — let’s keep moving."
          : err?.message || "preview failed. try a different vibe?";
      throw new Error(message);
    }
  }, []);

  const handleTourTopic = useCallback(async (topic: string) => {
    try {
      await submitTourNowTopic(topic);
      void postAlphaMetric("tour_now_topic", { topic }).catch(() => {});
    } catch (err: any) {
      throw new Error(err?.message || "couldn’t lock the topic — try again.");
    }
  }, []);

  const handleTourMemory = useCallback(async (note: string) => {
    try {
      await submitTourMemoryNote(note);
      void postAlphaMetric("tour_memory_note").catch(() => {});
    } catch (err: any) {
      throw new Error(err?.message || "memory save hiccuped — try again?");
    }
  }, []);

  const handleTourStepMetric = useCallback(async (slug: string) => {
    await postAlphaMetric("tour_step_entered", { slug }).catch(() => {});
  }, []);

  const handleTourFeedback = useCallback(async (helpful: boolean, comment?: string) => {
    await sendTourFeedback(helpful, comment).catch(() => {});
    void postAlphaMetric("tour_feedback", { helpful }).catch(() => {});
  }, []);

  const handleSkipTour = useCallback(() => {
    setView("chat");
    void postAlphaMetric("tour_skipped").catch(() => {});
  }, []);

  const handleTourComplete = useCallback(() => {
    setView("chat");
    void postAlphaMetric("tour_completed").catch(() => {});
  }, []);

  const handleDownloadMemory = useCallback(async () => {
    setDownloadingLog(true);
    try {
      const { blob, filename } = await downloadSessionMemory();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast({
        status: "success",
        title: "session diary downloaded",
        duration: 2500,
      });
      void postAlphaMetric("memory_downloaded").catch(() => {});
    } catch (err: any) {
      toast({
        status: "error",
        title: "download failed",
        description: err?.message || "couldn’t download the session log.",
        duration: 4000,
      });
    } finally {
      setDownloadingLog(false);
    }
  }, [toast]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (view !== "chat") return;
    let mounted = true;
    (async () => {
      try {
        const data = await fetchPersona();
        if (!mounted || !data) return;
        setPersona(data);
        appendMessage({
          sender: "system",
          content: `👋 ${(data as any).mode ?? "default"} persona loaded!`,
        });
        if ((data as any).image_path) {
          const imgPath = (data as any).image_path;
          const imgUrl = imgPath.startsWith("http") ? imgPath : `${BACKEND}${imgPath}`;
          setAvatarUrl(imgUrl);
        }
      } catch (err: any) {
        console.error("fetchPersona failed:", err);
        if (!mounted) return;
        appendMessage({
          sender: "system",
          error: true,
          content:
            err?.name === "TimeoutError"
              ? "⏱️ Persona service is taking too long to respond. Try again shortly."
              : "⚠️ Couldn't reach the persona service. Please retry in a moment.",
        });
      }
    })();
    return () => {
      mounted = false;
    };
  }, [appendMessage, view]);

  useEffect(() => {
    if (view !== "chat") return;
    void postAlphaMetric("entered_chat").catch(() => {});
  }, [view]);

  // 🧠 INTELLIGENT TRAIT GATHERING LOOP
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

        let avatarRes: any;
        try {
          avatarRes = await generateAvatar(res.prompt ?? "");
        } catch (e) {
          console.error("generateAvatar failed:", e);
          appendMessage({ sender: "ai", content: "[Avatar update failed]" });
          return;
        }

        // pull most-likely image fields in order
        const firstImageField = (o: any): string | undefined =>
          ["image", "image_url", "path", "url", "file"].map((k) => o?.[k]).find(Boolean);

        let img = firstImageField(avatarRes);

        if (!img || typeof img !== "string") {
          console.warn("No usable image field in avatar response:", avatarRes);
          appendMessage({ sender: "ai", content: "[No image returned]" });
          return;
        }

        // ——— Normalize to a /static/... path the backend actually serves ———
        // If backend returned an absolute URL, use it as-is (append cache bust)
        let imgUrl: string;
        if (img.startsWith("http://") || img.startsWith("https://")) {
          const hasQ = img.includes("?");
          imgUrl = `${img}${hasQ ? "&" : "?"}v=${Date.now()}`;
        } else {
          // Normalize to a /static/... path the backend serves
          const normImg = img.startsWith("/static/")
            ? img
            : `/static/lexi/avatars/${img.split("/").pop()}`;
          const cacheBust = normImg.includes("?") ? `&v=${Date.now()}` : `?v=${Date.now()}`;
          imgUrl = `${BACKEND}${normImg}${cacheBust}`;
        }

        setAvatarUrl(imgUrl);
        appendMessage({ sender: "ai", content: "📸 Here's your avatar!" });

        // Now update persona, but do **NOT** set avatarUrl from it!
        try {
          const updated = await fetchPersona();
          setPersona(updated);
        } catch (e) {
          console.warn("fetchPersona failed:", e);
        }

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

    // ——— 1) If we’re already mid-flow, keep going —
    if (avatarFlow) {
      handleTraitFlow(text);
      return;
    }

    // ——— 2) Otherwise ask the backend “chat, avatar_flow, avatar_edit?” —
    let intent = "chat";
    try {
      ({ intent } = await classifyIntent(text));
      console.log("🕵️ Intent is", intent);
    } catch (e) {
      console.warn("Intent API failed, defaulting to chat", e);
    }

    // === NEW: start fresh look ===
    if (intent === "new_look") {
      const aiId = mkId();
      appendMessage({ id: aiId, sender: "ai", content: "Starting a fresh look…" });
      try {
        const avatarRes: any = await generateAvatar({ mode: "txt2img", fresh_base: true });
        const firstImageField = (o: any): string | undefined =>
          ["image", "image_url", "path", "url", "file"].map((k) => o?.[k]).find(Boolean);
        const img = firstImageField(avatarRes);
        if (!img || typeof img !== "string") throw new Error("no image returned");
        let imgUrl: string;
        if (img.startsWith("http://") || img.startsWith("https://")) {
          imgUrl = `${img}${img.includes("?") ? "&" : "?"}v=${Date.now()}`;
        } else {
          const normImg = img.startsWith("/static/") ? img : `/static/lexi/avatars/${img.split("/").pop()}`;
          const cacheBust = normImg.includes("?") ? `&v=${Date.now()}` : `?v=${Date.now()}`;
          imgUrl = `${BACKEND}${normImg}${cacheBust}`;
        }
        setAvatarUrl(imgUrl);
        setMessages((prev) => prev.map((m) => (m.id === aiId ? { ...m, content: "📸 Fresh base created." } : m)));
        try { const updated = await fetchPersona(); setPersona(updated); } catch {}
      } catch (e) {
        console.error("new_look failed:", e);
        setMessages((prev) => prev.map((m) => (m.id === aiId ? { ...m, content: "[Fresh render failed]" } : m)));
      }
      return;
    }

    // === NEW: direct small edits ===
    if (intent === "avatar_edit") {
      const aiId = mkId();
      appendMessage({ id: aiId, sender: "ai", content: "Updating your look…" });

      try {
        const avatarRes: any = await generateAvatar({
          mode: "img2img",
          // Let the backend use existing traits/prompt; we pass the delta only
          changes: text,
          denoise: 0.30, // sweet spot for small edits (identity preserved)
        });

        const firstImageField = (o: any): string | undefined =>
          ["image", "image_url", "path", "url", "file"].map((k) => o?.[k]).find(Boolean);

        let img = firstImageField(avatarRes);
        if (!img || typeof img !== "string") {
          setMessages((prev) =>
            prev.map((m) => (m.id === aiId ? { ...m, content: "[No image returned]" } : m))
          );
          return;
        }

        let imgUrl: string;
        if (img.startsWith("http://") || img.startsWith("https://")) {
          const hasQ = img.includes("?");
          imgUrl = `${img}${hasQ ? "&" : "?"}v=${Date.now()}`;
        } else {
          const normImg = img.startsWith("/static/")
            ? img
            : `/static/lexi/avatars/${img.split("/").pop()}`;
          const cacheBust = normImg.includes("?") ? `&v=${Date.now()}` : `?v=${Date.now()}`;
          imgUrl = `${BACKEND}${normImg}${cacheBust}`;
        }

        setAvatarUrl(imgUrl);
        setMessages((prev) =>
          prev.map((m) => (m.id === aiId ? { ...m, content: "📸 Here's your avatar!" } : m))
        );

        try {
          const updated = await fetchPersona();
          setPersona(updated);
        } catch (e) {
          console.warn("fetchPersona failed:", e);
        }
      } catch (e) {
        console.error("avatar_edit failed:", e);
        appendMessage({ sender: "ai", content: "[Avatar update failed]" });
      }
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
            ? { ...m, content: res.cleaned?.trim() || "[no response]" }
            : m
        )
      );
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) => (m.id === aiId ? { ...m, error: true, content: "[error]" } : m))
      );
    } finally {
      setLoading(false);
    }
  }, [input, loading, avatarFlow, appendMessage, handleTraitFlow]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (view !== "chat") {
    if (sessionError) {
      return (
        <Flex direction="column" align="center" justify="center" minH="100vh" bg={bg} px={6} textAlign="center">
          <Heading size="md" mb={4}>
            session bootstrap hiccup
          </Heading>
          <Text color="gray.500" mb={6}>
            {sessionError}
          </Text>
          <Button colorScheme="purple" onClick={() => window.location.reload()}>
            refresh and try again
          </Button>
        </Flex>
      );
    }

    if (view === "welcome" && sessionId) {
      return (
        <AlphaWelcome
          consent={consent}
          onConsentChange={handleConsentToggle}
          onChoose={handleOnboardingChoice}
          alphaStrict={alphaStrict}
        />
      );
    }

    if (view === "tour") {
      if (!tourInitialized) {
        return (
          <Flex align="center" justify="center" minH="100vh" bg={bg}>
            <Spinner size="xl" thickness="4px" color="purple.400" />
          </Flex>
        );
      }
      return (
        <AlphaTour
          steps={tourSteps}
          alphaStrict={alphaStrict}
          onPreview={handleTourPreview}
          onSetTopic={handleTourTopic}
          onRemember={handleTourMemory}
          onFeedback={handleTourFeedback}
          onMetric={handleTourStepMetric}
          onComplete={handleTourComplete}
          onSkip={handleSkipTour}
        />
      );
    }

    return (
      <Flex align="center" justify="center" minH="100vh" bg={bg}>
        <Spinner size="xl" thickness="4px" color="purple.400" />
      </Flex>
    );
  }

  return (
    <Flex direction="column" h="100vh" bg={bg} fontFamily="'Nunito', sans-serif">
      <HStack p={3} bg="pink.500" color="white" boxShadow="0 2px 12px rgba(255, 105, 180, 0.6)">
        <Box fontWeight="bold" fontSize="xl">Lexi Chat</Box>
        {persona && (
          <Box fontSize="sm" color="pink.100" ml={2}>
            mode: {(persona as any).mode ?? "default"}
          </Box>
        )}
        {alphaStrict && (
          <Badge ml={2} colorScheme="purple" variant="subtle">
            alpha strict
          </Badge>
        )}
        <Spacer />
        <Button
          leftIcon={<DownloadIcon />}
          size="sm"
          variant="solid"
          colorScheme="purple"
          onClick={handleDownloadMemory}
          isLoading={downloadingLog}
          mr={2}
        >
          download session
        </Button>
        <IconButton
          aria-label="Toggle dark mode"
          icon={colorMode === "light" ? <MoonIcon /> : <SunIcon />}
          onClick={toggleColorMode}
          variant="ghost"
          color="white"
        />
      </HStack>
      <Box bg={bannerBg} color={bannerColor} px={4} py={2} boxShadow="inset 0 -1px 0 rgba(128, 90, 213, 0.25)">
        <HStack spacing={3}>
          <Text fontWeight="semibold">closed alpha — things reset on logout.</Text>
          <Badge colorScheme={consent ? "green" : "red"} variant="outline">
            consent {consent ? "on" : "off"}
          </Badge>
          <Spacer />
          {alphaStrict && <Text fontSize="sm">heavy features paused for this run</Text>}
        </HStack>
      </Box>

      <Box flex="1" position="relative" overflow="hidden">
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
              alt="Lexi avatar"
              maxW="250px"
              borderRadius="2xl"
              border="2px solid hotpink"
              backdropFilter="blur(4px)"
            />
          </Box>
        )}

        <Box h="100%" overflowY="auto" px={4} pt={4} pb={2} css={{ scrollbarColor: "hotpink transparent", scrollbarWidth: "thin" }}>
          <VStack spacing={4} align="stretch">
            {messages.map((m) => {
              if (typeof m.content !== "string" || m.content.trim().length === 0) return null;
              return (
                <Box
                  key={m.id}
                  alignSelf={
                    m.sender === "user" ? "flex-end" :
                    m.sender === "ai" ? "flex-start" : "center"
                  }
                  bg={
                    m.sender === "user" ? "pink.400" :
                    m.sender === "ai" ? "rgba(100, 100, 100, 0.3)" : "purple.600"
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
