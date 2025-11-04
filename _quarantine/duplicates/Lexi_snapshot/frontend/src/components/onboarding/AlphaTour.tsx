import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  AlertIcon,
  Box,
  Button,
  Flex,
  Heading,
  Image,
  Input,
  Link,
  Spinner,
  Text,
  Textarea,
  VStack,
} from "@chakra-ui/react";
import EmotionPulse from "./EmotionPulse";

export type TourStep = {
  slug: string;
  prompt: string;
  narration: string;
};

type AlphaTourProps = {
  steps: TourStep[];
  alphaStrict: boolean;
  onPreview: (prompt: string) => Promise<string>;
  onSetTopic: (topic: string) => Promise<void>;
  onRemember: (note: string) => Promise<void>;
  onFeedback: (helpful: boolean, comment?: string) => Promise<void>;
  onMetric?: (slug: string) => Promise<void> | void;
  onComplete: () => void;
  onSkip: () => void;
};

type StepState = "idle" | "pending" | "done";

const AlphaTour = ({
  steps,
  alphaStrict,
  onPreview,
  onSetTopic,
  onRemember,
  onFeedback,
  onMetric,
  onComplete,
  onSkip,
}: AlphaTourProps) => {
  const safeSteps = useMemo(() => (steps.length ? steps : DEFAULT_STEPS), [steps]);
  const [index, setIndex] = useState(0);
  const [input, setInput] = useState("");
  const [note, setNote] = useState("");
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [state, setState] = useState<StepState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [feedbackGiven, setFeedbackGiven] = useState(false);

  const current = safeSteps[Math.min(index, safeSteps.length - 1)];

  useEffect(() => {
    setState("idle");
    setError(null);
    setInput("");
    setNote("");
    setPreviewUrl(null);
    if (current && onMetric) {
      void Promise.resolve(onMetric(current.slug)).catch(() => {});
    }
  }, [index, current, onMetric]);

  const goNext = () => setIndex((prev) => Math.min(prev + 1, safeSteps.length - 1));

  const handlePreview = async () => {
    if (!input.trim()) {
      setError("give me a vibe first!");
      return;
    }
    setError(null);
    setState("pending");
    try {
      const url = await onPreview(input.trim());
      setPreviewUrl(url);
      setState("done");
    } catch (err: any) {
      setError(err?.message || "preview failed — try again?");
      setState("idle");
    }
  };

  const handleTopic = async () => {
    if (!input.trim()) {
      setError("pick a topic so i can keep it in mind.");
      return;
    }
    setError(null);
    setState("pending");
    try {
      await onSetTopic(input.trim());
      setState("done");
      goNext();
    } catch (err: any) {
      setError(err?.message || "couldn’t set the topic, try again?");
      setState("idle");
    }
  };

  const handleRemember = async () => {
    if (!note.trim()) {
      setError("tell me at least one thing to keep for this session.");
      return;
    }
    setError(null);
    setState("pending");
    try {
      await onRemember(note.trim());
      setState("done");
      goNext();
    } catch (err: any) {
      setError(err?.message || "memory saver hiccuped — try again?");
      setState("idle");
    }
  };

  const handleFeedback = async (helpful: boolean) => {
    if (feedbackGiven) {
      onComplete();
      return;
    }
    setFeedbackGiven(true);
    try {
      await onFeedback(helpful);
    } catch {
      // feedback is best-effort
    } finally {
      onComplete();
    }
  };

  const renderBody = () => {
    if (!current) return null;
    switch (current.slug) {
      case "avatar_preview":
        return (
          <VStack align="stretch" spacing={5}>
            <Text fontSize="xl">{current.narration}</Text>
            {alphaStrict && (
              <Alert status="info" borderRadius="md">
                <AlertIcon />
                alpha-strict mode: you’ll see a placeholder preview instead of a full render.
              </Alert>
            )}
            <Input
              placeholder="cozy cyberpunk librarian?"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              size="lg"
            />
            <Button colorScheme="purple" onClick={handlePreview} isLoading={state === "pending"}>
              run a tiny preview
            </Button>
            {error && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                {error}
              </Alert>
            )}
            {state === "done" && previewUrl && (
              <VStack spacing={3}>
                <Image
                  src={previewUrl}
                  alt="avatar preview"
                  borderRadius="lg"
                  boxShadow="lg"
                  maxH="280px"
                  objectFit="cover"
                />
                <Button variant="ghost" onClick={goNext}>
                  looks good — what’s next?
                </Button>
              </VStack>
            )}
          </VStack>
        );
      case "now_topic":
        return (
          <VStack align="stretch" spacing={5}>
            <Text fontSize="xl">{current.narration}</Text>
            <Input
              placeholder="e.g., ‘motivation hacks’ or ‘calmer mornings’"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              size="lg"
            />
            <Button colorScheme="purple" onClick={handleTopic} isLoading={state === "pending"}>
              lock it in
            </Button>
            {error && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                {error}
              </Alert>
            )}
          </VStack>
        );
      case "emotion_axes":
        return (
          <VStack align="stretch" spacing={5}>
            <Text fontSize="xl">{current.narration}</Text>
            <EmotionPulse />
            <Button colorScheme="purple" onClick={goNext}>
              got it — continue
            </Button>
          </VStack>
        );
      case "memory_explainer":
        return (
          <VStack align="stretch" spacing={5}>
            <Text fontSize="xl">{current.narration}</Text>
            <Textarea
              placeholder="tell me something personal to weave into this session…"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              size="lg"
              rows={4}
            />
            <Button colorScheme="purple" onClick={handleRemember} isLoading={state === "pending"}>
              save it for this session
            </Button>
            {error && (
              <Alert status="error" borderRadius="md">
                <AlertIcon />
                {error}
              </Alert>
            )}
          </VStack>
        );
      case "wrap":
        return (
          <VStack align="stretch" spacing={6}>
            <Text fontSize="xl">{current.narration}</Text>
            <Flex gap={4} wrap="wrap">
              <Button colorScheme="purple" size="lg" onClick={() => handleFeedback(true)}>
                let’s chat
              </Button>
              <Button variant="outline" size="lg" onClick={() => handleFeedback(false)}>
                maybe later
              </Button>
            </Flex>
            <Text fontSize="sm" color="gray.500">
              want to revisit the instructions?{" "}
              <Link color="purple.400" onClick={() => setIndex(0)}>
                replay the tour
              </Link>
            </Text>
          </VStack>
        );
      default:
        return (
          <VStack align="stretch" spacing={4}>
            <Text fontSize="xl">{current.narration}</Text>
            <Button colorScheme="purple" onClick={goNext}>
              continue
            </Button>
          </VStack>
        );
    }
  };

  if (!current) {
    return (
      <Flex align="center" justify="center" minH="200px">
        <Spinner size="lg" />
      </Flex>
    );
  }

  return (
    <Flex direction="column" minH="100vh" px={6} py={10} align="center" justify="center">
      <Box
        w="100%"
        maxW="720px"
        bg="whiteAlpha.900"
        _dark={{ bg: "gray.800" }}
        borderRadius="3xl"
        boxShadow="2xl"
        p={{ base: 6, md: 10 }}
      >
        <Flex justify="space-between" align="center" mb={6}>
          <Heading size="lg">alpha tour — step {index + 1} / {safeSteps.length}</Heading>
          <Button variant="ghost" onClick={onSkip}>
            skip tour
          </Button>
        </Flex>
        <Text fontSize="md" color="gray.500" mb={4}>
          {current.prompt}
        </Text>
        {renderBody()}
      </Box>
    </Flex>
  );
};

const DEFAULT_STEPS: TourStep[] = [
  {
    slug: "intro",
    prompt: "describe a vibe, i’ll sketch a look.",
    narration: "awesome. we’ll do a quick spin: avatar vibes → ‘now’ topic → emotions → memory. ready?",
  },
];

export default AlphaTour;
