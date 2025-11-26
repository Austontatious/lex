import { useEffect, useState } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  HStack,
  IconButton,
  Input,
  Text,
  Textarea,
  useColorModeValue,
  VStack,
} from "@chakra-ui/react";
import { CloseIcon } from "@chakra-ui/icons";

import { apiFetch } from "../services/api";

type FeedbackModalProps = {
  onClose: () => void;
};

export function FeedbackModal({ onClose }: FeedbackModalProps) {
  const [message, setMessage] = useState("");
  const [email, setEmail] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const surfaceColor = useColorModeValue("white", "gray.900");
  const textColor = useColorModeValue("gray.800", "whiteAlpha.900");
  const helperColor = useColorModeValue("gray.500", "whiteAlpha.700");

  const trimmedMessage = message.trim();

  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose]);

  const handleSubmit = async () => {
    if (!trimmedMessage) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await apiFetch("/feedback", {
        method: "POST",
        body: JSON.stringify({
          message: trimmedMessage,
          email: email.trim() || null,
        }),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "Failed to send feedback");
        throw new Error(text || "Failed to send feedback");
      }
      setDone(true);
    } catch (err) {
      const fallback = err instanceof Error ? err.message : "Something went wrong";
      setError(fallback);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Box
      position="fixed"
      inset={0}
      bg="blackAlpha.700"
      zIndex={2000}
      display="flex"
      alignItems="center"
      justifyContent="center"
      px={4}
      data-testid="feedback-modal"
      role="dialog"
      aria-modal="true"
    >
      <Box
        w="100%"
        maxW="lg"
        bg={surfaceColor}
        color={textColor}
        p={6}
        borderRadius="2xl"
        boxShadow="2xl"
      >
        <HStack justify="space-between" mb={4}>
          <Text fontSize="lg" fontWeight="semibold">
            Send feedback
          </Text>
          <IconButton
            aria-label="Close feedback"
            icon={<CloseIcon boxSize={3} />}
            size="sm"
            variant="ghost"
            onClick={onClose}
          />
        </HStack>

        {done ? (
          <VStack spacing={4} align="stretch">
            <Text fontSize="sm">Thank you — your feedback has been saved.</Text>
            <Button colorScheme="pink" onClick={onClose}>
              Close
            </Button>
          </VStack>
        ) : (
          <VStack spacing={4} align="stretch">
            <FormControl>
              <FormLabel fontSize="xs" textTransform="uppercase" letterSpacing="wider">
                Tell us anything about your experience
              </FormLabel>
              <Textarea
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Bug reports, ideas, what you liked or didn’t like..."
                minH="10rem"
                maxH="20rem"
                resize="none"
                borderRadius="xl"
                borderWidth={1}
                borderColor="whiteAlpha.400"
                bg="blackAlpha.200"
                _focus={{ borderColor: "pink.300", boxShadow: "0 0 0 1px rgba(236, 72, 153, 0.4)" }}
                overflowY="auto"
              />
              <Text fontSize="xs" mt={1} color={helperColor}>
                This box scrolls — write as much as you want.
              </Text>
            </FormControl>

            <FormControl>
              <FormLabel fontSize="xs" textTransform="uppercase" letterSpacing="wider">
                Email (optional)
              </FormLabel>
              <Input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="Only if you want us to follow up"
                borderRadius="xl"
                borderWidth={1}
                borderColor="whiteAlpha.400"
                bg="blackAlpha.200"
                _focus={{ borderColor: "pink.300", boxShadow: "0 0 0 1px rgba(236, 72, 153, 0.4)" }}
              />
            </FormControl>

            {error && (
              <Text fontSize="xs" color="red.400">
                {error}
              </Text>
            )}

            <HStack justify="flex-end">
              <Button variant="ghost" onClick={onClose} isDisabled={submitting}>
                Cancel
              </Button>
              <Button
                colorScheme="pink"
                onClick={handleSubmit}
                isDisabled={!trimmedMessage || submitting}
              >
                {submitting ? "Saving..." : "Submit feedback"}
              </Button>
            </HStack>
          </VStack>
        )}
      </Box>
    </Box>
  );
}

