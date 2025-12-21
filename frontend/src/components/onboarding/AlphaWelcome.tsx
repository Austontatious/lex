import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  VStack,
  HStack,
} from "@chakra-ui/react";

export type AlphaWelcomeCopy = {
  tour_text?: string;
  skip_text?: string;
  intro?: string;
};

type AlphaWelcomeProps = {
  copy: AlphaWelcomeCopy | null;
  onYes: () => void;
  onNo: () => void;
  loadingChoice?: "yes" | "no" | null;
};

const AlphaWelcome = ({ copy, onYes, onNo, loadingChoice }: AlphaWelcomeProps) => {
  const tourText =
    copy?.tour_text ||
    copy?.intro ||
    "Hi, I'm Lexi - your emotionally-aware AI companion. I can adapt to you, remember our chats, and stay playful while we talk. Do you want the boring legal version, or should we just get started?";
  const busy = Boolean(loadingChoice);
  const paragraphs = tourText.split(/\n\s*\n/).filter((p) => p.trim().length > 0);

  return (
    <Flex
      className="appShell"
      direction="column"
      align="center"
      justify="center"
      minH="var(--app-dvh)"
      px={6}
      py={10}
    >
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
          {paragraphs.map((para, idx) =>
            idx === 0 ? (
              <Heading key={idx} size="lg" lineHeight={1.2}>
                {para}
              </Heading>
            ) : (
              <Text key={idx} color="gray.600" _dark={{ color: "gray.300" }} lineHeight={1.6}>
                {para}
              </Text>
            )
          )}
          <Text color="gray.500" _dark={{ color: "gray.400" }}>
            Want the boring legal version? Pick yes to see it, or no to skip straight to chatting.
          </Text>
          <HStack spacing={4} justify="flex-start">
            <Button
              colorScheme="purple"
              size="lg"
              onClick={onYes}
              isLoading={loadingChoice === "yes"}
              isDisabled={busy}
            >
              yes - show me the legal stuff
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={onNo}
              isLoading={loadingChoice === "no"}
              isDisabled={busy}
            >
              no - let's just talk
            </Button>
          </HStack>
        </VStack>
      </Box>
    </Flex>
  );
};

export default AlphaWelcome;
