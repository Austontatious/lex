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
  intro?: string;
  disclaimer_short?: string;
  disclaimer?: string;
  disclaimer_full?: string;
  steps?: Array<{ id: string; title: string; copy: string }>;
};

type AlphaWelcomeProps = {
  copy: AlphaWelcomeCopy | null;
  onChoose: (choice: "tour" | "skip") => void;
  loadingChoice?: "tour" | "skip" | null;
};

const AlphaWelcome = ({ copy, onChoose, loadingChoice }: AlphaWelcomeProps) => {
  const intro =
    copy?.intro ||
    "Hey there 😘 I’m Lexi — your companion, coach, confidant… whatever you need. Want the tour or should we just talk?";
  const busy = Boolean(loadingChoice);

  return (
    <Flex direction="column" align="center" justify="center" minH="100vh" px={6} py={10}>
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
          <Heading size="lg">{intro}</Heading>
          {copy?.disclaimer_short && (
            <Text color="gray.500" lineHeight={1.5}>
              {copy.disclaimer_short}
            </Text>
          )}
          <Text color="gray.400">
            Don’t worry about the “right” answer—pick whichever feels good. I’ll roll with it.
          </Text>
          <HStack spacing={4} justify="flex-start">
            <Button
              colorScheme="purple"
              size="lg"
              onClick={() => onChoose("tour")}
              isLoading={loadingChoice === "tour"}
              isDisabled={busy}
            >
              give me the tour
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={() => onChoose("skip")}
              isLoading={loadingChoice === "skip"}
              isDisabled={busy}
            >
              let’s just talk
            </Button>
          </HStack>
        </VStack>
      </Box>
    </Flex>
  );
};

export default AlphaWelcome;
