import { useMemo, useState } from "react";
import { Box, Button, Flex, Heading, Progress, Text, VStack } from "@chakra-ui/react";

export type TourStep = {
  id: string;
  title: string;
  copy: string;
};

type AlphaTourProps = {
  steps: TourStep[];
  onComplete: () => void;
  onSkip: () => void;
};

const FALLBACK_STEPS: TourStep[] = [
  {
    id: "preview",
    title: "What I am",
    copy: "I’m an AI companion that mixes practical help with a heavy vibe-check.",
  },
];

const AlphaTour = ({ steps, onComplete, onSkip }: AlphaTourProps) => {
  const safeSteps = useMemo(() => (steps.length ? steps : FALLBACK_STEPS), [steps]);
  const [index, setIndex] = useState(0);
  const current = safeSteps[Math.min(index, safeSteps.length - 1)];
  const progress = ((index + 1) / safeSteps.length) * 100;

  const goNext = () => {
    if (index >= safeSteps.length - 1) {
      onComplete();
      return;
    }
    setIndex((prev) => Math.min(prev + 1, safeSteps.length - 1));
  };

  if (!current) {
    return null;
  }

  return (
    <Flex direction="column" minH="100vh" px={6} py={10} align="center" justify="center">
      <Box
        w="100%"
        maxW="640px"
        bg="whiteAlpha.900"
        _dark={{ bg: "gray.800" }}
        borderRadius="3xl"
        boxShadow="2xl"
        p={{ base: 6, md: 10 }}
      >
        <Flex justify="space-between" align="center" mb={6}>
          <Heading size="lg">
            tour — step {index + 1} / {safeSteps.length}
          </Heading>
          <Button variant="ghost" onClick={onSkip}>
            skip
          </Button>
        </Flex>
        <Progress value={progress} colorScheme="pink" borderRadius="full" mb={6} />
        <VStack align="stretch" spacing={4}>
          <Text fontSize="xl" fontWeight="bold">
            {current.title}
          </Text>
          <Text fontSize="lg" color="gray.600" _dark={{ color: "gray.100" }}>
            {current.copy}
          </Text>
          <Button colorScheme="pink" onClick={goNext}>
            {index >= safeSteps.length - 1 ? "Let’s chat" : "Next"}
          </Button>
        </VStack>
      </Box>
    </Flex>
  );
};

export default AlphaTour;
