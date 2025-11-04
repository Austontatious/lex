import { useEffect, useMemo, useState } from "react";
import { Box, Text, VStack } from "@chakra-ui/react";

type EmotionPulseProps = {
  axes?: string[];
  intervalMs?: number;
};

const DEFAULT_AXES = ["warmth", "energy", "curiosity", "confidence", "playfulness"];

const EmotionPulse = ({ axes, intervalMs = 1600 }: EmotionPulseProps) => {
  const labels = useMemo(() => (axes && axes.length ? axes : DEFAULT_AXES), [axes]);
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    if (!labels.length) return;
    const timer = setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % labels.length);
    }, intervalMs);
    return () => clearInterval(timer);
  }, [labels, intervalMs]);

  return (
    <VStack align="stretch" spacing={3} w="100%">
      {labels.map((axis, idx) => {
        const isActive = idx === activeIndex;
        return (
          <Box
            key={axis}
            p={3}
            borderRadius="md"
            bg={isActive ? "purple.50" : "blackAlpha.50"}
            _dark={{ bg: isActive ? "purple.900" : "whiteAlpha.100" }}
          >
            <Text fontWeight="semibold" textTransform="uppercase" fontSize="xs" color="gray.500">
              {axis}
            </Text>
            <Box
              mt={2}
              h="6px"
              borderRadius="full"
              bg="gray.200"
              _dark={{ bg: "whiteAlpha.200" }}
              overflow="hidden"
            >
              <Box
                h="100%"
                bgGradient="linear(to-r, purple.400, pink.400)"
                width={isActive ? "96%" : "24%"}
                transition="width 0.6s ease"
              />
            </Box>
            <Text mt={2} fontSize="sm" color="gray.600" _dark={{ color: "gray.300" }}>
              {isActive ? "lexi is vibing here right now" : "watch for little pulses as the chat flows"}
            </Text>
          </Box>
        );
      })}
    </VStack>
  );
};

export default EmotionPulse;
