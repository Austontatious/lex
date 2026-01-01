import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  SimpleGrid,
  Spacer,
  Spinner,
  Stat,
  StatHelpText,
  StatLabel,
  StatNumber,
  Text,
  useColorModeValue,
} from "@chakra-ui/react";
import { fetchAnalyticsSummary, type AnalyticsSummary } from "../services/api";

const REFRESH_INTERVAL_MS = 5000;

export default function AnalyticsDashboard() {
  const cardBg = useColorModeValue("rgba(255, 255, 255, 0.88)", "rgba(16, 4, 20, 0.72)");
  const cardBorder = useColorModeValue("rgba(255, 255, 255, 0.6)", "rgba(255, 255, 255, 0.08)");
  const textMuted = useColorModeValue("gray.600", "gray.300");
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cards = useMemo(
    () =>
      summary
        ? [
            {
              label: "Concurrent now",
              value: summary.concurrent_now,
              help: "Active in the last 120s",
            },
            {
              label: "Peak concurrent today",
              value: summary.peak_concurrent_today,
              help: "Highest since midnight CT",
            },
            {
              label: "Unique visitors today",
              value: summary.unique_today,
              help: "Distinct visitor IDs today",
            },
            {
              label: "Unique visitors all time",
              value: summary.unique_all_time,
              help: "Distinct visitor IDs overall",
            },
          ]
        : [],
    [summary]
  );

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAnalyticsSummary();
      setSummary(data);
      setLastUpdated(new Date());
    } catch (err) {
      console.warn("analytics summary fetch failed", err);
      setError("Unable to load analytics right now.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const tick = async () => {
      if (!mounted) return;
      await refresh();
    };
    void tick();
    const interval = window.setInterval(() => {
      void tick();
    }, REFRESH_INTERVAL_MS);
    return () => {
      mounted = false;
      window.clearInterval(interval);
    };
  }, [refresh]);

  return (
    <Box minH="var(--app-dvh)" px={{ base: 5, md: 12 }} py={{ base: 10, md: 14 }}>
      <Flex align="center" gap={4} mb={{ base: 8, md: 10 }} wrap="wrap">
        <Box>
          <Heading size="lg">Lexi Analytics</Heading>
          <Text color={textMuted} mt={2} maxW="520px">
            Live snapshot of visitors and concurrency. Updates every 5 seconds.
          </Text>
        </Box>
        <Spacer />
        <HStack spacing={3} align="center">
          <Badge colorScheme="pink" variant="solid">
            Live
          </Badge>
          <Button size="sm" onClick={refresh} isLoading={loading}>
            Refresh now
          </Button>
        </HStack>
      </Flex>

      <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6} mb={8}>
        {cards.map((card) => (
          <Box
            key={card.label}
            bg={cardBg}
            border="1px solid"
            borderColor={cardBorder}
            borderRadius="24px"
            px={{ base: 6, md: 8 }}
            py={{ base: 6, md: 7 }}
            boxShadow="0 18px 40px rgba(0,0,0,0.18)"
            backdropFilter="blur(12px)"
          >
            <Stat>
              <StatLabel fontSize="sm" color={textMuted} textTransform="uppercase">
                {card.label}
              </StatLabel>
              <StatNumber fontSize={{ base: "3xl", md: "4xl" }} mt={2}>
                {card.value}
              </StatNumber>
              <StatHelpText color={textMuted}>{card.help}</StatHelpText>
            </Stat>
          </Box>
        ))}
      </SimpleGrid>

      <Box
        bg={cardBg}
        border="1px solid"
        borderColor={cardBorder}
        borderRadius="24px"
        px={{ base: 6, md: 8 }}
        py={{ base: 6, md: 7 }}
        boxShadow="0 18px 40px rgba(0,0,0,0.16)"
        backdropFilter="blur(12px)"
      >
        <Flex align="center" gap={4} wrap="wrap">
          <Box>
            <Text fontSize="sm" color={textMuted} textTransform="uppercase">
              Current day (America/Chicago)
            </Text>
            <Heading size="md" mt={2}>
              {summary?.day ?? "-"}
            </Heading>
          </Box>
          <Spacer />
          <Box textAlign={{ base: "left", md: "right" }}>
            {loading && !summary ? (
              <HStack spacing={3}>
                <Spinner size="sm" />
                <Text color={textMuted}>Loading metrics...</Text>
              </HStack>
            ) : (
              <Text color={textMuted}>
                Last updated: {lastUpdated ? lastUpdated.toLocaleTimeString() : "-"}
              </Text>
            )}
            {error && (
              <Text color="red.300" fontSize="sm" mt={2}>
                {error}
              </Text>
            )}
          </Box>
        </Flex>
      </Box>
    </Box>
  );
}
