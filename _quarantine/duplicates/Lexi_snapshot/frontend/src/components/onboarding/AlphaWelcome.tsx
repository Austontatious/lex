import { useState } from "react";
import {
  Badge,
  Box,
  Button,
  Flex,
  Heading,
  HStack,
  Switch,
  Text,
  Tooltip,
  VStack,
} from "@chakra-ui/react";

type WelcomeStage = "intro" | "disclaimer";

type AlphaWelcomeProps = {
  consent: boolean;
  onConsentChange: (value: boolean) => void;
  onChoose: (choice: "tour" | "chat") => void;
  alphaStrict: boolean;
};

const AlphaWelcome = ({ consent, onConsentChange, onChoose, alphaStrict }: AlphaWelcomeProps) => {
  const [stage, setStage] = useState<WelcomeStage>("intro");

  return (
    <Flex direction="column" align="center" justify="center" minH="100vh" px={6} py={10}>
      <Box
        maxW="600px"
        w="100%"
        bg="whiteAlpha.900"
        _dark={{ bg: "gray.800" }}
        borderRadius="2xl"
        boxShadow="xl"
        p={{ base: 6, md: 10 }}
      >
        <VStack spacing={6} align="stretch">
          <Heading size="lg">hey, i’m lexi 👋</Heading>
          <Text fontSize="lg" lineHeight={1.6}>
            your companion, coach, co-conspirator… whatever you need 😉 want the 2-minute tour,
            or should we just talk?
          </Text>

          {alphaStrict && (
            <Badge alignSelf="flex-start" colorScheme="purple">
              alpha strict mode — heavy features are on standby
            </Badge>
          )}

          <Box>
            <Text fontWeight="semibold" mb={2}>
              anonymized session logs ok?
            </Text>
            <HStack spacing={4}>
              <Switch
                isChecked={consent}
                onChange={(e) => onConsentChange(e.target.checked)}
                colorScheme="purple"
              />
              <Tooltip
                label="session events are anonymized and archived for the dev team — toggle off to redact content."
                hasArrow
              >
                <Text fontSize="sm" color="gray.500">
                  keeps my diary for the boss (default on)
                </Text>
              </Tooltip>
            </HStack>
          </Box>

          {stage === "intro" ? (
            <HStack spacing={4} justify="flex-start">
              <Button colorScheme="purple" size="lg" onClick={() => onChoose("tour")}>
                give me the tour
              </Button>
              <Button variant="outline" size="lg" onClick={() => setStage("disclaimer")}>
                let’s just talk
              </Button>
            </HStack>
          ) : (
            <VStack spacing={4} align="stretch">
              <Text lineHeight={1.6}>
                done. i can riff on almost anything. heads-up: this alpha forgets everything when you log out.
                i do keep an anonymized session diary for… “quality time” with my creator. only the boss sees it.
                he’s allergic to reading, so your secrets are safe-ish. proceed? 🗝️
              </Text>
              <HStack spacing={4}>
                <Button colorScheme="purple" size="lg" onClick={() => onChoose("chat")}>
                  let’s chat
                </Button>
                <Button variant="outline" size="lg" onClick={() => onChoose("tour")}>
                  show me the tour anyway
                </Button>
              </HStack>
            </VStack>
          )}
        </VStack>
      </Box>
    </Flex>
  );
};

export default AlphaWelcome;
