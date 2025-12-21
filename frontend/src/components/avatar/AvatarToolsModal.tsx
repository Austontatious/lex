import { useEffect, useState } from "react";
import {
  Box,
  Button,
  FormControl,
  FormLabel,
  HStack,
  IconButton,
  Image,
  Select,
  SimpleGrid,
  Spinner,
  Text,
  Textarea,
  VStack,
  useColorModeValue,
  useToast,
} from "@chakra-ui/react";
import { CloseIcon } from "@chakra-ui/icons";

import {
  AvatarGenMode,
  LexiverseStyle,
  requestAvatarGeneration,
  fetchAvatarGenerationStatus,
} from "../../services/api";
import { refreshAvatar } from "../../lib/refreshAvatar";

type HairKey = "brunette" | "blonde" | "redhead" | "black";
type HairStyleKey = "straight" | "wavy" | "curly" | "updo";
type SkinKey = "fair" | "light_medium" | "olive" | "tan" | "deep";
type EyeKey = "brown" | "hazel" | "green" | "blue";
type OutfitKey = "lbd" | "lounge" | "casual" | "business" | "sporty";
type VibeKey = "soft" | "playful" | "elegant" | "confident" | "sultry";

type AvatarFormState = {
  hair: HairKey;
  hairStyle: HairStyleKey;
  skinTone: SkinKey;
  eyes: EyeKey;
  outfit: OutfitKey;
  vibe: VibeKey;
  lexiverseStyle: LexiverseStyle;
  extraDetails: string;
};

const DEFAULT_FORM_STATE: AvatarFormState = {
  hair: "brunette",
  hairStyle: "wavy",
  skinTone: "light_medium",
  eyes: "hazel",
  outfit: "lbd",
  vibe: "soft",
  lexiverseStyle: "promo",
  extraDetails: "",
};

type AvatarToolsModalProps = {
  isOpen: boolean;
  onClose: () => void;
  currentAvatarUrl?: string | null;
  onAvatarUpdated?: (url: string) => void;
  onGenerationStart?: () => void;
  onGenerationEnd?: () => void;
};

export function AvatarToolsModal({
  isOpen,
  onClose,
  currentAvatarUrl,
  onAvatarUpdated,
  onGenerationStart,
  onGenerationEnd,
}: AvatarToolsModalProps) {
  const toast = useToast();
  const [formState, setFormState] = useState<AvatarFormState>(DEFAULT_FORM_STATE);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const surface = useColorModeValue("white", "gray.900");
  const textColor = useColorModeValue("gray.800", "whiteAlpha.900");
  const helperColor = useColorModeValue("gray.500", "whiteAlpha.700");
  const badgeBg = useColorModeValue("pink.50", "whiteAlpha.200");

  useEffect(() => {
    if (!isOpen) {
      setFormState(DEFAULT_FORM_STATE);
      setIsSubmitting(false);
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [isOpen, onClose]);

  const updateField = <K extends keyof AvatarFormState>(key: K, value: AvatarFormState[K]) => {
    setFormState((prev) => ({ ...prev, [key]: value }));
  };

  const handleGenerate = async (mode: AvatarGenMode) => {
    if (isSubmitting) return;
    setIsSubmitting(true);
    onGenerationStart?.();
    try {
      const payload = {
        sd_mode: mode,
        lexiverse_style: formState.lexiverseStyle,
        traits: {
          hair: formState.hair,
          hair_style: formState.hairStyle,
          skin_tone: formState.skinTone,
          eyes: formState.eyes,
          outfit: formState.outfit,
          vibe: formState.vibe,
        },
        extra_details: formState.extraDetails || undefined,
        // keep strength in case img2img is revived later
        strength: mode === "img2img" ? 0.6 : undefined,
      } as const;
      const res = await requestAvatarGeneration(payload);
      let candidate: string | null =
        res.avatar_url || res.url || res.image || res.image_url || null;

      if (!candidate && res.status === "running" && res.prompt_id) {
        candidate = await pollGenerationStatus(res.prompt_id);
      }

      if (!candidate) {
        candidate = await pollPersonaAvatar();
      }

      if (!candidate) {
        throw new Error(res.error || "No avatar URL returned");
      }

      onAvatarUpdated?.(candidate);
      toast({
        status: "success",
        title: mode === "img2img" ? "Updated your look" : "New Lexi coming up",
        duration: 2500,
        isClosable: true,
      });
      onClose();
    } catch (err) {
      console.error(err);
      toast({
        status: "error",
        title: "Avatar generation failed",
        description: "Please try again.",
        duration: 3500,
        isClosable: true,
      });
    } finally {
      setIsSubmitting(false);
      onGenerationEnd?.();
    }
  };

  const pollGenerationStatus = async (promptId: string): Promise<string | null> => {
    const start = Date.now();
    const timeoutMs = 180_000;
    let lastError: unknown = null;
    while (Date.now() - start < timeoutMs) {
      try {
        const status = await fetchAvatarGenerationStatus(promptId);
        if (status.status === "done") {
          const url = status.avatar_url || status.url || status.image || status.image_url || "";
          if (url) return url;
        }
        if (status.status === "error") {
          lastError = new Error(status.error || "generation error");
          break;
        }
      } catch (err) {
        lastError = err;
      }
      await new Promise((resolve) => setTimeout(resolve, 4000));
    }
    if (lastError) {
      console.warn("avatar status polling ended with error", lastError);
    }
    return null;
  };

  const pollPersonaAvatar = async (): Promise<string | null> => {
    const start = Date.now();
    const timeoutMs = 180_000;
    while (Date.now() - start < timeoutMs) {
      try {
        const refreshed = await refreshAvatar();
        if (refreshed) {
          return refreshed;
        }
      } catch (err) {
        console.warn("persona avatar poll failed", err);
      }
      await new Promise((resolve) => setTimeout(resolve, 5000));
    }
    return null;
  };

  if (!isOpen) return null;

  return (
    <Box
      position="fixed"
      inset={0}
      bg="blackAlpha.700"
      zIndex={2100}
      display="flex"
      alignItems="center"
      justifyContent="center"
      px={{ base: 4, md: 6 }}
      pt="calc(var(--safe-top) + 12px)"
      pb="calc(var(--safe-bottom) + 12px)"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <Box
        w="100%"
        maxW="1100px"
        maxH="calc(var(--app-dvh) - var(--safe-top) - var(--safe-bottom) - 24px)"
        bg={surface}
        color={textColor}
        p={{ base: 5, md: 7 }}
        borderRadius="2xl"
        boxShadow="2xl"
        position="relative"
        overflow="hidden"
        display="flex"
        flexDirection="column"
        onClick={(event) => event.stopPropagation()}
      >
        <Box
          position="absolute"
          inset={0}
          bgGradient="linear(to-br, rgba(255,105,180,0.15), rgba(118,75,255,0.12))"
          pointerEvents="none"
        />
        <Box position="relative" zIndex={1} display="flex" flexDirection="column" flex="1" minH="0">
          <HStack justify="space-between" mb={4} flexShrink={0}>
            <Box>
              <Text fontWeight="bold" fontSize="xl">
                Avatar tools
              </Text>
              <Text fontSize="sm" color={helperColor}>
                Dress Lexi up with hair, skin, outfit, and vibe tweaks—no model knobs needed.
              </Text>
            </Box>
            <IconButton
              aria-label="Close avatar tools"
              icon={<CloseIcon boxSize={3} />}
              size="lg"
              minW={11}
              minH={11}
              variant="ghost"
              onClick={onClose}
            />
          </HStack>

          <Box
            flex="1"
            minH="0"
            overflowY="auto"
            pb="calc(var(--safe-bottom) + 12px)"
            css={{ WebkitOverflowScrolling: "touch" }}
          >
            <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
              <VStack align="stretch" spacing={4}>
                <HStack spacing={3}>
                  <FormControl>
                    <FormLabel fontSize="sm">Hair</FormLabel>
                    <Select
                      size="lg"
                      value={formState.hair}
                      onChange={(e) => updateField("hair", e.target.value as HairKey)}
                    >
                      <option value="brunette">Dark brunette</option>
                      <option value="blonde">Blonde</option>
                      <option value="redhead">Redhead</option>
                      <option value="black">Black hair</option>
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel fontSize="sm">Style</FormLabel>
                    <Select
                      size="lg"
                      value={formState.hairStyle}
                      onChange={(e) => updateField("hairStyle", e.target.value as HairStyleKey)}
                    >
                      <option value="straight">Straight</option>
                      <option value="wavy">Wavy</option>
                      <option value="curly">Curly</option>
                      <option value="updo">Updo</option>
                    </Select>
                  </FormControl>
                </HStack>

                <SimpleGrid columns={{ base: 1, sm: 2 }} spacing={3}>
                  <FormControl>
                    <FormLabel fontSize="sm">Skin tone</FormLabel>
                    <Select
                      size="lg"
                      value={formState.skinTone}
                      onChange={(e) => updateField("skinTone", e.target.value as SkinKey)}
                    >
                      <option value="fair">Fair</option>
                      <option value="light_medium">Light/medium</option>
                      <option value="olive">Olive</option>
                      <option value="tan">Tan</option>
                      <option value="deep">Deep</option>
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel fontSize="sm">Eyes</FormLabel>
                    <Select
                      size="lg"
                      value={formState.eyes}
                      onChange={(e) => updateField("eyes", e.target.value as EyeKey)}
                    >
                      <option value="brown">Brown</option>
                      <option value="hazel">Hazel</option>
                      <option value="green">Green</option>
                      <option value="blue">Blue</option>
                    </Select>
                  </FormControl>
                </SimpleGrid>

                <SimpleGrid columns={{ base: 1, sm: 2 }} spacing={3}>
                  <FormControl>
                    <FormLabel fontSize="sm">Outfit</FormLabel>
                    <Select
                      size="lg"
                      value={formState.outfit}
                      onChange={(e) => updateField("outfit", e.target.value as OutfitKey)}
                    >
                      <option value="lbd">Little black dress</option>
                      <option value="lounge">Cozy loungewear</option>
                      <option value="casual">Casual jeans and tee</option>
                      <option value="business">Business chic</option>
                      <option value="sporty">Sporty / gym fit</option>
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel fontSize="sm">Vibe / mood</FormLabel>
                    <Select
                      size="lg"
                      value={formState.vibe}
                      onChange={(e) => updateField("vibe", e.target.value as VibeKey)}
                    >
                      <option value="soft">Soft & cuddly</option>
                      <option value="playful">Playful</option>
                      <option value="elegant">Elegant</option>
                      <option value="confident">Confident</option>
                      <option value="sultry">Sultry</option>
                    </Select>
                  </FormControl>
                </SimpleGrid>

                <FormControl>
                  <FormLabel fontSize="sm">Anything specific you want to add?</FormLabel>
                  <Textarea
                    size="lg"
                    value={formState.extraDetails}
                    onChange={(e) => updateField("extraDetails", e.target.value)}
                    placeholder="e.g., soft pink nails, studio lighting, a hint of freckles…"
                    minH="80px"
                    maxH="160px"
                    resize="vertical"
                  />
                </FormControl>

                {/* Lexiverse is always on; promo intensity locked in */}
              </VStack>

              <VStack align="stretch" spacing={4}>
                <Box
                  borderWidth="1px"
                  borderRadius="xl"
                  overflow="hidden"
                  boxShadow="sm"
                  bg="blackAlpha.50"
                >
                  {currentAvatarUrl ? (
                    <Image
                      src={currentAvatarUrl}
                      alt="Current avatar"
                      w="100%"
                      h="100%"
                      loading="lazy"
                      decoding="async"
                      objectFit="cover"
                      aspectRatio="3 / 4"
                      sizes="(max-width: 768px) 100vw, 320px"
                    />
                  ) : (
                    <Box p={6}>
                      <Text color={helperColor} textAlign="center">
                        Your current avatar will show here.
                      </Text>
                    </Box>
                  )}
                </Box>
                <Box>
                  <VStack align="stretch" spacing={3}>
                    <Button
                      colorScheme="pink"
                      size="lg"
                      onClick={() => void handleGenerate("txt2img")}
                      isDisabled={isSubmitting}
                    >
                      {isSubmitting ? <Spinner size="sm" mr={2} /> : null}
                      Try something new
                    </Button>
                    <Text fontSize="xs" color={helperColor} ml={1}>
                      Fresh Lexiverse look with your selected traits.
                    </Text>
                  </VStack>
                </Box>

                {isSubmitting && (
                  <HStack spacing={2} color={helperColor}>
                    <Spinner size="sm" color="pink.400" />
                    <Text fontSize="sm">Generating your new look…</Text>
                  </HStack>
                )}
              </VStack>
            </SimpleGrid>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

export default AvatarToolsModal;
