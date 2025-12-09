import { useEffect, useState } from "react";
import {
  Badge,
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

const BASE_AVATAR_PROMPT =
  "cinematic full-body portrait of Lexi, professional instagram influencer aesthetic, neutral studio backdrop, ultra flattering editorial lighting, soft volumetric rim light, golden hour glow effect, dewy realistic skin texture, confident subtle expression, tasteful suggestive pose, shallow depth of field, high detail, cinematic color grading, hyperrealistic, instagram editorial aesthetic";

const DEFAULT_NEGATIVE_PROMPT =
  "low quality, blurry, distorted face, extra limbs, deformed hands, watermark, logo, text, duplicate body parts";

const HAIR_MAP = {
  keep: "soft natural hair color",
  brunette: "rich dark brunette",
  blonde: "soft golden blonde",
  redhead: "warm copper red",
  black: "glossy jet black",
};

const HAIR_STYLE_MAP = {
  keep: "loosely styled",
  straight: "sleek straight",
  wavy: "soft beachy waves",
  curly: "big soft curls",
  updo: "messy romantic updo",
};

const SKIN_TONE_MAP = {
  keep: "naturally glowing skin",
  fair: "fair porcelain skin",
  light_medium: "light peachy skin",
  olive: "warm olive-toned skin",
  tan: "sun-kissed medium tone skin",
  deep: "rich deep brown skin",
};

const EYE_MAP = {
  keep: "expressive eyes",
  brown: "warm brown eyes",
  hazel: "gold-flecked hazel eyes",
  green: "vibrant green eyes",
  blue: "bright blue eyes",
};

const OUTFIT_MAP = {
  keep: "a simple, form-flattering outfit that shows off her figure",
  lbd: "a sleek, curve-hugging little black dress, subtle but glamorous",
  lounge: "soft, fitted loungewear, cozy knit top and snug lounge pants",
  casual: "a fitted tee and high-waisted skinny jeans, elevated casual",
  business: "a tailored blazer over a fitted top and slim trousers, polished and sharp",
  sporty: "a fitted sports bra and high-waisted leggings, gym-ready and athletic",
};

const VIBE_MAP = {
  soft: "soft relaxed body language, warm gentle smile, inviting eyes, cozy approachable energy",
  playful: "sparkling eyes, playful smirk, slightly tilted head, teasing fun energy",
  elegant: "poised posture, graceful hands, serene expression, refined elegant presence",
  confident:
    "strong stance, shoulders back, direct eye contact, subtle knowing smile, unapologetically confident energy",
  sultry:
    "slow smoldering gaze, lips slightly parted, subtly arched posture, relaxed but undeniably sensual energy",
};

const LEXIVERSE_STYLE_MAP: Record<LexiverseStyle, string> = {
  off: "",
  soft: "subtle stylized Lexiverse lighting and color",
  full: "bold Lexiverse-inspired stylized lighting, slightly surreal color grading",
  promo: "dramatic Lexiverse promo art style, highly stylized lighting",
};

type HairKey = keyof typeof HAIR_MAP;
type HairStyleKey = keyof typeof HAIR_STYLE_MAP;
type SkinKey = keyof typeof SKIN_TONE_MAP;
type EyeKey = keyof typeof EYE_MAP;
type OutfitKey = keyof typeof OUTFIT_MAP;
type VibeKey = keyof typeof VIBE_MAP;

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
  hair: "keep",
  hairStyle: "keep",
  skinTone: "keep",
  eyes: "keep",
  outfit: "keep",
  vibe: "soft",
  lexiverseStyle: "soft",
  extraDetails: "",
};

function buildAvatarPrompt(state: AvatarFormState): string {
  const hairColor = HAIR_MAP[state.hair];
  const hairStyle = HAIR_STYLE_MAP[state.hairStyle];
  const hairCombined = [hairColor, hairStyle].filter(Boolean).join(" ");

  const parts = [
    BASE_AVATAR_PROMPT,
    hairCombined ? `${hairCombined} hair` : "",
    EYE_MAP[state.eyes],
    SKIN_TONE_MAP[state.skinTone],
    OUTFIT_MAP[state.outfit],
    VIBE_MAP[state.vibe],
    LEXIVERSE_STYLE_MAP[state.lexiverseStyle],
    state.extraDetails.trim(),
  ];

  return parts.filter(Boolean).join(", ");
}

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
      const prompt = buildAvatarPrompt(formState);
      const payload = {
        prompt,
        negative_prompt: DEFAULT_NEGATIVE_PROMPT,
        sd_mode: mode,
        lexiverse_style: formState.lexiverseStyle,
        // push img2img harder off the base to break repetition
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
      role="dialog"
      aria-modal="true"
    >
      <Box
        w="100%"
        maxW="1100px"
        bg={surface}
        color={textColor}
        p={{ base: 5, md: 7 }}
        borderRadius="2xl"
        boxShadow="2xl"
        position="relative"
        overflow="hidden"
      >
        <Box
          position="absolute"
          inset={0}
          bgGradient="linear(to-br, rgba(255,105,180,0.15), rgba(118,75,255,0.12))"
          pointerEvents="none"
        />
        <Box position="relative" zIndex={1}>
          <HStack justify="space-between" mb={4}>
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
              size="sm"
              variant="ghost"
              onClick={onClose}
            />
          </HStack>

          <SimpleGrid columns={{ base: 1, md: 2 }} spacing={6}>
            <VStack align="stretch" spacing={4}>
              <HStack spacing={3}>
                <FormControl>
                  <FormLabel fontSize="sm">Hair</FormLabel>
                  <Select
                    value={formState.hair}
                    onChange={(e) => updateField("hair", e.target.value as HairKey)}
                  >
                    <option value="keep">Keep as is</option>
                    <option value="brunette">Dark brunette</option>
                    <option value="blonde">Blonde</option>
                    <option value="redhead">Redhead</option>
                    <option value="black">Black hair</option>
                  </Select>
                </FormControl>
                <FormControl>
                  <FormLabel fontSize="sm">Style</FormLabel>
                  <Select
                    value={formState.hairStyle}
                    onChange={(e) => updateField("hairStyle", e.target.value as HairStyleKey)}
                  >
                    <option value="keep">Keep natural</option>
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
                    value={formState.skinTone}
                    onChange={(e) => updateField("skinTone", e.target.value as SkinKey)}
                  >
                    <option value="keep">Keep as is</option>
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
                    value={formState.eyes}
                    onChange={(e) => updateField("eyes", e.target.value as EyeKey)}
                  >
                    <option value="keep">Keep as is</option>
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
                    value={formState.outfit}
                    onChange={(e) => updateField("outfit", e.target.value as OutfitKey)}
                  >
                    <option value="keep">Keep as is</option>
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
                  value={formState.extraDetails}
                  onChange={(e) => updateField("extraDetails", e.target.value)}
                  placeholder="e.g., soft pink nails, studio lighting, a hint of freckles…"
                  minH="80px"
                  maxH="160px"
                  resize="vertical"
                />
              </FormControl>

              <FormControl>
                <FormLabel fontSize="sm">Lexiverse style intensity</FormLabel>
                <Select
                  value={formState.lexiverseStyle}
                  onChange={(e) =>
                    updateField("lexiverseStyle", e.target.value as AvatarFormState["lexiverseStyle"])
                  }
                >
                  <option value="off">Off</option>
                  <option value="soft">Soft Lexiverse</option>
                  <option value="full">Full Lexiverse</option>
                  <option value="promo">Promo mode</option>
                </Select>
              </FormControl>
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
                  <Image src={currentAvatarUrl} alt="Current avatar" w="100%" h="100%" />
                ) : (
                  <Box p={6}>
                    <Text color={helperColor} textAlign="center">
                      Your current avatar will show here.
                    </Text>
                  </Box>
                )}
              </Box>
              <Box>
                <Badge mb={2} colorScheme="pink" bg={badgeBg}>
                  Generation mode
                </Badge>
                <VStack align="stretch" spacing={3}>
                  <Button
                    colorScheme="pink"
                    onClick={() => void handleGenerate("img2img")}
                    isDisabled={isSubmitting}
                  >
                    {isSubmitting ? <Spinner size="sm" mr={2} /> : null}
                    Change my look
                  </Button>
                  <Text fontSize="xs" color={helperColor} ml={1}>
                    Lightly update my look while keeping me recognizably Lexi.
                  </Text>
                  <Button
                    variant="outline"
                    onClick={() => void handleGenerate("txt2img")}
                    isDisabled={isSubmitting}
                  >
                    {isSubmitting ? <Spinner size="sm" mr={2} /> : null}
                    Try something new
                  </Button>
                  <Text fontSize="xs" color={helperColor} ml={1}>
                    Generate a brand new Lexi from scratch.
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
  );
}

export default AvatarToolsModal;
