"""
flux prompt scaffolds used by the default avatar workflow.
"""

BASE_POS_1 = (
    "full-body portrait in the Lexiverse style of Lexi — a beautiful young woman indoors;\n"
    "camera at waist height, vertical 4:5 composition; subject centered with mild pose asymmetry;\n"
    "natural hand engagement, confident and relaxed expression; cinematic realism;\n"
    "bright natural light, balanced color tones, vivid depth, and soft highlights."
)

BASE_POS_2 = (
    "neutral studio key light + soft fill, no rimlight, no neon;\n"
    "natural skin texture, realistic specular highlights;\n"
    "filmic contrast, neutral color grade, accurate white balance;\n"
    "minimal casual outfit with neutral palette and clean textures; timeless aesthetic."
)

BASE_NEG_1 = (
    "flat lighting, harsh shadows, dim exposure, dull or washed-out tones, deformed proportions;\n"
    "low contrast, underexposed or uneven illumination;\n"
    "oversaturated orange or yellow cast, tungsten hue, artificial bronze skin, high-contrast shadows;\n"
    "visible lamps, spotlights, lighting gear, lens flare, strong backlight;\n"
    "no formal suits, business attire, dresses, jewelry, or overtly revealing outfits;\n"
    "avoid monochrome or flat tone, over-orange coloration, or yellow cast."
)

BASE_NEG_2 = (
    "flat lighting, harsh shadows, dim exposure, low contrast, washed-out tones;\n"
    "oversaturated warm hues or tungsten color cast;\n"
    "visible lighting equipment, glare, or strong backlight;\n"
    "unnatural bronze skin tone or exaggerated contrast;\n"
    "formalwear, jewelry, or explicit attire;\n"
    "monochrome or over-orange color grading;\n"
    "halo, outline, glow, rimlight, backlight, cutout, sticker, overprocessed HDR, bloom."
)

__all__ = [
    "BASE_POS_1",
    "BASE_POS_2",
    "BASE_NEG_1",
    "BASE_NEG_2",
]
