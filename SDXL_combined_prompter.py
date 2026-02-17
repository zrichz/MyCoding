"""
SDXL Prompt Generator
generates 400no. SDXL prompts, with 'Primary' and 'Secondary' stages.
(although auto1111 doesn't split them like comfyui)
Date: 2026-Feb

PRIMARY STAGES (8):
  1. Subject identity
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Semantic technical anchors

SECONDARY CATEGORIES (6):
  1. Camera / Perspective
  2. Lens / Focal length
  3. Color grading / Film style
  4. Depth of field / Bokeh
  5. Texture / Finish
  6. Mood / Subtle effects
"""

import gradio as gr
import random
from datetime import datetime

# PRIMARY STAGES (8)
PRIMARY_STAGES = {
    "Subject identity": [
        "full-body shot, legs, (nude-colored platform heels), a photo of a woman,hair up,30 years old,(smiling:0.5)",
        
    ],
    "Pose and action": [
        "three-quarter turn",
        "standing, hands in pockets",
        "seated, one knee up",
        "walking toward camera",
        "sitting on a low wall, legs crossed",
        "leaning on a railing, looking down",
        "holding a mug with both hands",
        "adjusting jacket collar casually",
        "reading a paperback, head tilted",
        "tying shoelace, looking away",
        "brushing hair back with one hand",
        "standing with arms folded loosely",
        "sitting on stairs, elbows on knees",
        "holding a takeaway coffee, mid-sip",
        "checking phone while standing",
        "leaning against doorframe, relaxed",
        "sitting at table, hands clasped",
        "looking out of a window, thoughtful",
        "standing, slight smile",
        "carrying a backpack, mid-step",
        "sitting on a bench, one arm draped",
        "walking up a short flight of steps",
        "adjusting glasses, slight tilt of head",
        "reaching for a shelf, casual stance",
        "sitting cross-legged on the floor",
        "leaning over a kitchen counter",
        "standing with one foot on a low step",
        "holding a camera at chest height",
        "sitting in a cafe booth, relaxed",
        "standing by a window, hands in pockets",
        "walking past a shopfront, glancing",
        "sitting on a windowsill, knees drawn up",
        "holding a book, reading",
        "standing with coat draped over shoulder",
        "sitting on a low wall, feet dangling",
        "mid-laugh, head thrown back slightly",
    ],
    "Framing and crop": [
        "waist-up",
        "head and shoulders",
        "full body",
        "three-quarter body",
        "close-up face",
        "knee-up",
        "environmental portrait, subject small in frame",
        "tight portrait, eyes centered",
        "mid-shot with foreground blur",
        "portrait orientation, headroom",
    ],
    "Clothing and key props": [
        "light coral botanical print minidress, simple cut",
        "dark teal plaid casual hoodie with light grey trainers",
        "navy and white striped t-shirt with denim shorts",
        "bright red gingham checked shirt and light blue denim",
        "cream polka dot knit sweater with burgundy skirt",
        "rainbow tie-dye polo shirt and faded blue jeans",
        "light purple geometric pattern yoga top with dark grey leggings",
        "leopard print bikini in tan and dark brown with coral sarong",
        "green and dark olive camouflage rave outfit with neon glowsticks",
        "dark red tartan oversized knit with black leggings",
        "turquoise ikat pattern gossamer blouse with light beige midi skirt",
        "bright yellow logo print sweatshirt with grey pleated skirt",
        "Soul Calibur cosplay outfit in electric blue and silver",
        "pastel pink abstract art print tee with light wash denim shorts",
        "dark purple monogram print evening dress",
        "sage green botanical print summer dress with floral pattern",
        "bright orange puffer jacket with dark indigo jeans",
        "light blue and cream tartan casual shirt",
        "neon rainbow geometric pattern athleisure wear",
        "dark green plaid flannel tee with distressed denim hotpants",
        "hot pink zebra print clubbing outfit",
        "light lavender polka dot silk shirt with cream skirt",
        "tan and forest green camouflage gilet with black jeans",
        "mint green and white horizontal striped casual shirt with smart watch",
        "baby pink gingham pyjama top with fuzzy white slippers",
        "deep emerald botanical print blouse with black jeans",
        "coral and white chevron pattern dress",
        "cheetah print tank top in golden brown with dark navy skirt",
        "sunset tie-dye t-shirt (orange, pink, yellow) with khaki shorts",
        "burgundy and black buffalo plaid shirt with tan chinos",
        "classic red tartan skirt with charcoal grey cardigan",
        "light grey and navy stripe sweater with dark olive skirt",
        "sky blue gingham sundress with ruffled hem",
        "lime green abstract art print tunic",
        "urban grey and black digital camouflage jacket with ripped jeans",
        "magenta logo print oversized hoodie with patterned leggings",
        "vintage cream and brown polka dot blouse with rust midi skirt",
        "vibrant multi-color ikat pattern kaftan",
        "light peach crop top with high-waisted floral print pants",
        "dark cherry red bomber jacket with acid wash jeans",
        "sunny yellow tank top with tropical leaf print shorts",
        "bright cobalt blue oversized sweater with light pink jogging shorts",
        "rainbow stripe crop hoodie with white denim skirt",
        "dark forest green utility vest with paisley print tee underneath",
        "light turquoise tie-dye sweatshirt with galaxy print leggings",
        "hot coral bandeau top with light denim overall shorts",
        "dark violet graphic tee with neon splatter print and black skinny jeans",
        "light mint polka dot tennis skirt with matching crop top",
        "bright fuchsia mesh athletic top with geometric print bike shorts",
        "dark teal color-block windbreaker with orange accents",
        "light peach fleece hoodie with cartoon character print",
        "bright lime ribbed crop tank with camo print cargo pants",
        "soft lavender swirl tie-dye romper",
        "dark maroon velour tracksuit with white racing stripes",
        "light salmon jersey dress with abstract brush stroke print",
        "electric yellow puffer vest over dark grey thermal henley",
        "light rose quilted jacket with floral embroidery and white hotpants",
        "dark navy denim jacket with enamel pin collection and rainbow stripe tee",
    ],
    "Expression and gaze": [
        "neutral expression, direct gaze to viewer",
        "candid smile, eyes to camera",
        "gentle laugh, eyes to camera",
        "contemplative, looking at viewer",
        "warm smile, eyes crinkled at camera",
        "softly serious, direct gaze at viewer",
        "subtle smile, direct eye contact",
        "calm, steady gaze at viewer",
    ],
    "Body descriptors": [
        "visible freckles on arms",
        "light sun tan",
        "pale complexion",
        "strong posture",
        "relaxed shoulders",
        "toned calves",
        "natural posture, relaxed",
        "slight asymmetry in stance",
    ],
    "Context or location": [
        "studio portrait",
        "outdoor urban alley",
        "window-lit interior",
        "bathroom mirror selfie",
        "kitchen counter in morning light",
        "local high street, daytime",
        "park bench near river",
        "train station platform",
        "coastal promenade, sunny",
        "cafe table by the window",
        "living room with bookshelf",
        "garden patio with potted plants",
        "street, early morning",
        "market stall",
        "bookshop aisle, warm light",
        "suburban front garden",
        "country lane with hedgerows",
        "ferry terminal, coastal travel",
        "small-town high street, late afternoon",
        "university quad, autumn leaves",
        "local pub beer garden",
        "train carriage window seat",
        "city square",
        "farmers market",
        "canal towpath, morning mist",
        "railway bridge, industrial backdrop",
        "seaside pier, muted light",
        "village green with benches",
        "cozy bookshop corner",
        "weekday office kitchen",
        "local bakery storefront",
        "bus interior, natural light",
        "small coastal town street",
        "city rooftop with distant skyline",
        "suburban high street cafe",
        "country pub interior",
        "holiday cottage kitchen",
        "train station concourse, travel vibe",
        "home office with desk lamp",
        "hotel lobby with soft lighting",
        "library reading room, afternoon",
        "hair salon styling station",
        "conference room with glass walls",
        "warehouse break room",
    ],
    "Semantic technical anchors": [
        "mirror selfie",
        "candid handheld shot",
        "phone camera at chest height",
        "window-lit natural portrait",
        "overhead light",
        "golden hour shot",
        "soft overcast daylight",
        "phone selfie with arm extended",
        "handheld at waist level",
        "shot from slightly above eye level",
        "shot from slightly below eye level",
        "natural window light from left",
        "natural window light from right",
        "softbox-like diffused light",
        "doorway light, subject half-lit",
        "natural shade",
        "ambient lighting",
        "overcast diffuse sky, even light",
        "warm morning light",
        "cool evening window light",
        "natural light with subtle shadow",
    ],
}

# SECONDARY CATEGORIES (6)
SECONDARY_CATEGORIES = {
    "Camera / Perspective": [
        "eye-level perspective",
        "slightly above eye-level",
        "slightly below eye-level",
        "three-quarter angle, natural",
        "environmental portrait perspective",
        "low-angle, modest dominance",
        "high-angle, modest vulnerability",
    ],
    "Lens / Focal length": [
        "50mm standard lens look",
        "35mm environmental portrait",
        "24mm slight wide environmental",
        "70mm short telephoto feel",
        "28mm modest wide angle",
        "40mm natural field of view",
        "60mm gentle compression",
        "50mm with slight bokeh",
        "24-70mm versatile zoom feel",
        "28mm for modest environmental hint",
        "35mm for casual feel",
    ],
    "Color grading / Film style": [
        "clean digital look, minimal processing",
        "slightly warm, low saturation",
        "cool tones, natural look",
        "faded film look, low contrast",
        "cinematic teal-orange, very subtle",
    ],
    "Depth of field / Bokeh": [
        "shallow depth of field, soft bokeh",
        "moderate depth, background readable",
        "deep focus, environmental detail",
        "soft background blur, natural",
        "slight background separation",
        "soft foreground blur, subject sharp",
   ],
    "Texture / Finish": [
        "clean digital finish",
        "subtle film grain",
        "matte finish, low contrast",
        "natural skin texture preserved",
        "soft clarity, minimal sharpening",
    ],
    "Mood / Subtle effects": [
        "natural, documentary tone",
        "quiet, candid mood",
        "subtle warmth, homely",
        "cheerful, candid",
        "subtle story-telling, natural",
    ],
}


def generate_prompts(primary_enabled, secondary_enabled):
    """Generate 400 combined prompts based on enabled stages."""
    prompts = []
    
    for _ in range(400):
        # Generate primary prompt
        primary_parts = []
        for stage_name, options in PRIMARY_STAGES.items():
            if primary_enabled.get(stage_name, True):
                primary_parts.append(random.choice(options))
        primary = "; ".join(primary_parts)
        
        # Generate secondary prompt
        secondary_parts = []
        for cat_name, options in SECONDARY_CATEGORIES.items():
            if secondary_enabled.get(cat_name, True):
                secondary_parts.append(random.choice(options))
        secondary = ", ".join(secondary_parts)
        
        # Combine in format: <primary> | <secondary>
        combined = f"{primary} | {secondary}"
        prompts.append(combined)
    
    return prompts


def generate_and_display(*checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (8 primary + 7 secondary = 15 total)
    primary_enabled = {}
    secondary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    secondary_names = list(SECONDARY_CATEGORIES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    for i, name in enumerate(secondary_names):
        secondary_enabled[name] = checkboxes[len(primary_names) + i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, secondary_enabled)
    
    # Format output - show only last 8 prompts with prefix and suffix
    last_8 = prompts[-8:]
    output_lines = [f'--prompt "{prompt}" --negative_prompt "asian, makeup, (tanned:0.15)"' for prompt in last_8]
    output = "\n\n".join(output_lines)

    return output, prompts


def save_prompts(prompts_data):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sdxl_combined_prompts_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "asian, makeup, (tanned:0.15)"\n')
    
    return f"✓ Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SDXL Combined Prompt Generator")
    gr.Markdown("Generate 400 randomized prompts in format: `<primary> | <secondary>`")
    gr.Markdown("Enable/disable stages to customize output. All stages enabled by default.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            primary_checks = []
            primary_defaults = [True, True, False, False, True, False, False, False]  # 1,2,5 enabled
            for i, stage_name in enumerate(PRIMARY_STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
        
        with gr.Column(scale=1):
            gr.Markdown("### SECONDARY CATEGORIES")
            secondary_checks = []
            secondary_defaults = [False, False, True, False, True, True]  # 3,5,6 enabled
            for i, cat_name in enumerate(SECONDARY_CATEGORIES.keys()):
                secondary_checks.append(gr.Checkbox(label=cat_name, value=secondary_defaults[i]))
    
    generate_btn = gr.Button("Generate 400 Prompts", variant="primary", size="lg")
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    output_text = gr.Textbox(
        label="Generated Prompts (400 total)",
        lines=20,
        max_lines=30,
        interactive=False
    )
    
    # Hidden state to store prompts for saving
    prompts_state = gr.State([])
    
    # Wire up interactions
    all_checkboxes = primary_checks + secondary_checks
    generate_btn.click(
        fn=generate_and_display,
        inputs=all_checkboxes,
        outputs=[output_text, prompts_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
