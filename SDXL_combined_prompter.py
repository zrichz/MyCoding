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

SECONDARY CATEGORIES (4):
  1. Color grading / Film style
  2. Depth of field / Bokeh
  3. Texture / Finish
  4. Mood / Subtle effects
"""

import gradio as gr
import random
from datetime import datetime

# PRIMARY STAGES (8)
PRIMARY_STAGES = {
    "Subject identity": [
        "amateur photo, boobs, stunning, full-body shot, a photo of a young, beautiful woman",
        
    ],
    "Pose and action": [
        "three-quarter turn",
        "standing",
        "seated",
        "walking toward camera",
        "sitting, legs crossed",
        "leaning",
        "brushing hair back with one hand",
        "standing with arms folded loosely",
        "sitting, elbows on knees",
        "leaning, relaxed",
        "sitting",
        "looking, thoughtful",
        "reaching, casual stance",
        "sitting cross-legged on the floor",
        "leaning over",
        
    ],
    "Framing and crop": [
        "full body",
        
    ],
    "Clothing and key props": [
        "light coral botanical print minidress, simple cut",
        "navy and white striped t-shirt with denim shorts",
        "rainbow tie-dye polo shirt and faded blue jeans",
        "light purple geometric pattern yoga top with dark grey leggings",
        "leopard print bikini in tan and dark brown with coral sarong",
        "green and dark olive camouflage rave outfit with neon glowsticks",
        "turquoise ikat pattern gossamer blouse with light beige mini skirt",
        "bright yellow logo print sweatshirt with grey pleated skirt",
        "Soul Calibur cosplay outfit in light blue and silver",
        "pastel pink abstract art print tee with light wash denim shorts",
        "dark purple monogram print evening dress",
        "sage green botanical print summer dress with floral pattern",
        "bright orange puffer jacket with dark indigo jeans",
        "light blue and cream tartan casual shirt",
        "neon rainbow geometric pattern athleisure wear",
        "dark green plaid flannel tee with distressed denim hotpants",
        "hot pink zebra print clubbing outfit",
        "baby pink silk babydoll top with fuzzy white slippers",
        "cheetah print tank top in golden brown with dark navy skirt",
        "sunset tie-dye t-shirt (orange, pink, yellow) with khaki shorts",
        "classic red tartan skirt",
        "sunny yellow tank top with tropical leaf print shorts",
        "light mint tennis skirt with matching crop top",
        "bright fuchsia mesh athletic top with geometric print bike shorts",
        
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
        "strong posture",
        "relaxed shoulders",
        "toned calves",
        "natural posture, relaxed",
        "asymmetric stance",
    ],
    "Context or location": [
        "window-lit interior",
        "bathroom mirror selfie",
        "kitchen counter in morning light",
        "park bench near river",
        "coastal, sunny",
        "living room",
        "garden patio with potted plants",
        "early morning",
        "tropical beach",
        "garden",
        "country lane with hedgerows",
        "canal towpath, morning mist",
        "railway bridge, industrial backdrop",
        "seaside pier, muted light",
        "city rooftop with distant skyline",
        "holiday cottage kitchen",
        "home office with desk lamp",
        "hotel lobby with soft lighting",
        
    ],
    "Semantic technical anchors": [
        "mirror selfie",
        "candid handheld shot",
        "selfie",
        "natural portrait",
        "overhead light",
        "golden hour shot",
        "soft overcast daylight",
        "phone selfie with arm extended",
        "shot from above eye level",
        "shot from below eye level",
        "natural light from left",
        "natural light from right",
        "softbox-like diffused light",
        "subject half-lit",
        "natural shade",
        "ambient lighting",
        "overcast diffuse, even light",
        "warm morning light",
        "cool evening light",
        "natural light with subtle shadow",
    ],
}

# SECONDARY CATEGORIES (4)
SECONDARY_CATEGORIES = {
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
        "subtle film grain",
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
    output_lines = [f'--prompt "{prompt}" --negative_prompt "asian, makeup"' for prompt in last_8]
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
    gr.Markdown("Enable/disable stages to customize output.")
    
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
            secondary_defaults = [True, False, True, True]  # 1,3,4 enabled
            for i, cat_name in enumerate(SECONDARY_CATEGORIES.keys()):
                secondary_checks.append(gr.Checkbox(label=cat_name, value=secondary_defaults[i]))
    
    generate_btn = gr.Button("Generate 400 Prompts", variant="primary", size="lg")
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    output_text = gr.Textbox(
        label="Generated 400 Prompts",
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
