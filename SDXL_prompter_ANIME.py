"""
SDXL Prompt Generator with Danbooru Tags
generates 400no. SDXL prompts, with 'Primary' and 'Secondary' stages.
(although auto1111 doesn't split them like comfyui)
Date: 2026-April

PRIMARY STAGES (8):
  1. Subject identity
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Semantic technical anchors

DANBOORU WILDCARDS (4):
  1. Artist tags (top 200)
  2. Character tags (top 200)
  3. Copyright tags (top 200)
  4. General tags (top 200)

SECONDARY CATEGORY (1):
  1. Secondary effects (color/depth/texture/mood)
"""

import gradio as gr
import random
import json
from datetime import datetime
from pathlib import Path


# Load Danbooru tags from JSON files (top 200 each)
def load_danbooru_tags(filename, limit=200):
    """Load danbooru tags from JSON file, limited to top N items."""
    try:
        filepath = Path(__file__).parent / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            tags = json.load(f)
        return tags[:limit]
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return []


def load_wildcard_txt(filename):
    """Load wildcard items from .txt file in wildcards directory."""
    try:
        filepath = Path(__file__).parent / "wildcards" / filename
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        # Split by comma and newline, strip whitespace, filter empty
        items = [item.strip() for item in content.replace('\n', ',').split(',') if item.strip()]
        return items
    except Exception as e:
        print(f"Warning: Could not load {filename}: {e}")
        return []


# Load all danbooru tag categories
DANBOORU_ARTISTS = load_danbooru_tags("danbooru_tags_artist.json", 200)
DANBOORU_CHARACTERS = load_danbooru_tags("danbooru_tags_character.json", 200)
DANBOORU_COPYRIGHT = load_danbooru_tags("danbooru_tags_copyright.json", 200)
DANBOORU_GENERAL = load_danbooru_tags("danbooru_tags_general.json", 200)

# Load clothing wildcards from .txt files
DRESS_TYPES = load_wildcard_txt("dress_type.txt")
DRESS_COLORS = load_wildcard_txt("dress_color.txt")
DRESS_MATERIALS = load_wildcard_txt("dress_material.txt")
PATTERNS = load_wildcard_txt("patterns.txt")
FOOTWEAR_TYPES = load_wildcard_txt("footwear_type.txt")
FOOTWEAR_COLORS = load_wildcard_txt("footwear_color.txt")
FOOTWEAR_MATERIALS = load_wildcard_txt("footwear_material.txt")


def generate_clothing_combo():
    """Generate random clothing combination from wildcards."""
    dress_type = random.choice(DRESS_TYPES) if DRESS_TYPES else "dress"
    dress_color = random.choice(DRESS_COLORS) if DRESS_COLORS else "colored"
    dress_material = random.choice(DRESS_MATERIALS) if DRESS_MATERIALS else "fabric"
    pattern = random.choice(PATTERNS) if PATTERNS else "plain"
    footwear_type = random.choice(FOOTWEAR_TYPES) if FOOTWEAR_TYPES else "shoes"
    footwear_color = random.choice(FOOTWEAR_COLORS) if FOOTWEAR_COLORS else "colored"
    footwear_material = random.choice(FOOTWEAR_MATERIALS) if FOOTWEAR_MATERIALS else "material"
    
    return f"{dress_type}, {dress_color}, {dress_material}, {pattern}, {footwear_type}, {footwear_color}, {footwear_material}"

# PRIMARY STAGES (8)
PRIMARY_STAGES = {
    "Subject identity": [
        "an award-winning realistic illustration of a beautiful woman, boobs focus, blonde, hair up,smile",
        
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
        "WILDCARD_CLOTHING",  # Special marker for dynamic wildcard generation
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
        "school rooftop at sunset",
        "cherry blossom park in spring",
        "cozy café interior with large windows",
        "traditional Japanese shrine courtyard",
        "modern city street at dusk",
        "classroom with afternoon sunlight streaming in",
        "beach at golden hour",
        "moonlit garden with flowers",
        "train station platform, evening",
        "bedroom with soft window light",
        "shopping district, vibrant atmosphere",
        "library interior with book shelves",
        "park bench under autumn trees",
        "riverside with bridge background",
        "festival grounds with lanterns",
        "music room with instruments",
        "starry night sky backdrop",
        "urban rooftop with city lights",
        
    ],
    "Semantic technical anchors": [
        "key visual composition",
        "dynamic three-quarter view",
        "dutch angle, dramatic tilt",
        "low angle perspective, heroic",
        "high angle view, intimate",
        "centered portrait composition",
        "rule of thirds placement",
        "diagonal composition, energetic",
        "symmetrical framing, balanced",
        "depth layering, foreground blur",
        "anime screenshot aesthetic",
        "light novel cover style",
        "visual novel CG composition",
        "manga panel layout influence",
        "character focus, detailed rendering",
        "wide-angle environment emphasis",
        "close-up character study",
        "cinematic widescreen framing",
        "promotional art composition",
        "official art style reference",
    ],
}

# DANBOORU WILDCARDS (4) - Dynamic categories loaded from JSON files
DANBOORU_WILDCARDS = {
    "Artist style": DANBOORU_ARTISTS,
    "Character": DANBOORU_CHARACTERS,
    "Copyright/Franchise": DANBOORU_COPYRIGHT,
    "General tags": DANBOORU_GENERAL,
}

# SECONDARY CATEGORY (1)
SECONDARY_CATEGORIES = {
    "Secondary effects": [
        # Rendering style
        "clean cel shading, sharp edges",
        "soft watercolor style, gentle blending",
        "vibrant digital painting, bold colors",
        "pastel color palette, dreamy atmosphere",
        # Line art style
        "crisp line art, defined contours",
        "thick outline style, manga aesthetic",
        "delicate linework, fine details",
        # Lighting effects
        "dramatic rim lighting, character emphasis",
        "soft diffuse glow, ethereal mood",
        "high contrast lighting, bold shadows",
        "gentle backlighting, atmospheric depth",
        # Artistic effects
        "subtle bloom effect, soft highlights",
        "sparkle accents, magical atmosphere",
        "dynamic motion blur, energetic feel",
        # Atmosphere and mood
        "vibrant saturated colors, cheerful tone",
        "muted tones, melancholic atmosphere",
        "high energy composition, dynamic angles",
        "serene calm mood, balanced composition",
    ],
}

def generate_prompts(primary_enabled, danbooru_enabled, secondary_enabled):
    """Generate 400 combined prompts based on enabled stages."""
    prompts = []
    
    for _ in range(400):
        # Generate primary prompt
        primary_parts = []
        for stage_name, options in PRIMARY_STAGES.items():
            if primary_enabled.get(stage_name, True):
                selected = random.choice(options)
                # Handle dynamic wildcard generation for clothing
                if selected == "WILDCARD_CLOTHING":
                    selected = generate_clothing_combo()
                primary_parts.append(selected)
        
        # Add danbooru wildcards if enabled
        danbooru_parts = []
        for wildcard_name, options in DANBOORU_WILDCARDS.items():
            if danbooru_enabled.get(wildcard_name, False) and options:
                # Select 2-3 random tags from this category
                num_tags = random.randint(2, 3)
                selected_tags = random.sample(options, min(num_tags, len(options)))
                danbooru_parts.extend(selected_tags)
        
        # Combine primary and danbooru
        if danbooru_parts:
            random.shuffle(danbooru_parts)  # Randomize tag order
            primary_parts.append(", ".join(danbooru_parts))
        
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
    # Parse checkboxes (8 primary + 4 danbooru + 1 secondary = 13 total)
    primary_enabled = {}
    danbooru_enabled = {}
    secondary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    danbooru_names = list(DANBOORU_WILDCARDS.keys())
    secondary_names = list(SECONDARY_CATEGORIES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    for i, name in enumerate(danbooru_names):
        danbooru_enabled[name] = checkboxes[len(primary_names) + i]
    
    for i, name in enumerate(secondary_names):
        secondary_enabled[name] = checkboxes[len(primary_names) + len(danbooru_names) + i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, danbooru_enabled, secondary_enabled)
    
    # Format output - show only last 8 prompts with prefix and suffix
    last_8 = prompts[-8:]
    output_lines = [f'--prompt "{prompt}" --negative_prompt "photo, asian, makeup"' for prompt in last_8]
    output = "\n\n".join(output_lines)

    return output, prompts


def save_prompts(prompts_data):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SDXL_ANIME_prompts_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "asian, makeup, (tanned:0.15)"\n')
    
    return f"Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SDXL Combined Prompt Generator with Danbooru Tags")
    gr.Markdown("Generate 400 randomized prompts in format: `<primary> | <secondary>`")
    gr.Markdown("Enable/disable stages to customize output. Danbooru tags add 2-3 random items per enabled category.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            primary_checks = []
            primary_defaults = [True, True, False, False, True, False, False, False]  # 1,2,5 enabled
            for i, stage_name in enumerate(PRIMARY_STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
        
        with gr.Column(scale=1):
            gr.Markdown("### DANBOORU WILDCARDS")
            danbooru_checks = []
            danbooru_defaults = [False, False, False, True]  # Only general tags enabled by default
            for i, wildcard_name in enumerate(DANBOORU_WILDCARDS.keys()):
                count = len(DANBOORU_WILDCARDS[list(DANBOORU_WILDCARDS.keys())[i]])
                label = f"{wildcard_name} ({count} tags)"
                danbooru_checks.append(gr.Checkbox(label=label, value=danbooru_defaults[i]))
            
            gr.Markdown("### SECONDARY CATEGORY")
            secondary_checks = []
            secondary_defaults = [True]  # enabled by default
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
    all_checkboxes = primary_checks + danbooru_checks + secondary_checks
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
