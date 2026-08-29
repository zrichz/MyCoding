#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
F2K Anime Prompt Generator
Generates flux2klein prompts for anime-style illustrations
Focus: Solo girl, risque aesthetic, anime illustration style
Date: 2026-Aug

STAGES:
  1. Core style and quality tags
  2. Character base (girl, solo)
  3. Hair and eye details
  4. Expression and pose
  5. Clothing (risque)
  6. Body descriptors
  7. Framing and composition
  8. Background and setting
  9. Lighting and atmosphere
  10. Art style modifiers
"""

import gradio as gr
import random
from datetime import datetime
import os

# WILDCARD LOADER
def load_wildcard_file(folder, filename):
    """Load and parse a wildcard file, returning list of options."""
    filepath = os.path.join("AI_PROMPTING", folder, filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by comma or newline and strip whitespace
    items = [item.strip() for item in content.replace('\n', ',').split(',') if item.strip()]
    return items

# Load wildcards
WILDCARD_CLOTHING = {
    "dress_color": load_wildcard_file("prompt_clothing_wildcards", "dress_color.txt"),
    "patterns": load_wildcard_file("prompt_clothing_wildcards", "patterns.txt"),
    "dress_material": load_wildcard_file("prompt_clothing_wildcards", "dress_material.txt"),
    "dress_type": load_wildcard_file("prompt_clothing_wildcards", "dress_type.txt"),
}

# CORE ANIME STYLE TAGS
STYLE_BASE = [
    "anime illustration",
    "anime artstyle",
    "detailed anime art",
    "high quality anime",
    "beautiful anime illustration",
    "professional anime art",
]

QUALITY_TAGS = [
    "masterpiece",
    "best quality",
    "high resolution",
    "extremely detailed",
    "beautiful detailed",
    "absurdres",
]

# CHARACTER BASE
CHARACTER_BASE = [
    "1girl",
    "solo",
    "solo girl",
    "single girl",
]

# HAIR STYLES AND COLORS
HAIR_COLORS = [
    "blonde hair",
    "silver hair",
    "white hair",
    "pink hair",
    "blue hair",
    "purple hair",
    "red hair",
    "black hair",
    "brown hair",
    "platinum blonde hair",
    "pastel pink hair",
    "pastel blue hair",
    "mint green hair",
    "lavender hair",
    "gradient hair",
    "multicolored hair",
    "two-tone hair",
]

HAIR_STYLES = [
    "long hair",
    "very long hair",
    "short hair",
    "medium hair",
    "twintails",
    "ponytail",
    "side ponytail",
    "high ponytail",
    "twin braids",
    "single braid",
    "messy hair",
    "straight hair",
    "wavy hair",
    "curly hair",
    "hair between eyes",
    "bangs",
    "side-swept bangs",
    "blunt bangs",
    "ahoge",
    "hair bun",
    "double bun",
]

HAIR_ACCESSORIES = [
    "hair ribbon",
    "hair bow",
    "hairclip",
    "hairpin",
    "hair ornament",
    "flower in hair",
    "hair scrunchie",
    "headband",
]

# EYE COLORS AND DETAILS
EYE_COLORS = [
    "blue eyes",
    "green eyes",
    "red eyes",
    "purple eyes",
    "pink eyes",
    "amber eyes",
    "golden eyes",
    "heterochromia",
    "gradient eyes",
    "bright eyes",
    "glowing eyes",
]

EYE_DETAILS = [
    "looking at viewer",
    "eye contact",
    "half-closed eyes",
    "bedroom eyes",
    "sultry gaze",
    "sidelong glance",
    "looking back",
    "eyes visible through hair",
]

# EXPRESSIONS
EXPRESSIONS = [
    "smirk",
    "slight smile",
    "seductive smile",
    "confident smile",
    "playful expression",
    "teasing expression",
    "blush",
    "light blush",
    "flushed cheeks",
    "closed mouth",
    "parted lips",
    "open mouth",
    "tongue out",
    "smug expression",
    "coy expression",
]

# POSES AND ACTIONS
POSES = [
    "standing",
    "sitting",
    "kneeling",
    "leaning forward",
    "arching back",
    "stretching",
    "arms up",
    "arms behind back",
    "arms behind head",
    "hand on hip",
    "hands on hips",
    "hand on own chest",
    "adjusting hair",
    "hand in own hair",
    "lying down",
    "lying on side",
    "lying on back",
    "from behind",
    "back view",
    "looking back at viewer",
    "over shoulder",
    "bent over",
    "squatting",
    "on all fours",
    "crossed legs",
    "legs apart",
    "one knee up",
]

# RISQUE CLOTHING OPTIONS
CLOTHING_RISQUE = [
    "bikini",
    "micro bikini",
    "string bikini",
    "lingerie",
    "lace lingerie",
    "bra and panties",
    "see-through clothing",
    "transparent clothing",
    "crop top",
    "sports bra",
    "tube top",
    "off-shoulder shirt",
    "open shirt",
    "unbuttoned shirt",
    "loose shirt",
    "oversized shirt",
    "torn clothes",
    "ripped clothes",
    "shorts",
    "short shorts",
    "hot pants",
    "mini skirt",
    "micro skirt",
    "pleated skirt",
    "thigh strap",
    "garter belt",
    "thighhighs",
    "stockings",
    "fishnets",
    "bodystocking",
    "bodysuit",
    "leotard",
    "swimsuit",
    "one-piece swimsuit",
    "sling bikini",
    "side-tie bikini",
    "latex outfit",
    "tight clothing",
    "skin-tight clothing",
    "cleavage cutout",
    "navel cutout",
    "sideboob cutout",
    "sleeveless",
    "bare shoulders",
    "bare arms",
    "midriff",
    "underboob",
    "side cut",
    "revealing outfit",
]

CLOTHING_DETAILS = [
    "lace trim",
    "frills",
    "ribbon",
    "bow",
    "choker",
    "necklace",
    "collar",
    "armband",
    "wristband",
    "gloves",
    "fingerless gloves",
    "elbow gloves",
]

# BODY DESCRIPTORS
BODY_FEATURES = [
    "slender",
    "curvy",
    "hourglass figure",
    "athletic build",
    "toned",
    "petite",
    "tall",
    "perfect proportions",
    "beautiful detailed body",
]

BODY_DETAILS = [
    "large breasts",
    "medium breasts",
    "cleavage",
    "bare shoulders",
    "bare legs",
    "thighs",
    "thick thighs",
    "toned legs",
    "navel",
    "collarbone",
    "arms",
    "midriff",
    "hips",
    "narrow waist",
]

# FRAMING AND COMPOSITION
FRAMING = [
    "full body",
    "upper body",
    "cowboy shot",
    "from below",
    "from above",
    "from side",
    "dynamic angle",
    "dutch angle",
    "centered composition",
    "off-center composition",
]

# BACKGROUNDS AND SETTINGS
BACKGROUNDS = [
    "bedroom",
    "indoors",
    "outdoors",
    "beach",
    "poolside",
    "city background",
    "night sky",
    "sunset background",
    "urban setting",
    "rooftop",
    "balcony",
    "window",
    "simple background",
    "gradient background",
    "blurred background",
    "bokeh background",
    "abstract background",
]

# LIGHTING
LIGHTING = [
    "soft lighting",
    "dramatic lighting",
    "rim lighting",
    "backlight",
    "side lighting",
    "sunset lighting",
    "golden hour lighting",
    "window light",
    "natural light",
    "studio lighting",
    "volumetric lighting",
    "glowing",
    "light particles",
    "sunbeam",
    "lens flare",
]

# ATMOSPHERE
ATMOSPHERE = [
    "warm atmosphere",
    "cool atmosphere",
    "dreamy atmosphere",
    "romantic atmosphere",
    "intimate atmosphere",
    "sensual atmosphere",
    "ethereal atmosphere",
    "vibrant colors",
    "soft colors",
    "pastel colors",
    "saturated colors",
]

# ART STYLE MODIFIERS
ART_STYLES = [
    "modern anime style",
    "cell shaded",
    "soft shading",
    "detailed shading",
    "gradient shading",
    "clean lineart",
    "detailed lineart",
    "smooth lines",
    "painterly",
    "semi-realistic anime",
    "manga style coloring",
]

COLOR_PALETTES = [
    "warm color palette",
    "cool color palette",
    "pastel palette",
    "vibrant palette",
    "muted colors",
    "rich colors",
]

def generate_prompt():
    """Generate a single anime illustration prompt."""
    parts = []
    
    # 1. Core style and quality
    parts.append(random.choice(STYLE_BASE))
    parts.append(random.choice(QUALITY_TAGS))
    
    # 2. Character base
    parts.append(random.choice(CHARACTER_BASE))
    
    # 3. Hair details
    hair_color = random.choice(HAIR_COLORS)
    hair_style = random.choice(HAIR_STYLES)
    parts.append(f"{hair_color}, {hair_style}")
    if random.random() < 0.3:  # 30% chance for hair accessory
        parts.append(random.choice(HAIR_ACCESSORIES))
    
    # 4. Eye details
    parts.append(random.choice(EYE_COLORS))
    parts.append(random.choice(EYE_DETAILS))
    
    # 5. Expression
    parts.append(random.choice(EXPRESSIONS))
    
    # 6. Pose
    parts.append(random.choice(POSES))
    
    # 7. Clothing (risque focus)
    main_clothing = random.choice(CLOTHING_RISQUE)
    parts.append(main_clothing)
    
    # Add clothing details (30% chance)
    if random.random() < 0.3:
        parts.append(random.choice(CLOTHING_DETAILS))
    
    # 8. Body descriptors
    parts.append(random.choice(BODY_FEATURES))
    # Add 1-2 body details
    num_body_details = random.randint(1, 2)
    for _ in range(num_body_details):
        parts.append(random.choice(BODY_DETAILS))
    
    # 9. Framing
    parts.append(random.choice(FRAMING))
    
    # 10. Background
    parts.append(random.choice(BACKGROUNDS))
    
    # 11. Lighting
    parts.append(random.choice(LIGHTING))
    
    # 12. Atmosphere
    if random.random() < 0.5:  # 50% chance
        parts.append(random.choice(ATMOSPHERE))
    
    # 13. Art style modifiers
    parts.append(random.choice(ART_STYLES))
    if random.random() < 0.4:  # 40% chance
        parts.append(random.choice(COLOR_PALETTES))
    
    # Join with commas
    prompt = ", ".join(parts)
    
    return prompt

def generate_batch(num_prompts, progress=gr.Progress()):
    """Generate a batch of anime prompts."""
    prompts = []
    
    for i in range(num_prompts):
        prompt = generate_prompt()
        prompts.append(f"{prompt}\n")
        
        if i % 10 == 0:
            progress((i + 1) / num_prompts, desc=f"Generating prompt {i+1}/{num_prompts}")
    
    return "".join(prompts)

def save_prompts(prompts_text):
    """Save prompts to file."""
    if not prompts_text:
        return "No prompts to save"
    
    output_dir = "AI_PROMPTING"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"flux2klein_anime_prompts_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(prompts_text)
    
    return f"Saved {filepath}"

def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(title="F2K Anime Prompt Generator") as demo:
        gr.Markdown("""
        # Flux2Klein Anime Illustration Prompt Generator
        
        Generates prompts for anime-style illustrations featuring solo girls with risque aesthetic.
        
        Focus: High-quality anime art, detailed character designs, sensual poses and outfits.
        """)
        
        with gr.Row():
            num_prompts = gr.Slider(
                minimum=1,
                maximum=500,
                value=100,
                step=1,
                label="Number of Prompts"
            )
        
        generate_btn = gr.Button("Generate Prompts", variant="primary", size="lg")
        
        output_text = gr.Textbox(
            label="Generated Prompts",
            lines=20,
            max_lines=30
        )
        
        save_btn = gr.Button("Save to File")
        save_status = gr.Textbox(label="Save Status", lines=1)
        
        # Event handlers
        generate_btn.click(
            fn=generate_batch,
            inputs=[num_prompts],
            outputs=[output_text]
        )
        
        save_btn.click(
            fn=save_prompts,
            inputs=[output_text],
            outputs=[save_status]
        )
        
        # Example section
        gr.Markdown("""
        ### Example Output
        ```
        anime illustration, masterpiece, 1girl, solo, pink hair, long hair, 
        blue eyes, looking at viewer, seductive smile, light blush, standing, 
        hand on hip, lingerie, lace trim, slender, large breasts, cleavage, 
        full body, bedroom, soft lighting, intimate atmosphere, modern anime style, 
        pastel palette
        ```
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(inbrowser=True)
