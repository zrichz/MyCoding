"""
SDXL Prompt Generator - Photo Mode
generates 400no. SDXL prompts, with 'Primary' and 'Secondary' stages.
(although auto1111 doesn't split them like comfyui)
Date: 2026-Feb, Updated: 2026-Jun

PRIMARY STAGES (7 + Subject Identity):
  1. Subject identity (always included)
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Shot and light variations
  9. Film grade
  10. Flava (stylistic atmosphere with adjustable emphasis weight)
"""

import gradio as gr
import random
from datetime import datetime
import os

# WILDCARD CLOTHING LOADER
def load_wildcard_file(filename):
    """Load and parse a wildcard file, returning list of options."""
    filepath = os.path.join("AI_PROMPTING", "prompt_clothing_wildcards", filename)
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by comma and strip whitespace
    items = [item.strip() for item in content.split(',') if item.strip()]
    return items

# Load wildcard clothing options
WILDCARD_CLOTHING = {
    "dress_color": load_wildcard_file("dress_color.txt"),
    "patterns": load_wildcard_file("patterns.txt"),
    "dress_material": load_wildcard_file("dress_material.txt"),
    "dress_type": load_wildcard_file("dress_type.txt"),
    "footwear_color": load_wildcard_file("footwear_color.txt"),
    "footwear_material": load_wildcard_file("footwear_material.txt"),
    "footwear_type": load_wildcard_file("footwear_type.txt")
}

def generate_wildcard_clothing():
    """Generate a clothing description from wildcard files."""
    # For dress: choose between color or pattern
    use_pattern = random.choice([True, False])
    if use_pattern and WILDCARD_CLOTHING["patterns"]:
        dress_color_or_pattern = random.choice(WILDCARD_CLOTHING["patterns"])
    else:
        dress_color_or_pattern = random.choice(WILDCARD_CLOTHING["dress_color"]) if WILDCARD_CLOTHING["dress_color"] else ""
    
    dress_material = random.choice(WILDCARD_CLOTHING["dress_material"]) if WILDCARD_CLOTHING["dress_material"] else ""
    dress_type = random.choice(WILDCARD_CLOTHING["dress_type"]) if WILDCARD_CLOTHING["dress_type"] else ""
    
    # For footwear
    footwear_color = random.choice(WILDCARD_CLOTHING["footwear_color"]) if WILDCARD_CLOTHING["footwear_color"] else ""
    footwear_material = random.choice(WILDCARD_CLOTHING["footwear_material"]) if WILDCARD_CLOTHING["footwear_material"] else ""
    footwear_type = random.choice(WILDCARD_CLOTHING["footwear_type"]) if WILDCARD_CLOTHING["footwear_type"] else ""
    
    # Construct clothing prompt
    dress_parts = [p for p in [dress_color_or_pattern, dress_material, dress_type] if p]
    footwear_parts = [p for p in [footwear_color, footwear_material, footwear_type] if p]
    
    clothing_items = []
    if dress_parts:
        clothing_items.append(" ".join(dress_parts))
    if footwear_parts:
        clothing_items.append(" ".join(footwear_parts))
    
    return ", ".join(clothing_items) if clothing_items else "casual outfit"

# PRIMARY STAGES (8)
# Subject identity
SUBJECT_IDENTITY = "amateur photo, full-body shot, boobs, a photo of a blonde woman, average build, hair up, (faint smile:0.2), (teeth:0.4)"

PRIMARY_STAGES = {
    "Pose and action": [
        "three-quarter turn","standing","lying on back","walking towards viewer","sitting, legs crossed",
        "stretching","posing","leaning, relaxed","on all fours","looking","reaching, casual stance","leaning over",
        
    ],
    "Framing and crop": [
        "full body",
        
    ],
    "Clothing and key props": [
        "black lace bralette with matching high-cut panties",
        "white satin babydoll with sheer overlay",
        "red lace teddy with cutout details",
        "pink silk chemise with thin straps",
        "black string bikini with minimal coverage",
        "white open blouse with cheeky bottoms",
        "ripped denim micro shorts with matching bandeau cotton top",
        "cotton crop top with matching hot pants",
        "iridescent silver micro dress",
        "cream patterned lace bodysuit",
        "white micro bodycon dress",
        "white lingerie",
        "lace-trimmed micro dress",
        "pink latex catsuit unzipped to navel",
        "choker with matching thong bodysuit",
        "white wet t-shirt over skimpy bikini bottoms",
        "bikini with side ties",
        "barely-there sling bikini in shimmering fabric",
        "open-front cardigan over lace bralette and panties",
        "shredded t-shirt exposing sides with denim cut-offs",
        
    ],
    "Expression and gaze": [
        "neutral expression, direct gaze to viewer", "candid, eyes to viewer", "direct eye contact",
        
    ],
    "Body descriptors": [
        "visible (freckles:0.5) on arms", "light sun tan", "toned calves", "natural posture, relaxed",
        
    ],
    "location": [ "interior", "selfie", "garden", "bedroom", "shower","exterior", ],
}

# Shot and light variations
SHOT_LIGHT = [
"intimate",
"casual",
"golden hour lighting mood",
"soft diffused lighting",
"dynamic perspective",
"shot from above",
"shot from below",
"dramatic half-lighting",
"soft diffused window light",
"hard direct sunlight",
"golden hour backlighting",
"overcast natural light",
"studio softbox lighting",
"studio beauty dish lighting",
"studio clamshell lighting",
"Rembrandt lighting",
"split lighting",
"loop lighting",
"broad lighting",
"short lighting",
"neon ambient lighting",
"practical tungsten lighting",
"fluorescent ambient lighting",
"mixed‑temperature lighting",
"cinematic rim lighting",
"cinematic top‑down lighting",
"moody low‑key lighting",
"bright high‑key lighting",
"bounce‑flash photography",
"off‑camera flash photography",
"ring‑light portrait lighting",
]

# FILM GRADE OPTIONS
FILM_GRADE = [
    "Kodak Portra 160 look",
    "Kodak Portra 400 look",
    "Kodak Portra 800 look",
    "Kodak Gold 200 look",
    "Kodak Ektar 100 look",
    "Fuji Pro 400H look",
    "Fuji Superia 400 look",
    "Ilford HP5 black-and-white look",
    "Ilford Delta 3200 black-and-white look",
    "cinematic teal-and-orange grade",
    "cinematic neutral-grade",
    "cinematic desaturated grade",
    "warm editorial colour grade",
    "cool fashion colour grade",
    "natural colour-accurate grade",
    "high-contrast monochrome",
    "soft low-contrast monochrome"
]

# FLAVA OPTIONS
FLAVA = [
    "arctic",
    "tropical",
    "monsoon",
    "desert",
    "nocturnal",
    "urban",
    "suburban",
    "industrial",
    "futuristic",
    "retro",
    "vintage",
    "neon",
    "infrared",
    "thermal",
    "surreal",
    "glacial",
    "volcanic",
    "coastal",
    "rain-soaked",
    "fogbound",
    "windblown",
    "stormlit",
    "moonlit",
    "sun-drenched",
    "overcast",
    "misty",
    "dusty",
    "gritty",
    "opulent",
    "minimalist",
    "baroque",
    "aristocratic",
    "bohemian",
    "arctic-blue",
    "tundra",
    "equatorial",
    "high-altitude",
    "underlit",
    "overexposed",
    "grain-heavy",
    "cinematic",
    "documentary",
    "editorial",
    "fashion-forward",
    "hyperreal",
    "monochrome",
    "chromatic",
    "saturated",
    "desaturated",
    "bleached",
    "sepia",
    "analog",
    "filmic",
    "glamour",
    "raw",
    "moody",
    "ethereal",
    "harsh",
    "ambient",
    "backlit",
    "rimlit",
    "sunset-grade",
    "twilight",
    "nebulous",
    "cosmic",
    "Martian",
    "lunar",
    "polar",
    "tropical-rainforest",
    "mosaic",
    "geometric",
    "architectural",
    "botanical",
    "oceanic",
    "arid",
    "lush",
    "windswept",
    "smoky",
    "holographic",
    "chromatic-aberration",
    "bokeh-rich",
    "macro-styled",
    "telephoto-styled"
]

def generate_prompts(primary_enabled, shot_light_enabled, film_grade_enabled, flava_enabled, flava_weight, use_wildcard_clothing=False):
    """Generate 400 combined prompts based on enabled stages."""
    prompts = []
    
    for _ in range(400):
        # Generate primary prompt
        primary_parts = []
        
        primary_parts.append(SUBJECT_IDENTITY)  # Always include subject identity
        
        for stage_name, options in PRIMARY_STAGES.items():
            if primary_enabled.get(stage_name, True):
                # Special handling for clothing stage
                if stage_name == "Clothing and key props" and use_wildcard_clothing:
                    primary_parts.append(generate_wildcard_clothing())
                else:
                    primary_parts.append(random.choice(options))
        
        # Add shot and light if enabled
        if shot_light_enabled:
            primary_parts.append(random.choice(SHOT_LIGHT))
        
        # Add film grade if enabled
        if film_grade_enabled:
            primary_parts.append(random.choice(FILM_GRADE))
        
        # Add flava if enabled
        if flava_enabled:
            keyword = random.choice(FLAVA)
            primary_parts.append(f"({keyword}:{flava_weight})")
        
        combined = "; ".join(primary_parts)
        prompts.append(combined)
    
    return prompts


def generate_and_display(shot_light_check, film_grade_check, flava_check, flava_weight, wildcard_clothing_check, *checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (7 primary stages)
    primary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, shot_light_check, film_grade_check, flava_check, flava_weight, wildcard_clothing_check)
    
    # Negative prompt
    negative_prompt = "asian, makeup"
    
    # Format output - show only last 8 prompts with prefix and suffix
    last_8 = prompts[-8:]
    output_lines = [f'--prompt "{prompt}" --negative_prompt "{negative_prompt}"' for prompt in last_8]
    output = "\n\n".join(output_lines)

    return output, prompts, negative_prompt


def save_prompts(prompts_data, negative_prompt_used):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    # Use the negative prompt that was generated during prompt creation
    negative_prompt = negative_prompt_used
    
    timestamp = datetime.now().strftime("%b%d_%H%M")
    filename = f"AI_PROMPTING/400_SDXLprompts_photo_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "{negative_prompt}"\n')
    
    return f"Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SDXL Photo Prompt Generator")
    gr.Markdown("Generate 400 randomized photo prompts with customizable primary stages")
    gr.Markdown("Enable or disable stages to customize output.")
    
    # Wildcard Clothing checkbox
    wildcard_clothing_check = gr.Checkbox(
        label="Use Wildcard Clothing (Section 4)",
        value=False,
        info="Generate clothing from wildcard files instead of built-in list"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            gr.Markdown("*Subject identity is automatically included*")
            primary_checks = []
            # Updated defaults for 7 stages (removed Subject identity)
            primary_defaults = [True, False, False, True, False, False]  # pose, framing, clothing, expression, body, location
            for i, stage_name in enumerate(PRIMARY_STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
            
            # Add shot and light as separate checkbox
            shot_light_check = gr.Checkbox(
                label="Shot and light variations",
                value=False,
                info="Lighting and framing options"
            )
            
            # Add film grade as separate checkbox
            film_grade_check = gr.Checkbox(
                label="Film grade",
                value=False,
                info="Film stock and colour grading options"
            )
            
            # Add flava as separate checkbox with weight slider
            flava_check = gr.Checkbox(
                label="Flava",
                value=False,
                info="Stylistic atmosphere keywords with adjustable emphasis"
            )
            
            flava_weight_slider = gr.Slider(
                minimum=0.5,
                maximum=2.5,
                value=1.0,
                step=0.1,
                label="Flava emphasis weight",
                info="Adjust the emphasis weight for the Flava keyword (e.g., 1.2 produces '(keyword:1.2)')"
            )
    
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
    
    # Hidden state to store prompts and negative prompt for saving
    prompts_state = gr.State([])
    negative_prompt_state = gr.State("")
    
    # Wire up interactions
    all_checkboxes = primary_checks
    generate_btn.click(
        fn=generate_and_display,
        inputs=[shot_light_check, film_grade_check, flava_check, flava_weight_slider, wildcard_clothing_check] + all_checkboxes,
        outputs=[output_text, prompts_state, negative_prompt_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state, negative_prompt_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
