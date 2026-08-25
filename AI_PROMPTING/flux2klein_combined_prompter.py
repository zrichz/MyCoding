"""
F2K Prompt Generator - Photo Mode
generates 400no. F2K prompts.
Date: 2026-Feb, Updated: 2026-Aug

STAGES:
  1. Subject identity (hard-coded)
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Shot and light variations
  9. Film grade
  10. Overall Feel (with adjustable emphasis weight)
"""

import gradio as gr
import random
from datetime import datetime
import os
import json

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
SUBJECT = "a photo of a woman, blonde hair styled in a casual updo, hazel eyes, kind expression (faint smile:0.2), (teeth:0.8)"

STAGES = {
    "Pose and action": [
        "three-quarter turn","standing","lying on back","walking towards viewer","sitting, legs crossed","kneeling",
        "deep in thought","stretching","posing","leaning, relaxed","on all fours","looking","reaching, casual stance","leaning over",
        
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

# OVERALL FEEL OPTIONS
OVERALL_FEEL = [
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

def generate_prompts(primary_enabled, shot_light_enabled, film_grade_enabled, overall_feel_enabled, overall_feel_weight, use_wildcard_clothing=False):
    """Generate 400 prompts based on enabled stages."""
    prompts = []
    
    for _ in range(400):
        # Generate prompt as JSON object
        prompt_dict = {}
        
        # Always include subject identity
        prompt_dict["Subject identity"] = SUBJECT
        
        # Add primary stages if enabled
        for stage_name, options in STAGES.items():
            if primary_enabled.get(stage_name, True):
                # Special handling for clothing stage
                if stage_name == "Clothing and key props" and use_wildcard_clothing:
                    prompt_dict[stage_name] = generate_wildcard_clothing()
                else:
                    prompt_dict[stage_name] = random.choice(options)
        
        # Add shot and light if enabled
        if shot_light_enabled:
            prompt_dict["Shot and light variations"] = random.choice(SHOT_LIGHT)
        
        # Add film grade if enabled
        if film_grade_enabled:
            prompt_dict["Film grade"] = random.choice(FILM_GRADE)
        
        # Add overall feel if enabled
        if overall_feel_enabled:
            keyword = random.choice(OVERALL_FEEL)
            prompt_dict["Overall Feel"] = f"({keyword}:{overall_feel_weight})"
        
        # Convert to JSON string (one line)
        json_prompt = json.dumps(prompt_dict, ensure_ascii=False)
        prompts.append(json_prompt)
    
    return prompts


def generate_and_display(shot_light_check, film_grade_check, overall_feel_check, overall_feel_weight, wildcard_clothing_check, *checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (7 primary stages)
    primary_enabled = {}
    
    primary_names = list(STAGES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, shot_light_check, film_grade_check, overall_feel_check, overall_feel_weight, wildcard_clothing_check)
    
    # Format output - show only last 8 prompts, prettified for display
    last_8 = prompts[-8:]
    output_lines = []
    for prompt_json in last_8:
        # Parse and prettify JSON for display
        prompt_dict = json.loads(prompt_json)
        pretty_json = json.dumps(prompt_dict, ensure_ascii=False, indent=2)
        output_lines.append(pretty_json)
    
    output = "\n\n".join(output_lines)

    return output, prompts


def save_prompts(prompts_data):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    timestamp = datetime.now().strftime("%b%d_%H%M")
    filename = f"AI_PROMPTING/400_F2Kprompts_photo_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'{prompt}\n')
    
    return f"Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("F2K Photo Prompt Generator")
    gr.Markdown("Generate 400 randomized, custom photo prompts")
    gr.Markdown("Enable or disable stages to customize output.")
    
    # Wildcard Clothing checkbox
    wildcard_clothing_check = gr.Checkbox(
        label="Use Wildcard Clothing (Section 4)",
        value=False,
        info="Generate clothing from wildcard files instead of built-in list"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("*Subject ID automatically included*")
            primary_checks = []
            # Updated defaults for 7 stages (removed Subject identity)
            primary_defaults = [True, False, False, True, False, False]  # pose, framing, clothing, expression, body, location
            for i, stage_name in enumerate(STAGES.keys()):
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
            
            # Add overall feel as separate checkbox with weight slider
            overall_feel_check = gr.Checkbox(
                label="Overall Feel",
                value=False,
                info="Stylistic atmosphere keywords with adjustable emphasis"
            )
            
            overall_feel_weight_slider = gr.Slider(
                minimum=0.5,
                maximum=2.5,
                value=1.0,
                step=0.1,
                label="Overall Feel emphasis weight",
                info="Adjust the emphasis weight for the Overall Feel keyword (e.g., 1.2 produces '(keyword:1.2)'"
            )
    
    generate_btn = gr.Button("Generate 400 Prompts", variant="primary", size="lg")
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    output_text = gr.Textbox(
        label="Generated 400 Prompts (Last 8 shown in JSON format)",
        lines=30,
        max_lines=50,
        interactive=False
    )
    
    # Hidden state to store prompts and negative prompt for saving
    prompts_state = gr.State([])

    # Wire up interactions
    all_checkboxes = primary_checks
    generate_btn.click(
        fn=generate_and_display,
        inputs=[shot_light_check, film_grade_check, overall_feel_check, overall_feel_weight_slider, wildcard_clothing_check] + all_checkboxes,
        outputs=[output_text, prompts_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
