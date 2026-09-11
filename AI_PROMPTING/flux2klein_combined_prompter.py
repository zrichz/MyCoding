"""
F2K Prompt Generator - Photo Mode
generates 400no. F2K prompts.
Date: 2026-Sept

STAGES:
  1. Subject identity (hard-coded)
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Shot and light variations
    9. Overall Feel (with adjustable emphasis weight)
"""

import gradio as gr
import random
from datetime import datetime
import os
import json

# WILDCARD CLOTHING LOADER
def load_wildcard_file(filename):
    """Load and parse a wildcard file, returning list of options."""
    filepath = os.path.join(os.path.dirname(__file__), "prompt_clothing_wildcards", filename)
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
    """Generate separate clothing and footwear descriptions."""
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
    
    # Construct separate clothing and footwear prompts.
    dress_parts = [p for p in [dress_color_or_pattern, dress_material, dress_type] if p]
    footwear_parts = [p for p in [footwear_color, footwear_material, footwear_type] if p]

    return {
        "clothing": " ".join(dress_parts) if dress_parts else "casual outfit",
        "footwear": " ".join(footwear_parts) if footwear_parts else "casual footwear",
    }

# PRIMARY STAGES (8)
# Subject identity
SUBJECT = "a photo of a woman, blonde hair styled in a casual updo, hazel eyes, kind expression (faint smile:0.2), (teeth:0.8)"

STAGES = {
    "Pose and action": [
        "three-quarter turn","standing","sitting","relaxing","walking towards viewer","hands on hips",
        "deep in thought","posing","looking","bending over",
        
    ],
    "Framing and crop": [
        "full body","medium shot","3/4 body portrait"
        
    ],
    "clothing": [
    ],
    "footwear": [
    ],
    "Expression and gaze": [
        "neutral expression, direct gaze to viewer", "candid, eyes to viewer", "direct eye contact",
        
    ],
    "Body descriptors": [
        "visible (freckles:0.5) on arms", "light sun tan", "toned calves", "natural posture, relaxed",
        
    ],
    "location": [ "interior", "mirror selfie", "garden", "bedroom", "exterior", ],
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

# OVERALL FEEL OPTIONS
OVERALL_FEEL = [
    "arctic",    "tropical",    "monsoon",    "desert",    "nocturnal",    "urban",    "suburban",    "industrial",    "futuristic",    "retro",    "vintage",
    "neon",    "infrared",    "thermal",    "surreal",    "glacial",    "volcanic",    "coastal",    "rain-soaked",
    "fogbound",    "windblown",    "moonlit",    "sun-drenched",    "overcast",    "misty",    "dusty",    "gritty",    "opulent",    "minimalist",
    "baroque",    "aristocratic",    "bohemian",    "arctic-blue",    "tundra",    "equatorial",    "high-altitude",    "underlit",    "overexposed",    "cinematic",
    "documentary",    "editorial",    "fashion-forward",    "hyperreal",    "monochrome",    "chromatic",    "saturated",    "desaturated",    "bleached",    "sepia",
    "analog",    "filmic",    "glamour",    "raw",    "moody",    "ethereal",    "harsh",    "ambient",    "backlit",    "rimlit",    "sunset-grade",    "twilight",
    "nebulous",    "cosmic",    "tropical",    "coastal",  "lush",    "windswept",    "smoky",    "holographic",    "chromatic-aberration",    "bokeh-rich",    "macro-styled",    "telephoto-styled"
]

def generate_prompts(primary_enabled, shot_light_enabled, overall_feel_enabled, overall_feel_weight):
    """Generate 400 prompts based on enabled stages."""
    prompts = []
    
    for _ in range(400):
        # Generate prompt as JSON object
        prompt_dict = {}
        
        # Always include subject identity
        prompt_dict["Subject identity"] = SUBJECT
        
        # Add primary stages if enabled
        wildcard_clothing = None
        for stage_name, options in STAGES.items():
            if primary_enabled.get(stage_name, True):
                # Generate the paired wildcard values once per prompt.
                if stage_name in ("clothing", "footwear"):
                    if wildcard_clothing is None:
                        wildcard_clothing = generate_wildcard_clothing()
                    prompt_dict[stage_name] = wildcard_clothing[stage_name]
                else:
                    prompt_dict[stage_name] = random.choice(options)
        
        # Add shot and light if enabled
        if shot_light_enabled:
            prompt_dict["Shot and light variations"] = random.choice(SHOT_LIGHT)
        
        # Add overall feel if enabled
        if overall_feel_enabled:
            keyword = random.choice(OVERALL_FEEL)
            prompt_dict["Overall Feel"] = f"({keyword}:{overall_feel_weight})"
        
        # Convert to JSON string (one line)
        json_prompt = json.dumps(prompt_dict, ensure_ascii=False)
        prompts.append(json_prompt)
    
    return prompts


def _with_article(value):
    """Add a simple indefinite article to a location phrase."""
    if not value:
        return ""
    article = "an" if value[0].lower() in "aeiou" else "a"
    return f"{article} {value}"


def format_natural_prompt(prompt_json):
    """Convert one internal JSON prompt into natural language."""
    prompt = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
    subject = prompt.get("Subject identity", "").strip().rstrip(".")
    paragraphs = [f"{subject}."] if subject else []

    scene_parts = []
    pose = prompt.get("Pose and action")
    location = prompt.get("location")
    clothing = prompt.get("clothing")
    footwear = prompt.get("footwear")

    if pose:
        scene_parts.append(f"She is {pose}")
    if location:
        scene_parts.append(f"in {_with_article(location)}")
    if clothing or footwear:
        worn_items = []
        if clothing:
            worn_items.append(clothing)
        if footwear:
            worn_items.append(footwear)
        scene_parts.append(f"wearing {' and '.join(worn_items)}")
    if scene_parts:
        paragraphs.append(" ".join(scene_parts) + ".")

    appearance_parts = []
    expression = prompt.get("Expression and gaze")
    body = prompt.get("Body descriptors")
    if expression:
        appearance_parts.append(expression)
    if body:
        appearance_parts.append(body)
    if appearance_parts:
        paragraphs.append(f"She has {', '.join(appearance_parts)}.")

    capture_parts = []
    shot = prompt.get("Shot and light variations")
    framing = prompt.get("Framing and crop")
    if shot:
        capture_parts.append(shot)
    if framing:
        capture_parts.append(f"{framing}")
    if capture_parts:
        paragraphs.append(f"The photo is {', '.join(capture_parts)}.")

    overall_feel = prompt.get("Overall Feel")
    if overall_feel:
        paragraphs.append(f"Overall Feel: {overall_feel}")

    return "\n".join(paragraphs)


def generate_and_display(shot_light_check, overall_feel_check, overall_feel_weight, *checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (7 primary stages)
    primary_enabled = {}
    
    primary_names = list(STAGES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    # Generate prompts
    prompts = generate_prompts(primary_enabled, shot_light_check, overall_feel_check, overall_feel_weight)
    
    # Format output - show only the last 8 prompts in natural language.
    last_8 = prompts[-8:]
    output_lines = [format_natural_prompt(prompt_json) for prompt_json in last_8]
    
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
            f.write(f'{format_natural_prompt(prompt)}\n\n')
    
    return f"Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("F2K Photo Prompt Generator")
    gr.Markdown("Generates 400 randomized photo prompts")
    gr.Markdown("Enable or disable stages to customize.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("*Subject ID automatically included*")
            primary_checks = []
            # Defaults for the 7 primary stages (subject identity is automatic).
            primary_defaults = [True, False, False, False, True, False, False]
            for i, stage_name in enumerate(STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
            
            # Add shot and light as separate checkbox
            shot_light_check = gr.Checkbox(
                label="Shot and light variations",
                value=False,
                info="Lighting and framing options"
            )
            
            # Add overall feel as separate checkbox with weight slider
            overall_feel_check = gr.Checkbox(
                label="Overall Feel",
                value=False,
                info="Stylistic atmosphere keywords with adjustable emphasis"
            )
            
            overall_feel_weight_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="Overall Feel emphasis weight",
                info="0.1 to 2.0"
            )
    
    generate_btn = gr.Button("Generate 400 Prompts", variant="primary", size="lg")
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    output_text = gr.Textbox(
        label="Generated 400 Prompts (Last 8 shown in natural language)",
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
        inputs=[shot_light_check, overall_feel_check, overall_feel_weight_slider] + all_checkboxes,
        outputs=[output_text, prompts_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
