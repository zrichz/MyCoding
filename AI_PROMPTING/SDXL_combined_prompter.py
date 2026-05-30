"""
SDXL Prompt Generator - Dual Mode (Photo & Illustration)
generates 400no. SDXL prompts, with 'Primary' and 'Secondary' stages.
(although auto1111 doesn't split them like comfyui)
Date: 2026-Feb, Updated: 2026-Apr

MODES:
  - Photo: Realistic photo prompts with camera-specific terms
  - Illustration: Artwork prompts without photo-specific terms

PRIMARY STAGES (7 + Subject Identity):
  1. Subject identity (mode-specific, always included)
  2. Pose and action
  3. Framing and crop
  4. Clothing and key props
  5. Expression and gaze
  6. Body descriptors
  7. Context or location
  8. Shot and light variations (mode-specific)
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

# CHARACTER STUDY LISTS (for illustration mode)
CHARACTER_STUDY_LISTS = {
    "pose": ["T‑pose","standing in a T‑pose","arms extended horizontally","arms straight out to the sides","neutral expression, rigid stance","reference pose","model sheet pose"],
    "framing": ["full‑body shot","full‑body view","entire character visible","from head to toe","standing, centered in frame","no cropping"],
    "shot": ["wide shot","wide‑angle view","orthographic style","front‑facing","neutral studio lighting","plain background"],
    "style": ["clean silhouette","unobstructed limbs","standing on flat ground"]
}

CHARACTER_STUDY_NEGATIVE = ["no dynamic pose","no action pose","no bent arms","no foreshortening","no close‑up","no portrait crop"]

# PRIMARY STAGES (8)
# Subject identity is now mode-specific and supports single/dual subjects
SUBJECT_IDENTITY = {
    "photo_single": "amateur photo, full-body shot, boobs, a photo of a blonde woman, average build, hair up, (faint smile:0.2), (teeth:0.4)",
    "photo_dual": "amateur photo, full-body shot, boobs, a photo of two blonde women, average build, hair up, (faint smile:0.2), (teeth:0.4)",
    "illustration_single": "illustrative realism, comic-book realism, painterly realism, art, a full-length illustration of a blonde woman, hair up, smile, teeth, boobs",
    "illustration_dual": "illustrative realism, comic-book realism, painterly realism, art, a full-length illustration of two blonde women, hair up, smile, teeth, boobs"
}

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

# MODE-SPECIFIC: Shot and light variations
SHOT_LIGHT_PHOTO = ["intimate","casual","golden hour lighting mood","soft diffused lighting",
    "dynamic perspective","shot from above","shot from below","dramatic half-lighting",
]

SHOT_LIGHT_ILLUSTRATION = [
    "intimate","casual","golden hour lighting mood","soft diffused lighting","dynamic perspective",
    "view from above","view from below","soft diffused light source", "dramatic half-lighting",
]

def generate_prompts(mode, subject_count, primary_enabled, shot_light_enabled, character_study=False, use_wildcard_clothing=False):
    """Generate 400 combined prompts based on mode, subject count, and enabled stages."""
    prompts = []
    
    # Select mode-specific options
    subject_key = f"{mode}_{subject_count}"
    subject_identity = SUBJECT_IDENTITY[subject_key]
    shot_light_options = SHOT_LIGHT_PHOTO if mode == "photo" else SHOT_LIGHT_ILLUSTRATION
    
    for _ in range(400):
        # Generate primary prompt
        primary_parts = []
        
        # Add character study prompts if enabled
        if character_study:
            for list_name in ["pose", "framing", "shot", "style"]:
                selected = random.sample(CHARACTER_STUDY_LISTS[list_name], 2)
                primary_parts.extend(selected)
        
        primary_parts.append(subject_identity)  # Always include subject identity
        
        for stage_name, options in PRIMARY_STAGES.items():
            if primary_enabled.get(stage_name, True):
                # Special handling for clothing stage
                if stage_name == "Clothing and key props" and use_wildcard_clothing:
                    primary_parts.append(generate_wildcard_clothing())
                else:
                    primary_parts.append(random.choice(options))
        
        # Add shot and light if enabled
        if shot_light_enabled:
            primary_parts.append(random.choice(shot_light_options))
        
        combined = "; ".join(primary_parts)
        prompts.append(combined)
    
    return prompts


def generate_and_display(mode, subject_count, shot_light_check, character_study_check, wildcard_clothing_check, *checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (7 primary stages)
    primary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    # Generate prompts
    prompts = generate_prompts(mode, subject_count, primary_enabled, shot_light_check, character_study_check, wildcard_clothing_check)
    
    # Mode-specific negative prompts
    base_negative = "asian, makeup" if mode == "photo" else "(photo:1.25),(asian:1.2), makeup, loli"
    
    # Add character study negative prompts if enabled
    if character_study_check:
        char_study_neg = random.sample(CHARACTER_STUDY_NEGATIVE, 2)
        negative_prompt = ", ".join(char_study_neg) + ", " + base_negative
    else:
        negative_prompt = base_negative
    
    # Format output - show only last 8 prompts with prefix and suffix
    last_8 = prompts[-8:]
    output_lines = [f'--prompt "{prompt}" --negative_prompt "{negative_prompt}"' for prompt in last_8]
    output = "\n\n".join(output_lines)

    return output, prompts, mode, negative_prompt, character_study_check


def save_prompts(prompts_data, mode, negative_prompt_used, character_study_enabled):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    # Use the negative prompt that was generated during prompt creation
    negative_prompt = negative_prompt_used
    
    timestamp = datetime.now().strftime("%b%d_%H%M")
    suffix = "_character.txt" if character_study_enabled else ".txt"
    filename = f"AI_PROMPTING/400_SDXLprompts_{mode}_{timestamp}{suffix}"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "{negative_prompt}"\n')
    
    return f"Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# SDXL Combined Prompt Generator")
    gr.Markdown("Generate 400 randomized prompts with customizable primary stages")
    gr.Markdown("Choose mode (Photo or Illustration) and enable/disable stages to customize output.")
    
    # Mode and subject count selection
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["photo", "illustration"],
            value="photo",
            label="Output Mode",
            info="Choose between realistic photo prompts or illustration prompts"
        )
        subject_count_radio = gr.Radio(
            choices=["single", "dual"],
            value="single",
            label="Subject Count",
            info="Choose single female subject or two female subjects"
        )
    
    # Character Study checkbox (only for illustration mode)
    character_study_check = gr.Checkbox(
        label="Make this a Character Study",
        value=False,
        info="Adds T-pose and reference-style prompts for character reference sheets"
    )
    
    # Wildcard Clothing checkbox
    wildcard_clothing_check = gr.Checkbox(
        label="Use Wildcard Clothing (Section 4)",
        value=False,
        info="Generate clothing from wildcard files instead of built-in list"
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            gr.Markdown("*Subject identity is automatically included based on mode*")
            primary_checks = []
            # Updated defaults for 7 stages (removed Subject identity)
            primary_defaults = [True, False, False, True, False, False]  # pose, framing, clothing, expression, body, location
            for i, stage_name in enumerate(PRIMARY_STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
            
            # Add shot and light as separate checkbox
            shot_light_check = gr.Checkbox(
                label="Shot and light variations",
                value=False,
                info="Mode-appropriate lighting and framing options"
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
    
    # Hidden state to store prompts, mode, negative prompt, and character study flag for saving
    prompts_state = gr.State([])
    mode_state = gr.State("photo")
    negative_prompt_state = gr.State("")
    character_study_state = gr.State(False)
    
    # Wire up interactions
    all_checkboxes = primary_checks
    generate_btn.click(
        fn=generate_and_display,
        inputs=[mode_radio, subject_count_radio, shot_light_check, character_study_check, wildcard_clothing_check] + all_checkboxes,
        outputs=[output_text, prompts_state, mode_state, negative_prompt_state, character_study_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state, mode_state, negative_prompt_state, character_study_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
