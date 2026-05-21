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

SECONDARY OPTIONS:
  Mode-specific list of 18 options covering:
  - Color grading / Color treatment
  - Depth of field (photo) / Detail rendering (illustration)
  - Texture / Finish
  - Mood / Subtle effects
  (3 random options selected per prompt when enabled)
"""

import gradio as gr
import random
from datetime import datetime

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
        "navy blue lace bralette with strappy back detail",
        "black string bikini with minimal coverage",
        "white open blouse with cheeky bottoms",
        "sheer white mesh beach cover-up with nothing underneath",
        "ripped denim micro shorts with matching bandeau top",
        "cotton crop top with matching hot pants",
        "iridescent silver micro dress",
        "black tactical harness with minimal coverage",
        "cream pattern bodysuit",
        "white micro dress",
        "white lingerie",
        "black leather fantasy harness with minimal fabric coverage",
        "light blue sheer micro dress",
        "lace-trimmed micro dress",
        "red latex catsuit unzipped to navel",
        "black latex catsuit unzipped to navel",
        "sheer lace robe over matching thong and bra",
        "choker with matching thong bodysuit",
        "white wet t-shirt over skimpy bikini bottoms",
        "silk kimono falling off one shoulder with minimal undergarments",
        "bikini with side ties",
        "barely-there sling bikini in shimmering fabric",
        "open-front cardigan over lace bralette and panties",
        "shredded band t-shirt exposing sides with denim cut-offs",
        
    ],
    "Expression and gaze": [
        "neutral expression, direct gaze to viewer", "candid, eyes to viewer", "direct eye contact",
        
    ],
    "Body descriptors": [
        "visible (freckles:0.5) on arms", "light sun tan", "toned calves", "natural posture, relaxed",
        
    ],
    "location": [ "interior", "selfie", "garden", "bedroom", "exterior", ],
}

# MODE-SPECIFIC: Shot and light variations
SHOT_LIGHT_PHOTO = ["intimate","casual","golden hour lighting mood","soft diffused lighting",
    "dynamic perspective","shot from above","shot from below","dramatic half-lighting",
]

SHOT_LIGHT_ILLUSTRATION = [
    "intimate","casual","golden hour lighting mood","soft diffused lighting","dynamic perspective",
    "view from above","view from below","soft diffused light source", "dramatic half-lighting",
]

# SECONDARY CATEGORIES - Split by mode
SECONDARY_PHOTO = [
    "clean digital look, minimal processing",
    "slightly warm, low saturation",
    "slightly cool tones, natural look",
    "faded film look, low contrast",
    "cinematic teal-orange, very subtle",
    "subtle film grain",
    "natural, documentary tone",
    "quiet, candid mood",
    "cheerful, candid",
    "subtle story-telling, natural",
]

SECONDARY_ILLUSTRATION = [
    # Color treatment
    "clean digital colors, minimal noise",
    "slightly warm color palette",
    "slightly cool color palette",
    "subtle color harmony, low contrast",
    "cinematic color balance, teal-orange hints",
    # Depth and detail
    "detailed foreground, soft background",
    "moderate detail throughout",
    "sharp detail, environmental context",
    "soft background rendering",
    "gentle background separation",
    "soft surrounding elements, sharp subject",
    # Texture / Finish
    "subtle canvas texture",
    "smooth digital finish",
    # Mood / Style effects
    "natural, narrative style",
    "quiet, intimate mood",
    "subtle warmth, inviting",
    "cheerful, engaging",
    "subtle storytelling atmosphere",
]

def generate_prompts(mode, subject_count, primary_enabled, shot_light_enabled, use_secondary, character_study=False):
    """Generate 400 combined prompts based on mode, subject count, and enabled stages."""
    prompts = []
    
    # Select mode-specific options
    subject_key = f"{mode}_{subject_count}"
    subject_identity = SUBJECT_IDENTITY[subject_key]
    shot_light_options = SHOT_LIGHT_PHOTO if mode == "photo" else SHOT_LIGHT_ILLUSTRATION
    secondary_options = SECONDARY_PHOTO if mode == "photo" else SECONDARY_ILLUSTRATION
    
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
                primary_parts.append(random.choice(options))
        
        # Add shot and light if enabled
        if shot_light_enabled:
            primary_parts.append(random.choice(shot_light_options))
        
        primary = "; ".join(primary_parts)
        
        # Generate secondary prompt (pick random items from mode-specific list)
        if use_secondary:
            # Pick 3 random secondary options
            num_secondary = min(3, len(secondary_options))
            secondary_parts = random.sample(secondary_options, num_secondary)
            secondary = ", ".join(secondary_parts)
            # Combine in format: <primary> | <secondary>
            combined = f"{primary} | {secondary}"
        else:
            combined = primary
        
        prompts.append(combined)
    
    return prompts


def generate_and_display(mode, subject_count, shot_light_check, character_study_check, *checkboxes):
    """Generate prompts and return formatted text with save option."""
    # Parse checkboxes (7 primary stages + 1 secondary = 8 total)
    primary_enabled = {}
    
    primary_names = list(PRIMARY_STAGES.keys())
    
    for i, name in enumerate(primary_names):
        primary_enabled[name] = checkboxes[i]
    
    # Last checkbox is for secondary
    use_secondary = checkboxes[len(primary_names)]
    
    # Generate prompts
    prompts = generate_prompts(mode, subject_count, primary_enabled, shot_light_check, use_secondary, character_study_check)
    
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
    gr.Markdown("Generate 400 randomized prompts in format: `<primary> | <secondary>`")
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
        
        with gr.Column(scale=1):
            gr.Markdown("### SECONDARY OPTIONS")
            secondary_check = gr.Checkbox(
                label="Use Secondary (3 random from mode-specific list)", 
                value=True,
                info="Includes color, depth, texture, and mood options appropriate to the selected mode"
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
    all_checkboxes = primary_checks + [secondary_check]
    generate_btn.click(
        fn=generate_and_display,
        inputs=[mode_radio, subject_count_radio, shot_light_check, character_study_check] + all_checkboxes,
        outputs=[output_text, prompts_state, mode_state, negative_prompt_state, character_study_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state, mode_state, negative_prompt_state, character_study_state],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
