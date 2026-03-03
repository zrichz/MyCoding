"""
SDXL Prompt Generator - SMOL VERSION
generates 400no. SDXL prompts, with 'Primary' and 'Secondary' stages.
(although auto1111 doesn't split them like comfyui)
Date: 2026-Feb

This is a smaller version with only 3 wildcards per category.

PRIMARY STAGES (9):
  1. Subject identity
  2. Pose and action
  3. TI and Lora
  4. Framing and crop
  5. Clothing and key props
  6. Expression and gaze
  7. Body descriptors
  8. Context or location
  9. Semantic technical anchors

SECONDARY CATEGORIES (4):
  1. Camera / Perspective
  2. Lens / Focal length
  3. Color grading / Film style
  4. Depth of field / Bokeh
"""

import gradio as gr
import random
from datetime import datetime

# PRIMARY STAGES (8)
PRIMARY_STAGES = {
    "Subject identity": [
        "full-body shot, legs, (nude-colored platform heels), a photo of a woman,hair up,30 years old,(smiling:0.25)",
        "full-body shot, legs, (open-toe platform heels), a photo of a woman,hair up,30 years old,(smiling:0.25)",
        "full-body shot, legs, (platform heels), a photo of a woman,hair up,30 years old,(smiling:0.25)",
    ],
    "Pose and action": [
        "three-quarter turn",
        "standing",
        "sitting",
    ],
    "TI and Lora": [
        "TI1",
        "TI2",
        "TI3",
    ],
    "Framing and crop": [
        "waist-up",
        "full body",
    ],
    "Clothing and key props": [
        "light coral botanical print minidress, simple cut",
        "navy and white striped t-shirt with denim shorts",
    ],
    "Expression and gaze": [
        "neutral expression, direct gaze to viewer",
        "candid smile, eyes to camera",
        "gentle laugh, eyes to camera",
    ],
    "Body descriptors": [
        "light sun tan",
        "toned",
    ],
    "Context or location": [
        "modern lounge",
        "modern kitchen",
        "modern hotel room",
    ],
    "Semantic technical anchors": [
        "mirror selfie",
        "candid handheld shot",
        "phone camera at chest height",
    ],
}

# SECONDARY CATEGORIES (6)
SECONDARY_CATEGORIES = {
    "Camera / Perspective": [
        "eye-level perspective",
        "slightly above eye-level",
        "slightly below eye-level",
    ],
    "Lens / Focal length": [
        "50mm standard lens look",
        "35mm environmental portrait",
        "24mm slight wide environmental",
    ],
    "Color grading / Film style": [
        "analog film look, subtle grain",
        "slightly warm, low saturation",
        "cool tones, natural look",
    ],
    "Depth of field / Bokeh": [
        "shallow depth of field, soft bokeh",
        "moderate depth, background readable",
        "deep focus, environmental detail",
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
    # Parse checkboxes (9 primary + 6 secondary = 15 total)
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
    gr.Markdown("# SDXL Combined Prompt Generator - SMOL VERSION")
    gr.Markdown("Generate 400 randomized prompts in format: `<primary> | <secondary>`")
    gr.Markdown("**Only 3 wildcards per category for faster, simpler generation**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### PRIMARY STAGES")
            primary_checks = []
            primary_defaults = [True, True, False, False, False, True, False, False, False]  # 1,2,6 enabled
            for i, stage_name in enumerate(PRIMARY_STAGES.keys()):
                primary_checks.append(gr.Checkbox(label=stage_name, value=primary_defaults[i]))
        
        with gr.Column(scale=1):
            gr.Markdown("### SECONDARY CATEGORIES")
            secondary_checks = []
            secondary_defaults = [False, False, True, False]  # 3 enabled
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
