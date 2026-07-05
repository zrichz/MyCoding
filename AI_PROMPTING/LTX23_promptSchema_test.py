import json
import os
import random
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WILDCARD_DIR = os.path.join(BASE_DIR, "flux2klein_wildcards")


def load_wildcard_file(path):
    """Load items from a wildcard file, handling comments and comma-separated values."""
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_items = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        raw_items.extend(part.strip() for part in stripped.split(","))

    return [item for item in raw_items if item]


def load_wildcard_pools():
    """Load all wildcard pools for the LTX2 schema."""
    pools = {
        "shot_scale": load_wildcard_file(os.path.join(WILDCARD_DIR, "shot_scale.txt")),
        "genre_or_style": load_wildcard_file(os.path.join(WILDCARD_DIR, "genre_or_style.txt")),
        "environment": load_wildcard_file(os.path.join(WILDCARD_DIR, "environment_ltx.txt")),
        "lighting_and_mood": load_wildcard_file(os.path.join(WILDCARD_DIR, "lighting_and_mood_ltx.txt")),
        "action_sequence": load_wildcard_file(os.path.join(WILDCARD_DIR, "action_sequence_ltx.txt")),
        "camera_movement": load_wildcard_file(os.path.join(WILDCARD_DIR, "camera_movement_ltx.txt")),
        "audio_cues": load_wildcard_file(os.path.join(WILDCARD_DIR, "audio_cues_ltx.txt")),
    }
    return pools


class LTX2PromptSchema(BaseModel):
    # 1. Establish the Shot
    shot_scale: str = Field(
        ..., 
        description="Cinematography terms (e.g., 'Wide shot', 'Macro shot')."
    )
    genre_or_style: str = Field(
        ..., 
        description="Film genre or aesthetic (e.g., 'Film noir', 'Analog film look')."
    )
    
    # 2. Set the Scene
    environment: str = Field(
        ..., 
        description="Location and conditions (e.g., 'Rain-slick pavement', 'Neon-lit alley')."
    )
    lighting_and_mood: str = Field(
        ..., 
        description="Light control (e.g., 'Soft rim light', 'Golden hour')."
    )
    
    # 3. Define the Character(s) & Physics
    character_details: Optional[str] = Field(
        None, 
        description="Physical cues (age, clothing, posture) only."
    )
    action_sequence: str = Field(
        ..., 
        description="Core movement, present tense verbs (e.g., 'walks', 'tilts')."
    )
    
    # 4. Identify Camera Movement
    camera_movement: str = Field(
        ..., 
        description="Motion relative to the subject (e.g., 'slow dolly in')."
    )
    
    # 5. Describe the Audio & Dialogue
    audio_cues: Optional[str] = Field(
        None, 
        description="Ambient sounds or speech in quotation marks."
    )
    
    # 6. Guardrails
    guardrails: List[str] = Field(
        default=["no text overlays", "no logos"],
        description="Negative constraints to minimize generation errors."
    )

    def generate_raw_prompt(self) -> str:
        """Assembles the schema into a single, cohesive paragraph."""
        paragraph = f"A {self.genre_or_style} {self.shot_scale.lower()} set in a {self.environment.lower()}. "
        paragraph += f"The scene features {self.lighting_and_mood.lower()}. "
        if self.character_details:
            paragraph += f"{self.character_details.strip()} "
        paragraph += f"{self.action_sequence.strip()} "
        paragraph += f"The camera executes a {self.camera_movement.lower()}. "
        if self.audio_cues:
            paragraph += f"Audio features {self.audio_cues.strip()}. "
        if self.guardrails:
            paragraph += f"Strictly avoid: {', '.join(self.guardrails)}."
        return paragraph


def choose(rng, items, fallback):
    """Choose a random item or fallback."""
    return rng.choice(items) if items else fallback


def generate_single_prompt(rng, pools, character_details, custom_guardrails):
    """Generate a single LTX2 prompt using wildcards."""
    values = {
        "shot_scale": choose(rng, pools.get("shot_scale", []), "Wide shot"),
        "genre_or_style": choose(rng, pools.get("genre_or_style", []), "Science fiction"),
        "environment": choose(rng, pools.get("environment", []), "a futuristic cityscape"),
        "lighting_and_mood": choose(rng, pools.get("lighting_and_mood", []), "neon lighting"),
        "action_sequence": choose(rng, pools.get("action_sequence", []), "walks forward"),
        "camera_movement": choose(rng, pools.get("camera_movement", []), "slow dolly in"),
        "audio_cues": choose(rng, pools.get("audio_cues", []), "electronic ambient sound"),
    }
    
    prompt = LTX2PromptSchema(
        shot_scale=values["shot_scale"],
        genre_or_style=values["genre_or_style"],
        environment=values["environment"],
        lighting_and_mood=values["lighting_and_mood"],
        character_details=character_details if character_details.strip() else None,
        action_sequence=values["action_sequence"],
        camera_movement=values["camera_movement"],
        audio_cues=values["audio_cues"],
        guardrails=custom_guardrails if custom_guardrails else ["no text overlays", "no logos"]
    )
    
    return prompt.generate_raw_prompt()


def generate_prompts(count, character_details, custom_guardrails, seed_text):
    """Generate multiple LTX2 prompts."""
    pools = load_wildcard_pools()
    
    if seed_text.strip():
        rng = random.Random(seed_text.strip())
    else:
        rng = random.Random()
    
    prompts = []
    for _ in range(int(count)):
        prompts.append(generate_single_prompt(rng, pools, character_details, custom_guardrails.split(",") if custom_guardrails.strip() else []))
    
    output = "\n\n".join(prompts)
    return output, prompts


def save_prompts(prompts):
    """Save generated prompts to a file."""
    if not prompts:
        return "No prompts to save."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"ltx23_prompts_{timestamp}.txt"
    output_path = os.path.join(BASE_DIR, filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n\n")
    
    return f"Saved {len(prompts)} prompts to {filename}."


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# LTX23 Video Prompt Generator")
    gr.Markdown("Generate cinematic prompts using the LTX2 schema with wildcard-based randomization.")
    
    with gr.Row():
        count = gr.Slider(minimum=1, maximum=100, value=5, step=1, label="Prompt count")
        seed = gr.Textbox(label="Seed (optional)", placeholder="Type a seed for repeatable output")
    
    with gr.Row():
        character_details = gr.Textbox(
            label="Character Details (optional)",
            placeholder="E.g., A figure in a sleek black jacket, cybernetic implants visible",
            lines=2
        )
    
    with gr.Row():
        custom_guardrails = gr.Textbox(
            label="Custom Guardrails (comma-separated, optional)",
            placeholder="E.g., no text overlays, no logos, no watermarks",
            lines=2
        )
    
    generate_btn = gr.Button("Generate Prompts", variant="primary")
    output = gr.Textbox(label="Generated prompts", lines=20, max_lines=30)
    state_prompts = gr.State([])
    
    save_btn = gr.Button("Save prompts")
    save_status = gr.Textbox(label="Save status", interactive=False)
    
    generate_btn.click(
        fn=generate_prompts,
        inputs=[count, character_details, custom_guardrails, seed],
        outputs=[output, state_prompts]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[state_prompts],
        outputs=[save_status]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
