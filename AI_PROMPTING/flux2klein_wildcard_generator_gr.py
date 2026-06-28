import json
import os
import random
from datetime import datetime

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WILDCARD_DIR = os.path.join(BASE_DIR, "flux2klein_wildcards")
RECIPE_PATH = os.path.join(WILDCARD_DIR, "default_recipe.json")
SCHEMA_PATH = os.path.join(WILDCARD_DIR, "schema_flux2klein_recipe.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_wildcard_file(path):
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


def validate_recipe_shape(recipe):
    required = [
        "version",
        "subject_prefix",
        "tone_options",
        "template",
        "wildcards",
        "tone_action_files",
    ]
    missing = [key for key in required if key not in recipe]
    if missing:
        raise ValueError(f"Recipe is missing required keys: {', '.join(missing)}")


def load_recipe_and_pools():
    _ = load_json(SCHEMA_PATH)
    recipe = load_json(RECIPE_PATH)
    validate_recipe_shape(recipe)

    pools = {}
    for key, filename in recipe["wildcards"].items():
        pools[key] = load_wildcard_file(os.path.join(WILDCARD_DIR, filename))

    action_pools = {}
    for tone, filename in recipe["tone_action_files"].items():
        action_pools[tone] = load_wildcard_file(os.path.join(WILDCARD_DIR, filename))

    return recipe, pools, action_pools


def choose(rng, items, fallback):
    return rng.choice(items) if items else fallback


def generate_single_prompt(rng, recipe, pools, action_pools, tone):
    action_items = action_pools.get(tone, [])
    values = {
        "subject": recipe["subject_prefix"],
        "action": choose(rng, action_items, "smiling naturally"),
        "scene": choose(rng, pools.get("scene", []), "a casual outdoor setting"),
        "wardrobe_upper": choose(rng, pools.get("wardrobe_upper", []), "a simple top"),
        "wardrobe_lower": choose(rng, pools.get("wardrobe_lower", []), "casual bottoms"),
        "outerwear_or_prop": choose(rng, pools.get("outerwear_or_prop", []), "minimal accessories"),
        "hair_detail": choose(rng, pools.get("hair_detail", []), "natural hair movement"),
        "lighting": choose(rng, pools.get("lighting", []), "natural soft lighting"),
        "background": choose(rng, pools.get("background", []), "a softly blurred background"),
        "camera_style": choose(rng, pools.get("camera_style", []), "a candid handheld composition"),
        "detail": choose(rng, pools.get("detail", []), "everyday realism"),
        "atmosphere": choose(rng, pools.get("atmosphere", []), "a candid social photo mood"),
    }
    return recipe["template"].format(**values)


def generate_prompts(tone_choice, count, seed_text):
    recipe, pools, action_pools = load_recipe_and_pools()

    if seed_text.strip():
        rng = random.Random(seed_text.strip())
    else:
        rng = random.Random()

    if tone_choice == "any":
        tones = recipe["tone_options"]
    else:
        tones = [tone_choice]

    prompts = []
    for _ in range(int(count)):
        tone = rng.choice(tones)
        prompts.append(generate_single_prompt(rng, recipe, pools, action_pools, tone))

    output = "\n\n".join(prompts)
    return output, prompts


def save_prompts(prompts):
    if not prompts:
        return "No prompts to save."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"flux2klein_prompts_{timestamp}.txt"
    output_path = os.path.join(BASE_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    return f"Saved {len(prompts)} prompts to {filename}."


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Flux2Klein Wildcard Prompt Generator")
    gr.Markdown("Generate prompts from a schema-backed wildcard recipe.")

    tone = gr.Dropdown(
        choices=["any", "warm_candid", "playful_irreverent", "subtle_flirty"],
        value="any",
        label="Tone"
    )
    count = gr.Slider(minimum=10, maximum=200, value=10, step=10, label="Prompt count")
    seed = gr.Textbox(label="Seed (optional)", placeholder="Type a seed for repeatable output")

    generate_btn = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Generated prompts", lines=18, max_lines=30)
    state_prompts = gr.State([])

    save_btn = gr.Button("Save prompts")
    save_status = gr.Textbox(label="Save status", interactive=False)

    generate_btn.click(
        fn=generate_prompts,
        inputs=[tone, count, seed],
        outputs=[output, state_prompts]
    )

    save_btn.click(
        fn=save_prompts,
        inputs=[state_prompts],
        outputs=[save_status]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
