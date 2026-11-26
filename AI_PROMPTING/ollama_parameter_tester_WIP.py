"""
Ollama Parameter Tester

This script sends the same request to each local Ollama model in sequence using
multiple parameter presets and seed values, then saves all responses to a
single text file beside the script. Replace the placeholder model name before use.
"""

from pathlib import Path
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
URL = "http://localhost:11434/api/generate"
MODELS = ["Weird24_Q4_verB", "BC24_Q4_verB"]
PROMPTS = {
    "1": (
        "Illustration",
        "You are a creative assistant that writes vivid, detailed image generation prompts.",
        (
            "Provide a single, highly descriptive prompt for an AI-image generator, "
            "where a young female subject is depicted in a detailed and immersive scene. "
            "The final image will be a detailed illustration, not a photograph. "
            "Do NOT use the word \"anime\" in your prompt. "
            "Use words and descriptions that will ensure the generation will be of an illustration, "
            "for example using words such as \"rendered\" and \"palette\". "
            "Keep the prompt to a concise yet vivid paragraph of around 200 words. "
            "Only include the final generated prompt in your response."
        ),
    ),
    "2": (
        "Photo",
        "You are a creative assistant that writes vivid, detailed image generation prompts.",
        (
            "Provide a single, highly descriptive prompt for an AI image generator, "
            "where a young female subject is depicted in a detailed and immersive scene. "
            "The final image should be a photorealistic photograph, not an illustration. "
            "Do NOT use the word \"anime\" in your prompt. "
            "Use concrete photographic details such as natural lighting, lens choice, "
            "composition, and realistic textures. "
            "Keep the prompt to a concise yet vivid paragraph of around 200 words. "
            "Only include the final generated prompt in your response."
        ),
    ),
    "3": (
        "Creative Writing Solo",
        "You are a creative writing assistant.",
        (
            "Write a concise, vivid scene of around 200 words in which one character "
            "speaks directly to the reader. Give the character a distinct voice and "
            "make the reader feel personally addressed. Include only this one speaker; "
            "do not add dialogue from other characters. "
            "Only include the finished scene in your response."
        ),
    ),
    "4": (
        "Creative Writing Duo",
        "You are a creative writing assistant.",
        (
            "Write a concise, vivid scene of around 200 words in which two distinct "
            "characters have a dialogue. Give each character a recognizable voice and "
            "let their exchange reveal the scene, their relationship, or a point of "
            "tension. Include only these two speaking characters. "
            "Only include the finished scene in your response."
        ),
    ),
}
SEEDS = [1,2,3]
OUTPUT_FILE_PREFIX = "ollama_parameter_test_results"
OUTPUT_SUFFIXES = {
    "1": "illust",
    "2": "photo",
    "3": "solo",
    "4": "duo",
}
NUM_PREDICT = 500
TIMEOUT_SECONDS = 180
FAILED_TEXT = "(timed out/errored)"

PRESETS = {
    "defaults": {
        "label": "Ollama Defaults",
        "params": {
            "temperature": 0.8,
            "min_p": 0.0,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
        },
    },
    
    "high_temperature": {
        "label": "High temperature",
        "params": {
            "temperature": 1.025,
            "min_p": 0.05,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.2,
            "repeat_last_n": 128,
        },
    },
    "higher_temperature": {
        "label": "Higher temperature",
        "params": {
            "temperature": 1.1,
            "min_p": 0.1,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.3,
            "repeat_last_n": 128,
        },
    },
}


def choose_prompt():
    print("Choose a prompt role:")
    for choice, (role, _, _) in PROMPTS.items():
        print(f"{choice}. {role}")

    while True:
        choice = input("Choose option 1-4: ").strip()
        if choice in PROMPTS:
            return choice, *PROMPTS[choice]
        print("Invalid selection. Please choose 1, 2, 3, or 4.")


def generate_response(model, seed, params, full_prompt):
    options = {**params, "seed": seed, "num_predict": NUM_PREDICT}
    response = requests.post(
        URL,
        json={
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": options,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("response"), str):
        raise ValueError("Ollama returned an unexpected response format.")
    return data["response"].strip()


def main():
    choice, role, system_prompt, prompt = choose_prompt()
    full_prompt = f"{system_prompt}\n{prompt}"
    print(f"Selected prompt role: {role}")

    output_path = SCRIPT_DIR / f"{OUTPUT_FILE_PREFIX}_{OUTPUT_SUFFIXES[choice]}.txt"
    with output_path.open("w", encoding="utf-8") as output_file:
        for model in MODELS:
            print(f"Testing model: {model}")
            output_file.write(f"=== MODEL: {model} ===\n\n")
            for config in PRESETS.values():
                print(f"Generating {len(SEEDS)} responses for {config['label']}...")
                output_file.write(f"=== {config['label'].upper()} ===\n")
                output_file.write(f"Role: {role}\n")
                output_file.write(f"Model: {model} | Prompt: {full_prompt}\n")
                output_file.write(f"Parameters: {config['params']}\n")
                output_file.write(f"num_predict: {NUM_PREDICT} | Timeout: {TIMEOUT_SECONDS}s\n")
                output_file.write("=" * 60 + "\n\n")

                for idx, seed in enumerate(SEEDS, start=1):
                    failed = False
                    try:
                        result = generate_response(model, seed, config["params"], full_prompt)
                    except (requests.RequestException, requests.exceptions.JSONDecodeError, ValueError) as error:
                        print(f"  -> Failed [{idx}/{len(SEEDS)}] (Seed: {seed}): {error}")
                        result = FAILED_TEXT
                        failed = True

                    output_file.write(f"[{idx}] (Seed: {seed})\n")
                    output_file.write(f"{result}\n\n")
                    output_file.write("-" * 60 + "\n\n")

                    if not failed:
                        print(f"  -> Completed [{idx}/{len(SEEDS)}] (Seed: {seed}): {result[:20]}...")

    print(f"\nDone. Saved '{output_path.name}'.")


if __name__ == "__main__":
    main()
