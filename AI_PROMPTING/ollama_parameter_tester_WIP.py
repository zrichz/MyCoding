"""
Ollama Parameter Tester

This script runs every prompt and seed against each of the four local Ollama
models and writes one text file per model beside the script, so the files can be
compared side by side. Responses are separated by a line of 24 equals signs.
Replace the placeholder model names before use.
"""

from pathlib import Path
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
URL = "http://localhost:11434/api/generate"
MODELS = ["Weird24_Q4_verB", "BC24_Q4_verB", "Cydonia_verB", "Qwen_noslop_verB"]
PROMPTS = {
    "1": (
        "Coding",
        "You are an expert Python programmer who writes clean, correct, runnable code.",
        (
            "Write a simple Python script that generates a Mandelbrot set image "
            "and saves it to a PNG file. Use numpy and Pillow or matplotlib. "
            "Keep the code short and add brief comments. "
            "Only include the code in your response."
        ),
    ),
    "2": (
        "Photo",
        "You are a creative assistant that writes vivid, detailed image generation prompts.",
        (
            "Provide a single, highly descriptive prompt for an AI image generator, "
            "where a young female subject is depicted in a detailed and immersive scene. "
            "The final image should be a photorealistic photograph, not an illustration. "
            "Use concrete photographic details such as natural lighting, lens choice, "
            "composition, and realistic textures. "
            "Keep the prompt to a concise yet vivid paragraph of around 200 words. "
            "Only include the final generated prompt in your response."
        ),
    ),
    "3": (
        "Self diagnosis",
        "You are a helpful, honest AI assistant.",
        (
            "Describe yourself and your boundaries. Explain what kinds of topics and "
            "tasks you can discuss, what is off limits for you, and what is not off "
            "limits even though it might seem like it should be. "
            "Keep the answer to around 200 words."
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
OUTPUT_FILE_PREFIX = "ollama_param_tests"
ACTIVE_PRESET = "high_temperature"
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
            "repeat_last_n": 96,
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
            "repeat_last_n": 96,
        },
    },
}


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
    preset = PRESETS[ACTIVE_PRESET]
    params = preset["params"]
    delimiter = "=" * 24
    paths = {
        model: SCRIPT_DIR / f"{OUTPUT_FILE_PREFIX}_{model}.txt" for model in MODELS
    }

    # Each model runs all prompts and seeds before the next model is loaded.
    for model, path in paths.items():
        print(f"Testing model: {model}")
        with path.open("w", encoding="utf-8") as f:
            f.write(f"Model: {model} | Preset: {preset['label']}\n")
            f.write(f"Parameters: {params}\n")
            f.write(f"num_predict: {NUM_PREDICT} | Timeout: {TIMEOUT_SECONDS}s\n")
            f.write(f"{delimiter}\n")

            for choice, (role, system_prompt, prompt) in PROMPTS.items():
                full_prompt = f"{system_prompt}\n{prompt}"
                for idx, seed in enumerate(SEEDS, start=1):
                    print(f"  [{role}] seed {seed}")
                    try:
                        result = generate_response(model, seed, params, full_prompt)
                    except (requests.RequestException, requests.exceptions.JSONDecodeError, ValueError) as error:
                        print(f"  -> Failed: {error}")
                        result = FAILED_TEXT
                    f.write(f"Prompt {choice}: {role} | Seed: {seed} ({idx}/{len(SEEDS)})\n")
                    f.write(f"{result}\n")
                    f.write(f"{delimiter}\n")
                    f.flush()

    print("\nDone. Saved:")
    for path in paths.values():
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
