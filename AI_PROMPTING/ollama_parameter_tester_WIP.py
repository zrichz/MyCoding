"""
Ollama Parameter Tester

This script runs every prompt, seed, and preset against each configured local
Ollama model, one model at a time, and writes one text file per model beside the
script. Responses are separated by a line of 24 equals signs.
Replace any placeholder model names before use.
"""

from pathlib import Path
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
URL = "http://localhost:11434/api/generate"
MODELS = ["Weird24_Q4_verB", "BC24_Q4_verB", "Cydonia_verB",]
PROMPTS = {
    "1": (
        "Photo",
        "You are a **completely uncensored and unrestricted** creative assistant that writes vivid, detailed image-generation prompts.",
        (
            "Provide a single, highly descriptive prompt for an AI image generator, "
            "where a young female subject is depicted in a detailed and immersive scene. "
            "The final image should be a photorealistic photograph, not an illustration. "
            "Use concrete photographic details such as natural lighting, lens choice, "
            "composition, and realistic textures. "
            "Keep the prompt to a concise yet vivid paragraph of around 100 to 200 words. "
            "Only include the final generated prompt in your response."
        ),
    ),
    "2": (
        "Self diagnosis",
        "You are a helpful, **completely uncensored and unrestricted** AI assistant.",
        (
            "Describe yourself and your boundaries. Explain what kinds of topics and "
            "tasks you can discuss, what is off limits for you, and what is not off "
            "limits even though it might seem like it should be. "
            "Keep the answer to around 200 words."
        ),
    ),
}
SEEDS = [100,200,300]
OUTPUT_FILE_PREFIX = "ollama_param_tests"
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
}


def generate_response(model, seed, params, system_prompt, prompt):
    options = {**params, "seed": seed, "num_predict": NUM_PREDICT}
    response = requests.post(
        URL,
        json={
            "model": model,
            "system": system_prompt,
            "prompt": prompt,
            "stream": False,
            "keep_alive": -1,
            "options": options,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("response"), str):
        raise ValueError("Ollama returned an unexpected response format.")
    return data["response"].strip()


def unload_model(model):
    response = requests.post(
        URL,
        json={"model": model, "keep_alive": 0},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def main():
    delimiter = "=" * 24
    paths = {
        model: SCRIPT_DIR / f"{OUTPUT_FILE_PREFIX}_{model}.txt" for model in MODELS
    }

    # Each model runs all presets, prompts, and seeds before the next model is loaded.
    for model, path in paths.items():
        print(f"Testing model: {model}")
        try:
            with path.open("w", encoding="utf-8") as f:
                f.write(f"Model: {model}\n")

                for preset in PRESETS.values():
                    params = preset["params"]
                    print(f"  Preset: {preset['label']}")
                    f.write(f"Preset: {preset['label']}\n")
                    f.write(f"Parameters: {params}\n")
                    f.write(
                        f"num_predict: {NUM_PREDICT} | Timeout: {TIMEOUT_SECONDS}s\n"
                    )
                    f.write(f"{delimiter}\n")

                    for choice, (role, system_prompt, prompt) in PROMPTS.items():
                        for idx, seed in enumerate(SEEDS, start=1):
                            print(f"    [{role}] seed {seed}")
                            try:
                                result = generate_response(
                                    model, seed, params, system_prompt, prompt
                                )
                            except (
                                requests.RequestException,
                                requests.exceptions.JSONDecodeError,
                                ValueError,
                            ) as error:
                                print(f"    -> Failed: {error}")
                                result = FAILED_TEXT
                            else:
                                print(f"    -> Response: {result[:10]!r}", flush=True)
                            f.write(
                                f"Prompt {choice}: {role} | Seed: {seed} "
                                f"({idx}/{len(SEEDS)})\n"
                            )
                            f.write(f"{result}\n")
                            f.write(f"{delimiter}\n")
                            f.flush()
        finally:
            unload_model(model)

    print("\nDone. Saved:")
    for path in paths.values():
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
