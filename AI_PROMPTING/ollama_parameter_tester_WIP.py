"""
Ollama Parameter Tester

This script sends the same image-prompt request to a local Ollama model using
multiple parameter presets and seed values, then saves all responses to a
single text file beside the script.
"""

from pathlib import Path
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
URL = "http://localhost:11434/api/generate"
MODEL = "Weird24_Q4_verB"
SYSTEM_PROMPT = "You are a creative assistant that writes vivid, detailed image generation prompts."
PROMPT = (
    "Provide a single, highly descriptive prompt for an AI-image generator, "
    "where a young female subject is depicted in a detailed and immersive scene. "
    "the final image will be a detailed illustration, not a photograph. "
    "Do NOT use the word \"anime\" in your prompt."
    "Use words and descriptions that will ensure the generation will be of an illustration, for example using words such as \"rendered\" and \"palette\""
    "Keep the prompt to a concise yet vivid paragraph of around 200 words. "
    "Only include the final generated prompt in your response."
)
FULL_PROMPT = f"{SYSTEM_PROMPT}\n{PROMPT}"
SEEDS = [42, 100, 256, 777]
OUTPUT_FILE = "ollama_parameter_test_results.txt"
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
    "recommended": {
        "label": "Recommended (min_p + high temp)",
        "params": {
            "temperature": 0.95,
            "min_p": 0.05,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.15,
            "repeat_last_n": 256,
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
            "repeat_last_n": 256,
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
            "repeat_last_n": 256,
        },
    },
}


def generate_response(seed, params):
    options = {**params, "seed": seed, "num_predict": NUM_PREDICT}
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "prompt": FULL_PROMPT,
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
    output_path = SCRIPT_DIR / OUTPUT_FILE
    with output_path.open("w", encoding="utf-8") as output_file:
        for config in PRESETS.values():
            print(f"Generating {len(SEEDS)} responses for {config['label']}...")
            output_file.write(f"=== {config['label'].upper()} ===\n")
            output_file.write(f"Model: {MODEL} | Prompt: {FULL_PROMPT}\n")
            output_file.write(f"Parameters: {config['params']}\n")
            output_file.write(f"num_predict: {NUM_PREDICT} | Timeout: {TIMEOUT_SECONDS}s\n")
            output_file.write("=" * 60 + "\n\n")

            for idx, seed in enumerate(SEEDS, start=1):
                failed = False
                try:
                    result = generate_response(seed, config["params"])
                except (requests.RequestException, requests.exceptions.JSONDecodeError, ValueError) as error:
                    print(f"  -> Failed [{idx}/{len(SEEDS)}] (Seed: {seed}): {error}")
                    result = FAILED_TEXT
                    failed = True

                output_file.write(f"[{idx}] (Seed: {seed})\n")
                output_file.write(f"{result}\n\n")
                output_file.write("-" * 60 + "\n\n")

                if not failed:
                    print(f"  -> Completed [{idx}/{len(SEEDS)}] (Seed: {seed}): {result[:20]}...")

    print(f"\nDone. Saved '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
