#!/home/rich/MyCoding/venvMyCoding/bin/python
"""Gradio UI for Kokoro ONNX text-to-speech with all available voice selection."""

from __future__ import annotations

import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import soundfile as sf

from kokoro_onnx import Kokoro


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "machine_learning" / "kokoro_models"
OUTPUT_DIR = BASE_DIR / "machine_learning" / "kokoro_tts_outputs"

MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
MODEL_FALLBACK_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.onnx"
)
VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)
MODEL_PATH = MODEL_DIR / "kokoro-v1.0.onnx"
VOICES_PATH = MODEL_DIR / "voices-v1.0.bin"

# Fallback choices shown until model voices are loaded.
FALLBACK_VOICES = [
    "af_heart",
    "am_michael",
    "bf_emma",
    "bm_george",
    "ef_dora",
    "ff_siwis",
    "hf_alpha",
    "if_sara",
    "jf_alpha",
    "pf_dora",
    "zf_xiaobei",
]

VOICE_LANG_PREFIX = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
}

_kokoro: Kokoro | None = None
_voice_cache: List[str] | None = None


def _download_if_missing(url: str, target_path: Path) -> None:
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target_path)


def _download_force(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()
    urllib.request.urlretrieve(url, target_path)


def _ensure_model_files() -> None:
    _download_if_missing(MODEL_URL, MODEL_PATH)
    _download_if_missing(VOICES_URL, VOICES_PATH)


def _get_kokoro() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        _ensure_model_files()
        try:
            _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
        except Exception:
            # If files are partially downloaded or invalid, force a clean refresh.
            _download_force(MODEL_URL, MODEL_PATH)
            _download_force(VOICES_URL, VOICES_PATH)
            try:
                _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
            except Exception:
                # Final fallback to non-int8 model artifact.
                _download_force(MODEL_FALLBACK_URL, MODEL_PATH)
                _kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))
    return _kokoro


def _get_all_voices() -> List[str]:
    global _voice_cache
    if _voice_cache is None:
        _voice_cache = _get_kokoro().get_voices()
    return _voice_cache


def _lang_for_voice(voice_name: str) -> str:
    prefix = voice_name.split("_", 1)[0][:1]
    return VOICE_LANG_PREFIX.get(prefix, "en-us")


def _sanitize_filename(text: str, max_len: int = 40) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "tts"
    return cleaned[:max_len]


def generate_tts(text: str, voice: str, speed: float) -> Tuple[str | None, str | None, str]:
    if not text or not text.strip():
        return None, None, "Enter text before generating audio."

    try:
        kokoro = _get_kokoro()

        try:
            valid_voices = _get_all_voices()
        except Exception:
            valid_voices = FALLBACK_VOICES

        selected_voice = voice if voice in valid_voices else valid_voices[0]
        lang = _lang_for_voice(selected_voice)

        fallback_note = ""
        try:
            samples, sample_rate = kokoro.create(
                text=text.strip(),
                voice=selected_voice,
                speed=float(speed),
                lang=lang,
            )
        except RuntimeError as runtime_err:
            if "not supported by the espeak backend" not in str(runtime_err):
                raise
            # Some local espeak installs do not include every language pack.
            lang = "en-us"
            samples, sample_rate = kokoro.create(
                text=text.strip(),
                voice=selected_voice,
                speed=float(speed),
                lang=lang,
            )
            fallback_note = " Language fallback used: en-us."

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = _sanitize_filename(text)
        output_name = f"{prefix}_{selected_voice}_{stamp}.wav"
        output_path = OUTPUT_DIR / output_name

        sf.write(str(output_path), samples, sample_rate)

        status = (
            f"Saved WAV to {output_path} using voice {selected_voice} "
            f"at speed {speed:.2f} with language {lang}."
            f"{fallback_note}"
        )
        return str(output_path), str(output_path), status

    except Exception as exc:
        return None, None, f"TTS generation failed: {exc}"


def _build_ui() -> gr.Blocks:
    try:
        initial_voices = _get_all_voices()
    except Exception:
        initial_voices = FALLBACK_VOICES

    with gr.Blocks(title="Kokoro TTS WAV Generator", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
# Kokoro TTS WAV Generator
Type text, select any installed voice, and generate a WAV file.
""")

        text_input = gr.Textbox(
            label="Text",
            lines=8,
            placeholder="Type text to convert to speech",
        )

        with gr.Row():
            voice = gr.Dropdown(
                choices=initial_voices,
                value=initial_voices[0] if initial_voices else None,
                label="Voice",
            )
            speed = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Speed",
            )

        generate_btn = gr.Button("Generate WAV")

        audio_preview = gr.Audio(label="Audio Preview", type="filepath")
        wav_file = gr.File(label="Download WAV")
        status = gr.Textbox(label="Status")

        generate_btn.click(
            fn=generate_tts,
            inputs=[text_input, voice, speed],
            outputs=[audio_preview, wav_file, status],
        )

    return demo


demo = _build_ui()

if __name__ == "__main__":
    demo.launch(inbrowser=True)
