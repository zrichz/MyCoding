"""Split audio into numbered WAV samples using optional names or the current date."""

import re
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from pydub import AudioSegment


def find_sections(audio, thresh_db, min_silence_ms, pad_ms, frame_ms=10):
    """Return list of (start_ms, end_ms) non-silent sections."""
    samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)
    samples /= float(1 << (8 * audio.sample_width - 1))

    frame = max(1, int(audio.frame_rate * frame_ms / 1000))
    n = len(samples) // frame
    if n == 0:
        return []
    rms = np.sqrt((samples[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    loud = 20 * np.log10(np.maximum(rms, 1e-10)) > thresh_db

    runs = []
    start = None
    for i, v in enumerate(loud):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, n])

    gap = max(1, int(min_silence_ms / frame_ms))
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    total = len(audio)
    return [
        (max(0, s * frame_ms - pad_ms), min(total, e * frame_ms + pad_ms))
        for s, e in merged
    ]


def read_names(path):
    names = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = re.sub(r"^\s*\d+\s*[.)\-:]\s*", "", line).strip()
        line = re.sub(r'[\\/:*?"<>|\s]+', "_", line).strip("_.")
        if line:
            names.append(line)
    return names


def split_file(src, names_file, out_dir, thresh_db=-50.0, min_silence_ms=300, pad_ms=5, log=print):
    """Split src into numbered sections and return the output directory."""
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    if src.suffix.lower() not in (".mp3", ".wav"):
        raise ValueError("Input must be a .mp3 or .wav file")

    out_dir = Path(out_dir) if out_dir else src.with_name(src.stem + "_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(src)
    sections = find_sections(audio, thresh_db, min_silence_ms, pad_ms)
    log(f"Found {len(sections)} non-silent sections in {len(audio) / 1000:.2f} s of audio")

    names = read_names(names_file) if names_file else []
    if names_file:
        log(f"Read {len(names)} names from {names_file}")
        if len(names) != len(sections):
            log(f"Warning: {len(names)} names but {len(sections)} sections")

    date = datetime.now().strftime("%y%m%d")
    for i, (s, e) in enumerate(sections):
        base = names[i] if i < len(names) else date
        audio[s:e].export(out_dir / f"{i + 1:04d}_{base}.wav", format="wav")

    log(f"Saved {len(sections)} files to {out_dir}")
    return out_dir



def run(audio_path, names_path, out_dir, thresh, min_silence, pad):
    if not audio_path:
        return "No audio file provided."
    lines = []
    try:
        split_file(audio_path, names_path, out_dir.strip() or "silence_splitter_output", thresh, int(min_silence), int(pad), log=lines.append)
    except Exception as exc:
        lines.append(f"Error: {exc}")
    return "\n".join(lines)


with gr.Blocks(title="Audio Silence Splitter") as demo:
    gr.Markdown(
        "## Audio Silence Splitter\n"
        "Split an audio file into numbered .wav samples, named using the optional text file "
        "or the current date."
    )
    with gr.Row():
        audio_in = gr.File(label="Audio file (.mp3 or .wav)", file_types=[".mp3", ".wav"], type="filepath")
        names_in = gr.File(label="Optional names file (.txt)", file_types=[".txt"], type="filepath")
    out_dir = gr.Textbox(label="Output directory (blank = silence_splitter_output in the current folder)")
    with gr.Row():
        thresh = gr.Slider(-80, -20, value=-50, step=1, label="Silence threshold (dBFS)")
        min_silence = gr.Slider(10, 1000, value=300, step=10, label="Minimum silence (ms)")
        pad = gr.Slider(0, 50, value=5, step=1, label="Padding (ms)")
    btn = gr.Button("Split", variant="primary")
    log = gr.Textbox(label="Log", lines=6)
    btn.click(run, [audio_in, names_in, out_dir, thresh, min_silence, pad], log)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
