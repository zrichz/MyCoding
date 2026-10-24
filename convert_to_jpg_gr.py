"""Gradio wrapper for convert_to_jpg.py: convert all images in a directory to JPG."""
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

from convert_to_jpg import convert_file


def browse_directory(current: str) -> str:
    """Open a native folder picker (the app runs locally) and return the choice."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        initial = current if current and Path(current).is_dir() else str(Path.home())
        chosen = filedialog.askdirectory(initialdir=initial, title="Select directory to convert")
        root.destroy()
        return chosen or current
    except Exception:
        return current


def convert_directory(input_dir: str, output_dir: str, quality: int, recursive: bool):
    src_dir = Path(input_dir.strip()) if input_dir else None
    if not src_dir or not src_dir.is_dir():
        yield "Please choose a valid directory.", []
        return

    out_dir = Path(output_dir.strip()) if output_dir and output_dir.strip() else src_dir / "jpg_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = Path(tempfile.mkdtemp(prefix="jpg_thumbs_"))

    files = [f for f in sorted(src_dir.glob("**/*" if recursive else "*"))
             if f.is_file() and out_dir not in f.parents]
    lines, previews = [], []
    ok = failed = 0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(src_dir)
        dst = out_dir / rel.parent / (src.stem + ".jpg")
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            convert_file(src, dst, int(quality))
            ok += 1
            if len(previews) < 24:
                with Image.open(dst) as im:
                    im.thumbnail((256, 256))
                    thumb = thumb_dir / f"{len(previews)}.jpg"
                    im.save(thumb, "JPEG")
                previews.append((str(thumb), str(rel)))
        except Exception as exc:
            failed += 1
            lines.append(f"Skipped {rel}: {exc}")
        if i % 10 == 0 or i == len(files):
            status = f"Processed {i} of {len(files)}: {ok} converted, {failed} skipped"
            yield status + "\n" + "\n".join(lines[-200:]), previews
    summary = f"Done: {ok} converted, {failed} skipped. Output directory: {out_dir}"
    (out_dir / "conversion_log.txt").write_text(summary + "\n" + "\n".join(lines) + "\n")
    yield summary + "\n" + "\n".join(lines[-200:]), previews


with gr.Blocks(title="Directory to JPG Converter") as demo:
    gr.Markdown("# Directory to JPG Converter")
    with gr.Row():
        input_dir = gr.Textbox(label="Input directory", placeholder="/path/to/images", scale=5)
        browse_btn = gr.Button("Browse", scale=1)
    output_dir = gr.Textbox(label="Output directory (blank = jpg_output inside input directory)")
    with gr.Row():
        quality = gr.Slider(1, 100, value=95, step=1, label="JPEG quality")
        recursive = gr.Checkbox(label="Include subdirectories", value=False)
    run_btn = gr.Button("Convert", variant="primary")
    log = gr.Textbox(label="Log", lines=12)
    gallery = gr.Gallery(label="Converted previews (first 24)", columns=6)

    browse_btn.click(browse_directory, inputs=input_dir, outputs=input_dir)
    run_btn.click(convert_directory, inputs=[input_dir, output_dir, quality, recursive], outputs=[log, gallery])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
