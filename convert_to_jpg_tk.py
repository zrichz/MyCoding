"""Minimal tkinter tool to convert all images in a directory to JPG.

Primary use: recovering images from files that carry a classic Mac (MacBinary)
header, such as *.file. These files have a 128-byte header in front of the real
image data, so ordinary viewers and PIL cannot identify them. For each file the
header is detected and stripped, leaving just the data fork (the actual image),
which is then decoded and saved as a JPG.

Also handles ordinary formats (PNG, GIF, BMP, TIFF, WEBP, ...) regardless of file
extension. Animated images use their first frame, and transparent images are
flattened onto a white background.

Choose an input directory, optionally an output directory (default is a
jpg_output folder inside the input directory), the JPEG quality, and whether to
include subdirectories. Folder structure is preserved in the output, and a
conversion_log.txt summarising any skipped files is written there."""
import io
import queue
import struct
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def unwrap_macbinary(data: bytes) -> bytes:
    """Return the data fork if data looks like a MacBinary file, else data."""
    if len(data) >= 128 and data[0] == 0 and data[74] == 0 and data[82] == 0 and 1 <= data[1] <= 63:
        fork_len = struct.unpack(">I", data[83:87])[0]
        if 0 < fork_len <= len(data) - 128:
            return data[128:128 + fork_len]
    return data


def convert_file(src: Path, dst: Path, quality: int) -> None:
    data = unwrap_macbinary(src.read_bytes())
    with Image.open(io.BytesIO(data)) as img:
        img.seek(0)  # first frame of animated images
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = img.convert("RGB")
        rgb.save(dst, "JPEG", quality=quality)


def convert_directory(src_dir: Path, out_dir: Path, quality: int, recursive: bool, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(src_dir.glob("**/*" if recursive else "*"))
             if f.is_file() and out_dir not in f.parents]
    lines = []
    used = set()
    ok = failed = 0
    for i, src in enumerate(files, 1):
        rel = src.relative_to(src_dir)
        dst = out_dir / rel.parent / (src.stem + ".jpg")
        # Names like BH1C.001 / BH1C.002 differ only by extension, so keep the full
        # name when the stem alone would collide with another output file.
        if dst in used or src.suffix[1:].isdigit():
            dst = out_dir / rel.parent / (src.name + ".jpg")
        used.add(dst)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            convert_file(src, dst, quality)
            ok += 1
        except Exception as exc:
            failed += 1
            lines.append(f"Skipped {rel}: {exc}")
            log(lines[-1])
        if i % 10 == 0 or i == len(files):
            log(f"Processed {i} of {len(files)}: {ok} converted, {failed} skipped")
    summary = f"Done: {ok} converted, {failed} skipped. Output directory: {out_dir}"
    (out_dir / "conversion_log.txt").write_text(summary + "\n" + "\n".join(lines) + "\n")
    log(summary)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Directory to JPG Converter")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.quality_var = tk.IntVar(value=95)
        self.recursive_var = tk.BooleanVar(value=False)
        self.messages = queue.Queue()

        frm = ttk.Frame(root, padding=8)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Input directory").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_var, width=60).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(frm, text="Browse", command=lambda: self.browse(self.input_var)).grid(row=0, column=2)

        ttk.Label(frm, text="Output directory").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(frm, text="Browse", command=lambda: self.browse(self.output_var)).grid(row=1, column=2)
        ttk.Label(frm, text="(blank = jpg_output inside input directory)").grid(row=2, column=1, sticky="w")

        opts = ttk.Frame(frm)
        opts.grid(row=3, column=0, columnspan=3, sticky="w", pady=4)
        ttk.Label(opts, text="JPEG quality").pack(side="left")
        ttk.Spinbox(opts, from_=1, to=100, textvariable=self.quality_var, width=5).pack(side="left", padx=4)
        ttk.Checkbutton(opts, text="Include subdirectories", variable=self.recursive_var).pack(side="left", padx=12)
        self.run_btn = ttk.Button(opts, text="Convert", command=self.start)
        self.run_btn.pack(side="left")

        self.log_box = scrolledtext.ScrolledText(frm, height=14, state="disabled")
        self.log_box.grid(row=4, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(4, weight=1)
        self.poll()

    def browse(self, var):
        current = var.get()
        initial = current if current and Path(current).is_dir() else str(Path.home())
        chosen = filedialog.askdirectory(initialdir=initial)
        if chosen:
            var.set(chosen)

    def log(self, text):
        self.messages.put(text)

    def poll(self):
        while True:
            try:
                msg = self.messages.get_nowait()
            except queue.Empty:
                break
            if msg is None:
                self.run_btn.config(state="normal")
                continue
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.root.after(100, self.poll)

    def start(self):
        src = Path(self.input_var.get().strip())
        if not src.is_dir():
            self.log("Please choose a valid directory.")
            return
        out_text = self.output_var.get().strip()
        out = Path(out_text) if out_text else src / "jpg_output"
        try:
            quality = max(1, min(100, int(self.quality_var.get())))
        except (tk.TclError, ValueError):
            quality = 95
        recursive = self.recursive_var.get()
        self.run_btn.config(state="disabled")

        def work():
            try:
                convert_directory(src, out, quality, recursive, self.log)
            except Exception as exc:
                self.log(f"Error: {exc}")
            finally:
                self.messages.put(None)

        threading.Thread(target=work, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
