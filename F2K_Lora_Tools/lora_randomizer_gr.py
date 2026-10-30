"""Randomly perturb the weights of a Flux 2 Klein LoRA (.safetensors).

Reads from ./Original_Loras and writes to ./Modified_Loras. Tensor names, shapes,
dtypes and file metadata are preserved, so the result stays a valid LoRA.
Each weight w is replaced by w * (1 + u), where u is uniform in [-amp, +amp].
"""

import os
import time

import gradio as gr
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "Original_Loras")
OUTPUT_DIR = os.path.join(BASE_DIR, "Modified_Loras")


def list_loras():
    if not os.path.isdir(INPUT_DIR):
        return []
    return sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".safetensors"))


def randomize_lora(lora_name, amplitude_pct, seed, skip_alpha):
    if not lora_name:
        return "No LoRA selected.", None

    src = os.path.join(INPUT_DIR, lora_name)
    if not os.path.isfile(src):
        return f"File not found: {src}", None

    amp = float(amplitude_pct) / 100.0
    seed = int(seed)
    if seed < 0:
        seed = int(time.time() * 1000) % (2**31)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    with safe_open(src, framework="pt") as f:
        metadata = f.metadata() or {}

    tensors = load_file(src)
    out = {}
    changed = 0
    for key, t in tensors.items():
        is_alpha = key.endswith(".alpha") or key.endswith("alpha")
        if not t.is_floating_point() or t.numel() == 0 or (skip_alpha and is_alpha):
            out[key] = t.clone()
            continue
        noise = (torch.rand(t.shape, generator=gen, dtype=torch.float32) * 2.0 - 1.0) * amp
        new = (t.to(torch.float32) * (1.0 + noise)).to(t.dtype)
        out[key] = new.contiguous()
        changed += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(lora_name)[0]
    out_name = f"{stem}_rand{amplitude_pct:.2f}pct_seed{seed}.safetensors"
    dst = os.path.join(OUTPUT_DIR, out_name)

    metadata["randomizer_amplitude_pct"] = f"{amplitude_pct:.2f}"
    metadata["randomizer_seed"] = str(seed)
    save_file(out, dst, metadata=metadata)

    # Verify structure matches the original
    check = load_file(dst)
    ok = (
        set(check) == set(tensors)
        and all(check[k].shape == tensors[k].shape and check[k].dtype == tensors[k].dtype for k in tensors)
        and all(torch.isfinite(v).all() for v in check.values() if v.is_floating_point())
    )

    msg = (
        f"Saved: {dst}\n"
        f"Amplitude: +/-{amplitude_pct:.2f} percent\n"
        f"Seed: {seed}\n"
        f"Tensors total: {len(tensors)}, perturbed: {changed}\n"
        f"Structure check (keys, shapes, dtypes, finite values): {'passed' if ok else 'FAILED'}"
    )
    return msg, dst


with gr.Blocks(title="Flux 2 Klein LoRA Randomizer") as demo:
    gr.Markdown("# Flux 2 Klein LoRA Randomizer")
    with gr.Row():
        lora = gr.Dropdown(list_loras(), label="LoRA (from Original_Loras)")
        refresh = gr.Button("Refresh list")
    amp = gr.Slider(0.0, 50.0, value=10.0, step=1.00, label="Amplitude (percent)")
    seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    skip_alpha = gr.Checkbox(True, label="Leave alpha scalars unchanged")
    run = gr.Button("Generate modified LoRA", variant="primary")
    status = gr.Textbox(label="Status", lines=6)
    file_out = gr.File(label="Modified LoRA")

    refresh.click(lambda: gr.update(choices=list_loras()), None, lora)
    run.click(randomize_lora, [lora, amp, seed, skip_alpha], [status, file_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
