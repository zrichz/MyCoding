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


def randomize_lora(lora_name, amplitude_pct, seed, skip_alpha, alpha_only):
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
        if (
            not t.is_floating_point()
            or t.numel() == 0
            or (alpha_only and not is_alpha)
            or (not alpha_only and skip_alpha and is_alpha)
        ):
            out[key] = t.clone()
            continue
        noise = (torch.rand(t.shape, generator=gen, dtype=torch.float32) * 2.0 - 1.0) * amp
        new = (t.to(torch.float32) * (1.0 + noise)).to(t.dtype)
        out[key] = new.contiguous()
        changed += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(lora_name)[0]
    mode = "alphaonly" if alpha_only else "rand"
    out_name = f"{stem}_{mode}{amplitude_pct:.2f}pct_seed{seed}.safetensors"
    dst = os.path.join(OUTPUT_DIR, out_name)

    metadata["randomizer_amplitude_pct"] = f"{amplitude_pct:.2f}"
    metadata["randomizer_seed"] = str(seed)
    metadata["randomizer_alpha_only"] = str(bool(alpha_only))
    save_file(out, dst, metadata=metadata)

    # Verify structure matches the original
    check = load_file(dst)
    ok = (
        set(check) == set(tensors)
        and all(check[k].shape == tensors[k].shape and check[k].dtype == tensors[k].dtype for k in tensors)
        and all(torch.isfinite(v).all() for v in check.values() if v.is_floating_point())
    )

    # Debug: compare the weights as read back from disk against the originals
    rel_means, n_identical, sq_diff, sq_orig = [], 0, 0.0, 0.0
    print(f"[debug] {lora_name} -> {out_name} (amp +/-{amplitude_pct:.2f}%, seed {seed})")
    for k in tensors:
        a, b = tensors[k], check[k]
        if not a.is_floating_point() or a.numel() == 0:
            continue
        a32, b32 = a.to(torch.float32), b.to(torch.float32)
        d = b32 - a32
        if torch.equal(a, b):
            n_identical += 1
        sq_diff += float((d ** 2).sum())
        sq_orig += float((a32 ** 2).sum())
        rel = float(d.abs().mean() / (a32.abs().mean() + 1e-12))
        rel_means.append(rel)
        print(f"[debug] {k}: dtype={a.dtype} mean|w|={float(a32.abs().mean()):.6g} "
              f"mean|delta|={float(d.abs().mean()):.6g} max|delta|={float(d.abs().max()):.6g} "
              f"rel_mean={rel * 100:.2f}% identical={torch.equal(a, b)}")
    overall_rel = (sq_diff ** 0.5) / (sq_orig ** 0.5 + 1e-12)
    avg_rel = sum(rel_means) / len(rel_means) if rel_means else 0.0
    print(f"[debug] overall relative L2 change: {overall_rel * 100:.2f}%, "
          f"mean per-tensor relative change: {avg_rel * 100:.2f}%, "
          f"tensors identical to original: {n_identical}")

    msg = (
        f"Debug: overall relative L2 change {overall_rel * 100:.2f} percent, "
        f"mean per-tensor relative change {avg_rel * 100:.2f} percent, "
        f"float tensors identical to original: {n_identical}\n"
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
    amp = gr.Slider(0.0, 200.0, value=10.0, step=1.00, label="Amplitude (percent)")
    seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
    skip_alpha = gr.Checkbox(True, label="Leave alpha scalars unchanged")
    alpha_only = gr.Checkbox(False, label="Perturb only the alpha scalars (leaves all A and B weights unchanged)")
    gr.Markdown(
        "Note: keep the amplitude small when perturbing only the alphas (for example 1 to 10 percent). "
        "Alpha scales the whole layer, so the changes do not average out and have a large effect."
    )
    run = gr.Button("Generate modified LoRA", variant="primary")
    status = gr.Textbox(label="Status", lines=6)
    file_out = gr.File(label="Modified LoRA")

    refresh.click(lambda: gr.update(choices=list_loras()), None, lora)
    run.click(randomize_lora, [lora, amp, seed, skip_alpha, alpha_only], [status, file_out])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
