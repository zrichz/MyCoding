#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Differentiable Circle Painter

Approximates a target RGB image using a number of soft, differentiable
circles. Each circle is fully opaque; its position, radius, and RGB
colour are all directly learned parameters, optimized to minimize
reconstruction error against the target image, rendered with a fully
differentiable renderer. Partway through training, extra small circles
are introduced to add fine detail, and the optimizer's learning
rate/momentum periodically restarts (keeping the current circle
layout) to help escape poor local minima.
"""

import copy
import math
import time

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Default largest permitted circle diameter, as a fraction of the smaller image dimension
DEFAULT_MAX_DIAMETER_FRACTION = 1.0 / 3.0

# Fixed opacity applied to every circle
FIXED_ALPHA = 1.0

# At this training step, extra small circles are introduced for fine detail
ADD_CIRCLES_AT_STEP = 200
ADD_CIRCLES_COUNT = 100
ADD_CIRCLES_MAX_RADIUS = 0.15  # small circles: raw radius sampled in [0, this]

# Number of times the optimizer (lr/momentum) restarts during training
NUM_RESTARTS = 2


def _to_logit(x):
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def _regular_grid_logits(n):
    """Regularly-spaced positions, mid-range radius, random RGB colour."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    xy_raw = torch.empty(n, 2)
    for i in range(n):
        row, col = divmod(i, ncols)
        xy_raw[i, 0] = (col + 0.5) / ncols
        xy_raw[i, 1] = (row + 0.5) / nrows

    radius_raw = torch.full((n,), 0.5)
    color_raw = torch.rand(n, 3)
    return _to_logit(xy_raw), _to_logit(radius_raw), _to_logit(color_raw)


def _random_small_circle_logits(n, max_radius=ADD_CIRCLES_MAX_RADIUS):
    """Random positions and colours, with small radii for fine detail."""
    xy_raw = torch.rand(n, 2)
    radius_raw = torch.rand(n) * max_radius
    color_raw = torch.rand(n, 3)
    return _to_logit(xy_raw), _to_logit(radius_raw), _to_logit(color_raw)


class CircleParams(nn.Module):
    """A group of circles whose position, radius, and colour are directly
    learnable parameters (stored as logits so sigmoid keeps them in [0,1])."""

    def __init__(self, xy_logit, radius_logit, color_logit):
        super().__init__()
        self.xy_logit = nn.Parameter(xy_logit)
        self.radius_logit = nn.Parameter(radius_logit)
        self.color_logit = nn.Parameter(color_logit)

    @property
    def num_circles(self):
        return self.xy_logit.shape[0]

    def forward(self):
        xy = torch.sigmoid(self.xy_logit)
        radius = torch.sigmoid(self.radius_logit)
        color = torch.sigmoid(self.color_logit)
        return xy, radius, color


# ============================================================
# 1. Differentiable circle renderer (vectorized for N circles)
# ============================================================

class CircleRenderer(nn.Module):
    def __init__(self, H, W, sharpness=50, max_diameter_fraction=DEFAULT_MAX_DIAMETER_FRACTION):
        super().__init__()
        self.H = H
        self.W = W
        self.sharpness = sharpness

        # Normalize coordinates by the smaller dimension so circles stay
        # circular (not stretched into ellipses) on non-square images.
        min_dim = min(H, W)
        self.aspect_w = W / min_dim
        self.aspect_h = H / min_dim
        self.max_radius = max_diameter_fraction / 2.0

        ys = torch.linspace(0, self.aspect_h, H)
        xs = torch.linspace(0, self.aspect_w, W)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

    def forward(self, xy, radius_raw, color, alpha=FIXED_ALPHA, hard=False, shape="circle"):
        """
        xy: (B, N, 2) raw [0,1] values -> x, y (learned)
        radius_raw: (B, N) raw [0,1] values (learned)
        color: (B, N, 3) RGB in [0,1], learned per circle
        hard: if True, render a crisp fully-filled shape (hard edge)
            instead of the soft, differentiable mask. Used only for
            display - the soft mask (hard=False) is what's used for the
            loss/backward pass.
        shape: "circle" (Euclidean distance) or "square" (Chebyshev
            distance, i.e. an axis-aligned square).
        returns: (B, 3, H, W)

        Shapes are overlaid (each contributes alpha * colour on top of the
        others), with the result simply clamped at full white rather than
        occluding one another.
        """
        cx = (xy[:, :, 0] * self.aspect_w).unsqueeze(-1).unsqueeze(-1)
        cy = (xy[:, :, 1] * self.aspect_h).unsqueeze(-1).unsqueeze(-1)
        r = (radius_raw * self.max_radius).unsqueeze(-1).unsqueeze(-1)

        dx = self.xx.unsqueeze(0).unsqueeze(0) - cx
        dy = self.yy.unsqueeze(0).unsqueeze(0) - cy

        if shape == "square":
            d = torch.maximum(dx.abs(), dy.abs())
        else:
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)

        if hard:
            m = (d <= r).float()  # (B, N, H, W)
        else:
            m = torch.sigmoid(self.sharpness * (r - d))  # (B, N, H, W)

        coverage = m * alpha

        color_reshaped = color.unsqueeze(-1).unsqueeze(-1)  # (B, N, 3, 1, 1)
        contrib = coverage.unsqueeze(2) * color_reshaped  # (B, N, 3, H, W)

        img = contrib.sum(dim=1).clamp(0, 1)
        return img


# ============================================================
# 2. Model holding all learnable circle parameters
# ============================================================

class CirclePainter(nn.Module):
    def __init__(self, H, W, num_circles=10, sharpness=50, max_diameter_fraction=DEFAULT_MAX_DIAMETER_FRACTION):
        super().__init__()
        self.H = H
        self.W = W

        self.renderer = CircleRenderer(H, W, sharpness, max_diameter_fraction)
        self.circles = CircleParams(*_regular_grid_logits(num_circles))
        self.extra_circles = None
        self.fixed_alpha = FIXED_ALPHA

    @property
    def total_circles(self):
        n = self.circles.num_circles
        if self.extra_circles is not None:
            n += self.extra_circles.num_circles
        return n

    def add_circles(self, n, max_radius=ADD_CIRCLES_MAX_RADIUS):
        """Introduce n additional small circles for fine detail."""
        device = self.circles.xy_logit.device
        self.extra_circles = CircleParams(*_random_small_circle_logits(n, max_radius)).to(device)

    def forward(self, hard=False, shape="circle"):
        xy, radius, color = self.circles()
        if self.extra_circles is not None:
            xy2, radius2, color2 = self.extra_circles()
            xy = torch.cat([xy, xy2], dim=0)
            radius = torch.cat([radius, radius2], dim=0)
            color = torch.cat([color, color2], dim=0)

        img = self.renderer(
            xy.unsqueeze(0), radius.unsqueeze(0), color.unsqueeze(0), self.fixed_alpha, hard=hard, shape=shape
        )
        return img


# ============================================================
# 3. Helpers
# ============================================================

def load_target_tensor(pil_image, max_side, device):
    """Convert a PIL image to a normalized (1, 3, H, W) RGB tensor,
    resizing so the longest side equals max_side while keeping aspect ratio."""
    img = pil_image.convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_uint8(tensor):
    """Convert a (1, 3, H, W) tensor in [0,1] to a uint8 HxWx3 numpy array."""
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0).astype(np.uint8)


# ============================================================
# 4. Training driver (generator, streams progress to the UI)
# ============================================================

def train_circle_painter(image, max_side, num_circles, sharpness, max_diameter_pct, steps, lr, log_every, shape):
    if image is None:
        yield None, "Please upload an image first."
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_side = int(max_side)
    num_circles = int(num_circles)
    steps = int(steps)
    log_every = max(1, int(log_every))
    max_diameter_fraction = max_diameter_pct / 100.0

    target = load_target_tensor(image, max_side, device)
    H, W = target.shape[-2:]

    model = CirclePainter(H, W, num_circles, sharpness, max_diameter_fraction).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Evenly-spaced steps at which the optimizer (lr/momentum) restarts.
    restart_points = sorted({
        round(steps * i / (NUM_RESTARTS + 1))
        for i in range(1, NUM_RESTARTS + 1)
    } - {0, steps})

    circles_added = False
    best_loss = float("inf")
    best_state = None

    target_uint8 = tensor_to_uint8(target)
    start_time = time.time()
    log_lines = [
        f"Device: {device}",
        f"Image size: {W}x{H}, circles: {num_circles}, shape: {shape}, "
        f"sharpness: {sharpness}, max diameter: {max_diameter_pct:.0f}% of smaller side, "
        f"steps: {steps}, lr: {lr}",
        "",
    ]

    for step in range(1, steps + 1):
        opt.zero_grad()
        recon = model(shape=shape)
        loss = F.mse_loss(recon, target)
        loss.backward()
        opt.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_state = copy.deepcopy(model.state_dict())

        if not circles_added and step >= ADD_CIRCLES_AT_STEP and steps > ADD_CIRCLES_AT_STEP:
            model.add_circles(ADD_CIRCLES_COUNT)
            opt.add_param_group({"params": model.extra_circles.parameters(), "lr": lr})
            circles_added = True
            log_lines.append(
                f"Step {step}: introduced {ADD_CIRCLES_COUNT} additional small circles "
                f"(total now {model.total_circles})"
            )

        if step in restart_points:
            # Reset the optimizer (lr/momentum) only - keep the current
            # circle layout, don't throw away training progress.
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            log_lines.append(f"Step {step}: learning rate/momentum restart (best loss so far {best_loss:.6f})")

        if step % log_every == 0 or step == steps:
            elapsed = time.time() - start_time
            with torch.no_grad():
                recon_hard = model(hard=True, shape=shape)
            recon_uint8 = tensor_to_uint8(recon_hard)

            log_lines.append(f"Step {step}/{steps} | Loss {current_loss:.6f} | {elapsed:.1f}s")
            status = "\n".join(log_lines[-12:])

            stacked = np.concatenate([target_uint8, recon_uint8], axis=0)

            yield stacked, status

    if best_state is not None and best_loss < current_loss:
        model.load_state_dict(best_state, strict=False)
        log_lines.append(f"Restored best checkpoint (loss {best_loss:.6f}) over final state (loss {current_loss:.6f})")

    with torch.no_grad():
        recon_hard = model(hard=True, shape=shape)
    recon_uint8 = tensor_to_uint8(recon_hard)
    stacked = np.concatenate([target_uint8, recon_uint8], axis=0)

    final_status = "\n".join(log_lines) + "\n\nTraining complete."
    yield stacked, final_status


# ============================================================
# 5. Gradio UI
# ============================================================

with gr.Blocks() as demo:
    gr.Markdown(
        "Differentiable Circle Painter\n"
        
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Target Image", type="pil")

            image_size = gr.Radio([128, 256, 512], value=128, label="Max Side Length (aspect ratio preserved)")
            num_circles = gr.Slider(1, 1000, value=10, step=1, label="Number of Circles")
            shape = gr.Radio(["circle", "square"], value="circle", label="Shape")
            sharpness = gr.Slider(5, 200, value=50, step=5, label="Edge Sharpness")
            max_diameter_pct = gr.Slider(5, 50, value=33, step=1, label="Max Circle Diameter (% of smaller side)")
            steps = gr.Slider(100, 10000, value=400, step=100, label="Training Steps")
            lr = gr.Slider(1e-4, 5e-2, value=1e-3, step=1e-4, label="Learning Rate")
            log_every = gr.Slider(1, 500, value=10, step=1, label="Update Every N Steps")

            train_btn = gr.Button("Train", variant="primary")

        with gr.Column(scale=2):
            comparison_output = gr.Image(label="Target (top) vs Reconstruction (bottom)")
            status_output = gr.Textbox(label="Training Log", lines=14)

    train_btn.click(
        fn=train_circle_painter,
        inputs=[image_input, image_size, num_circles, sharpness, max_diameter_pct, steps, lr, log_every, shape],
        outputs=[comparison_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
