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
    # The model stores bounded values as unconstrained logits. Applying a
    # sigmoid later maps those logits back into (0, 1), while optimization
    # can still move them freely in either direction. Clamping avoids an
    # infinite logit when an initialization happens to contain 0 or 1.
    x = x.clamp(1e-4, 1 - 1e-4)
    return torch.log(x / (1 - x))


def _regular_grid_logits(n):
    """Regularly-spaced positions, mid-range radius, random RGB colour."""
    # Start the large circles on a grid so the initial image has coverage
    # across the whole canvas instead of relying on random placement.
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    # These are still expressed in normalized coordinates. The renderer
    # later converts them to coordinates in its aspect-correct pixel space.
    xy_raw = torch.empty(n, 2)
    for i in range(n):
        row, col = divmod(i, ncols)
        xy_raw[i, 0] = (col + 0.5) / ncols
        xy_raw[i, 1] = (row + 0.5) / nrows

    # A radius of 0.5 is a neutral starting point; colours are randomized
    # so every circle can begin learning a different RGB contribution.
    radius_raw = torch.full((n,), 0.5)
    color_raw = torch.rand(n, 3)
    return _to_logit(xy_raw), _to_logit(radius_raw), _to_logit(color_raw)


def _random_small_circle_logits(n, max_radius=ADD_CIRCLES_MAX_RADIUS):
    """Random positions and colours, with small radii for fine detail."""
    # These circles are added after the broad image structure has been
    # learned. Random locations and small sizes give them room to capture
    # local detail without replacing the existing large-circle layout.
    xy_raw = torch.rand(n, 2)
    radius_raw = torch.rand(n) * max_radius
    color_raw = torch.rand(n, 3)
    return _to_logit(xy_raw), _to_logit(radius_raw), _to_logit(color_raw)


class CircleParams(nn.Module):
    """A group of circles whose position, radius, and colour are directly
    learnable parameters (stored as logits so sigmoid keeps them in [0,1])."""

    def __init__(self, xy_logit, radius_logit, color_logit):
        super().__init__()
        # nn.Parameter makes these tensors visible to the optimizer. The
        # logits themselves are intentionally unconstrained; forward() is
        # responsible for converting them into valid render parameters.
        self.xy_logit = nn.Parameter(xy_logit)
        self.radius_logit = nn.Parameter(radius_logit)
        self.color_logit = nn.Parameter(color_logit)

    @property
    def num_circles(self):
        return self.xy_logit.shape[0]

    def forward(self):
        # Keep every learned value in the renderer's expected range. The
        # sigmoid is differentiable, so gradients can update the logits.
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

        # The coordinate grid is fixed for one image size, so register it as
        # a buffer: it moves with the model between CPU and GPU, but is not a
        # learnable parameter and is not included in optimizer updates.
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
        # Expand each circle's center and radius to (B, N, 1, 1). This lets
        # PyTorch broadcast them against the (H, W) coordinate grid below.
        cx = (xy[:, :, 0] * self.aspect_w).unsqueeze(-1).unsqueeze(-1)
        cy = (xy[:, :, 1] * self.aspect_h).unsqueeze(-1).unsqueeze(-1)
        r = (radius_raw * self.max_radius).unsqueeze(-1).unsqueeze(-1)

        # Distance from every pixel to every circle center: (B, N, H, W).
        # Because both axes use the same normalized unit length, a circle
        # remains circular even when the source image is rectangular.
        dx = self.xx.unsqueeze(0).unsqueeze(0) - cx
        dy = self.yy.unsqueeze(0).unsqueeze(0) - cy

        if shape == "square":
            # Chebyshev distance produces an axis-aligned square: a pixel is
            # inside when its largest axis distance is no greater than r.
            d = torch.maximum(dx.abs(), dy.abs())
        else:
            # Euclidean distance produces a circle. The small epsilon keeps
            # the square-root operation numerically well behaved at a center.
            d = torch.sqrt(dx * dx + dy * dy + 1e-8)

        if hard:
            # This crisp comparison is useful for the final visual result,
            # but its binary edge does not provide smooth training gradients.
            m = (d <= r).float()  # (B, N, H, W)
        else:
            # The sigmoid is a soft approximation of d <= r. Increasing
            # sharpness makes the edge more like a hard boundary while still
            # retaining a differentiable transition for backpropagation.
            m = torch.sigmoid(self.sharpness * (r - d))  # (B, N, H, W)

        # Alpha is currently one, but keeping this multiplication explicit
        # leaves the renderer's coverage model clear and configurable.
        coverage = m * alpha

        # Add two singleton spatial dimensions to each RGB colour so it can
        # broadcast over every pixel: (B, N, 3, 1, 1).
        color_reshaped = color.unsqueeze(-1).unsqueeze(-1)  # (B, N, 3, 1, 1)
        contrib = coverage.unsqueeze(2) * color_reshaped  # (B, N, 3, H, W)

        # Circles are additive rather than painter-style occluding layers.
        # Summing over N gives an image, and clamping prevents accumulated
        # contributions from exceeding the valid RGB range.
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
        # Extra circles are created later, after the optimizer has already
        # learned the coarse composition from the initial set.
        self.extra_circles = None
        self.fixed_alpha = FIXED_ALPHA

    @property
    def total_circles(self):
        # Keep the count derived from the parameter groups so the UI log
        # remains correct before and after detail circles are introduced.
        n = self.circles.num_circles
        if self.extra_circles is not None:
            n += self.extra_circles.num_circles
        return n

    def add_circles(self, n, max_radius=ADD_CIRCLES_MAX_RADIUS):
        """Introduce n additional small circles for fine detail."""
        # New tensors are initialized on CPU by the helper, then moved to the
        # same device as the existing model parameters before training uses
        # them. The caller separately adds them to the optimizer.
        device = self.circles.xy_logit.device
        self.extra_circles = CircleParams(*_random_small_circle_logits(n, max_radius)).to(device)

    def forward(self, hard=False, shape="circle"):
        # Each CircleParams group returns one value per circle. Concatenate
        # the optional detail group so the renderer handles all circles in a
        # single vectorized operation.
        xy, radius, color = self.circles()
        if self.extra_circles is not None:
            xy2, radius2, color2 = self.extra_circles()
            xy = torch.cat([xy, xy2], dim=0)
            radius = torch.cat([radius, radius2], dim=0)
            color = torch.cat([color, color2], dim=0)

        # The renderer expects a batch dimension, while CircleParams returns
        # (N, ...), so unsqueeze(0) changes each input to (1, N, ...).
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
    # Convert first so grayscale, palette, and RGBA inputs all produce the
    # same three-channel target expected by the renderer.
    img = pil_image.convert("RGB")
    w, h = img.size
    # Scale by the longest side, preserving aspect ratio and guaranteeing a
    # positive output dimension even for very small source images.
    scale = max_side / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    # PIL/NumPy use HWC layout; PyTorch convolution-style image code uses
    # CHW, so permute the axes and add the batch dimension expected by model.
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_uint8(tensor):
    """Convert a (1, 3, H, W) tensor in [0,1] to a uint8 HxWx3 numpy array."""
    # Detach removes autograd history, move to CPU for NumPy compatibility,
    # remove the single batch dimension, and transpose back to image layout.
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0).astype(np.uint8)


# ============================================================
# 4. Training driver (generator, streams progress to the UI)
# ============================================================

def train_circle_painter(image, max_side, num_circles, sharpness, max_diameter_pct, steps, lr, log_every, shape):
    if image is None:
        # A generator must yield the UI-compatible pair before stopping so
        # the user receives a useful validation message.
        yield None, "Please upload an image first."
        return

    # CUDA is used when available; all target, model, and parameter tensors
    # must live on this same device for PyTorch operations to work.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gradio values arrive as UI values, so normalize their types and enforce
    # a valid logging interval before constructing the training loop.
    max_side = int(max_side)
    num_circles = int(num_circles)
    steps = int(steps)
    log_every = max(1, int(log_every))
    max_diameter_fraction = max_diameter_pct / 100.0

    # This target determines the exact H and W used by both the renderer and
    # the reconstruction, ensuring the MSE tensors have matching shapes.
    target = load_target_tensor(image, max_side, device)
    H, W = target.shape[-2:]

    model = CirclePainter(H, W, num_circles, sharpness, max_diameter_fraction).to(device)
    # Adam updates the circle logits using gradients produced by the soft
    # renderer and the MSE loss below.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Evenly-spaced steps at which the optimizer (lr/momentum) restarts.
    restart_points = sorted({
        round(steps * i / (NUM_RESTARTS + 1))
        for i in range(1, NUM_RESTARTS + 1)
    } - {0, steps})

    # Track the best parameter snapshot independently of the final iteration;
    # Adam can occasionally move away from an earlier lower-loss state.
    circles_added = False
    best_loss = float("inf")
    best_state = None

    # This copy is only for the UI comparison. Training continues to use the
    # normalized device tensor above.
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
        # Standard optimization order: clear old gradients, render the
        # current parameters, differentiate the loss, then update parameters.
        opt.zero_grad()
        recon = model(shape=shape)

        # The reconstruction and target both have shape (1, 3, H, W) and
        # contain normalized RGB values in [0, 1]. 
        # 
        # Loss = MSE over every colour channel and pixel:
        #
        #     loss = mean((recon - target) ** 2)
        #
        # eg each channel's error is squared and errors averaged across whole image
        # This uses the idea of "soft" circles in the differentiable renderer, rather than the "hard" circles used for display. 
        # The soft circles are produced by the differentiable renderer, so autograd can send gradients through the circle masks to their positions, radii,
        # and colours.
        loss = F.mse_loss(recon, target)
        loss.backward()
        opt.step()

        # Convert the tensor to a straight number (stripping its computation graph)
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            # use "deepcopy" - needed because otherwise model.state_dict() contains tensors that would continue changing during training.
            best_state = copy.deepcopy(model.state_dict())

        if not circles_added and step >= ADD_CIRCLES_AT_STEP and steps > ADD_CIRCLES_AT_STEP:
            # Add detail capacity only after the coarse stage. Since these
            # parameters did not exist when Adam was created, register them
            # explicitly in a new optimizer parameter group.
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
            # Display rendering is intentionally separate from training, as it simply uses "hard" circles.
            with torch.no_grad():
                recon_hard = model(hard=True, shape=shape)
            recon_uint8 = tensor_to_uint8(recon_hard)

            log_lines.append(f"Step {step}/{steps} | Loss {current_loss:.6f} | {elapsed:.1f}s")
            status = "\n".join(log_lines[-12:])

            # Stack vertically so the output image shows target on top and reconstruction underneath.
            stacked = np.concatenate([target_uint8, recon_uint8], axis=0)

            yield stacked, status

    # Restore the best observed state rather than returning a worse final
    # state if the last few Adam updates increased the loss.
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
# Gradio UI
# ============================================================

with gr.Blocks() as demo:
    # The Blocks context builds the page once; clicking Train later invokes
    # the generator below and streams each yielded image/status pair into the
    # two output components.
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

    # Keep the UI order explicit so each control is passed to the matching
    # positional argument of train_circle_painter.
    train_btn.click(
        fn=train_circle_painter,
        inputs=[image_input, image_size, num_circles, sharpness, max_diameter_pct, steps, lr, log_every, shape],
        outputs=[comparison_output, status_output],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
