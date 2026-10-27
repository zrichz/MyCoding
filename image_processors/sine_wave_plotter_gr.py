#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Sine Wave Image Recreator - Pen Plotter Emulation

Same wave layout as sine_wave_image_gr.py, but each wave is first passed
through a pen plotter physics pre-processing step before it is rendered:

  Mass / momentum   - the pen is a damped spring-mass following the commanded
                      path, so fast oscillations are attenuated, lag and
                      overshoot at sharp turns.
  Backlash          - mechanical play in the Y axis: after a direction reversal
                      the pen does not move until the slack is taken up.
  Jitter            - smoothed random positional noise (stepper / belt noise).
  Ink flow          - line width and opacity depend on pen speed (slow = pooled
                      and wide, fast = starved and thin) and on stroke direction
                      relative to a stub nib angle.
  Ink blobs         - extra ink deposited where the nib touches down and lifts.
  Bleed             - small blur emulating capillary spread into paper fibres.
"""

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from scipy.signal import lfilter, lfilter_zi

BASE_FREQ = 0.15  # cycles per pixel, the same everywhere

# Subtractive ink colours as RGB (0..1)
INKS = {
    "C": np.array([0.0, 1.0, 1.0], np.float32),
    "M": np.array([1.0, 0.0, 1.0], np.float32),
    "Y": np.array([1.0, 1.0, 0.0], np.float32),
    "K": np.array([0.0, 0.0, 0.0], np.float32),
}


class PlotterParams:
    def __init__(self, mass, damping, backlash, jitter, stub, nib_angle,
                 flow, blob, bleed, seed):
        self.mass = mass
        self.damping = damping
        self.backlash = backlash
        self.jitter = jitter
        self.stub = stub
        self.nib_angle = np.radians(nib_angle)
        self.flow = flow
        self.blob = blob
        self.bleed = bleed
        self.seed = int(seed)


def smooth_noise(rng, n, scale):
    """Gaussian noise low-pass filtered to a correlation length of ~scale samples."""
    raw = rng.standard_normal(n + 6 * int(scale) + 8)
    k = np.exp(-0.5 * (np.arange(-3 * int(scale) - 3, 3 * int(scale) + 4) / scale) ** 2)
    k /= np.sqrt(np.sum(k ** 2))
    return np.convolve(raw, k, mode="valid")[:n]


def apply_backlash(y, play):
    """Backlash operator: output only follows after the slack is taken up."""
    if play <= 0:
        return y
    out = np.empty_like(y)
    half = play / 2.0
    cur = y[0]
    for i, v in enumerate(y):
        if v > cur + half:
            cur = v - half
        elif v < cur - half:
            cur = v + half
        out[i] = cur
    return out


def apply_mass(y, mass, damping):
    """Second-order (spring-damper) response of a pen with inertia."""
    if mass <= 0:
        return y
    wn = 1.0 / (1.0 + 6.0 * mass)  # natural frequency in rad per sample
    z = damping
    b = [0.0, wn * wn]
    a = [1.0, -(2.0 - 2.0 * z * wn - wn * wn), 1.0 - 2.0 * z * wn]
    zi = lfilter_zi(b, a) * y[0]
    return lfilter(b, a, y, zi=zi)[0]


def stroke_dynamics(xs, ys, rng, p):
    """Turn the commanded stroke (xs, ys) into the path the pen actually draws."""
    n = len(xs)
    y = apply_backlash(ys, p.backlash)
    y = apply_mass(y, p.mass, p.damping)
    x = xs.astype(np.float64)
    if p.jitter > 0:
        y = y + p.jitter * smooth_noise(rng, n, 3.0)
        x = x + 0.5 * p.jitter * smooth_noise(rng, n, 3.0)
    return x, y


def draw_stroke(canvas, x, y, opacity, thickness, p, rng, direction_sign):
    """Render one stroke with speed and nib-angle dependent width and opacity."""
    n = len(x)
    dx = np.diff(x) * direction_sign
    dy = np.diff(y)
    speed = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)
    stub = 1.0 - p.stub * (1.0 - np.abs(np.sin(angle - p.nib_angle)))
    # Slow motion pools ink, fast motion starves the nib
    starve = np.exp(-p.flow * 0.25 * (speed - 1.0))
    width = thickness * stub * np.clip(starve, 0.6, 1.5)
    seg_op = opacity[:-1] * np.clip(starve, 0.5, 1.0)
    # Slow irregular ink supply variation
    seg_op = seg_op * (1.0 - 0.15 * p.flow * np.abs(smooth_noise(rng, n - 1, 12.0)))

    pts = np.round(np.stack([x, y], axis=1) * 16).astype(np.int32)
    thick_i = np.maximum(1, np.round(width)).astype(int)
    level = np.clip(seg_op * np.clip(width / thick_i, 0.3, 1.0), 0, 1)
    val = (level * 255).astype(int)
    for i in range(n - 1):
        if val[i] <= 0:
            continue
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), int(val[i]),
                 int(thick_i[i]), cv2.LINE_AA, shift=4)

    if p.blob > 0 and opacity.max() > 0.2:
        # Nib touch-down pool and lift-off dab
        for idx, scale in ((0, 1.0), (n - 1, 0.6)):
            r = max(1, int(round(thickness * (0.5 + p.blob) * scale)))
            c = (int(round(x[idx])), int(round(y[idx])))
            cv2.circle(canvas, c, r, int(255 * min(1.0, opacity[idx] * 0.8)), -1, cv2.LINE_AA)


def draw_layer(density, spacing, amplitude, mult, thickness, angle, p,
               row_offset=0, fade=False):
    """Draw one wave layer. density is 0..1 (1 = most ink). Returns coverage 0..1."""
    h, w = density.shape
    size = int(np.ceil(np.hypot(h, w))) + 2
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    M[0, 2] += size / 2.0 - w / 2.0
    M[1, 2] += size / 2.0 - h / 2.0
    frame = cv2.warpAffine(density, M, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    k = max(3, spacing | 1)
    smooth = cv2.blur(frame, (k, k))

    rng = np.random.default_rng(p.seed + int(angle * 10) + 1000)
    canvas = np.zeros((size, size), dtype=np.uint8)
    xs = np.arange(size, dtype=np.float64)
    start = spacing // 2 + int(row_offset) % spacing
    for row, yc in enumerate(range(start, size, spacing)):
        d = smooth[yc, :].astype(np.float64)
        # Plotter draws alternate rows in opposite directions
        reverse = row % 2 == 1
        if reverse:
            d = d[::-1]
        # Constant frequency; amplitude grows with darkness
        phase = 2.0 * np.pi * BASE_FREQ * mult * np.arange(size)
        ys = yc + amplitude * np.clip(d, 0.0, 1.0) * np.sin(phase)
        px = xs[::-1] if reverse else xs
        opacity = np.clip(d / 0.12, 0.0, 1.0) if fade else np.ones(size)
        x, y = stroke_dynamics(px, ys, rng, p)
        draw_stroke(canvas, x, y, opacity, thickness, p, rng, -1.0 if reverse else 1.0)

    cov = canvas.astype(np.float32) / 255.0
    if p.bleed > 0:
        cov = cv2.GaussianBlur(cov, (0, 0), p.bleed)
        cov = np.clip(cov * (1.0 + 0.6 * p.bleed), 0, 1)

    inv = cv2.invertAffineTransform(M)
    return cv2.warpAffine(cov, inv, (w, h), flags=cv2.INTER_LINEAR)


def rgb_to_cmyk(rgb):
    k = 1.0 - rgb.max(axis=2)
    denom = np.maximum(1.0 - k, 1e-6)
    c = (1.0 - rgb[..., 0] - k) / denom
    m = (1.0 - rgb[..., 1] - k) / denom
    y = (1.0 - rgb[..., 2] - k) / denom
    return {"C": c, "M": m, "Y": y, "K": k}


def render_waves(image, mode, spacing, amplitude, mult, thickness, invert,
                 ang_c, ang_m, ang_y, ang_k,
                 mass, damping, backlash, jitter, stub, nib_angle, flow, blob,
                 bleed, seed):
    if image is None:
        return None

    spacing = int(spacing)
    thickness = int(thickness)
    p = PlotterParams(mass, damping, backlash, jitter, stub, nib_angle,
                      flow, blob, bleed, seed)
    rgb = np.asarray(Image.fromarray(image).convert("RGB"), dtype=np.float32) / 255.0
    if invert:
        rgb = 1.0 - rgb
    h, w = rgb.shape[:2]

    if mode == "CMYK":
        channels = rgb_to_cmyk(rgb)
        angles = {"C": ang_c, "M": ang_m, "Y": ang_y, "K": ang_k}
        out = np.ones((h, w, 3), np.float32)
        for i, name in enumerate("CMYK"):
            d = np.clip(channels[name], 0, 1).astype(np.float32)
            cov = draw_layer(d, spacing, amplitude, mult, thickness,
                             angles[name], p, row_offset=spacing * i // 4, fade=True)
            out *= 1.0 - cov[..., None] * (1.0 - INKS[name])
    else:
        gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
        cov = draw_layer((1.0 - gray).astype(np.float32), spacing, amplitude,
                         mult, thickness, 0.0, p)
        if mode == "Line colour from image":
            k = max(3, spacing | 1)
            col = cv2.blur(rgb, (k, k))
            # Normalise so lines stay visible on white while keeping hue
            col = col * np.minimum(1.0, 0.6 / np.maximum(col.max(axis=2, keepdims=True), 1e-6))
            out = 1.0 - cov[..., None] * (1.0 - col)
        else:
            out = np.repeat(1.0 - cov[..., None], 3, axis=2)

    result = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    return result


def create_demo():
    with gr.Blocks(title="Sine Wave Pen Plotter") as demo:
        gr.Markdown("# Sine Wave Pen Plotter Emulator")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(label="Input image", type="numpy")
            with gr.Column(scale=1):
                mode = gr.Radio(["Black and white", "CMYK", "Line colour from image"],
                                value="Black and white", label="Mode")
                spacing = gr.Slider(2, 128, value=16, step=1, label="Vertical spacing (px)")
                amplitude = gr.Slider(0, 64, value=8, step=0.5,
                                      label="Maximum amplitude (px, used in the darkest areas)")
                mult = gr.Slider(0.1, 4.0, value=1.0, step=0.05,
                                 label="Frequency multiplier (same everywhere)")
                thickness = gr.Slider(1, 6, value=1, step=1, label="Line thickness (px)")
                invert = gr.Checkbox(value=False, label="Invert image")
                with gr.Accordion("Plotter physics", open=True):
                    mass = gr.Slider(0, 1, value=0.15, step=0.01,
                                     label="Pen mass (0 = ideal, higher = more lag and smoothing)")
                    damping = gr.Slider(0.1, 1.5, value=0.6, step=0.05,
                                        label="Damping (low = overshoot and ringing)")
                    backlash = gr.Slider(0, 6, value=1.0, step=0.1, label="Backlash (px)")
                    jitter = gr.Slider(0, 3, value=0.4, step=0.05, label="Jitter (px)")
                    seed = gr.Number(value=1, precision=0, label="Random seed")
                with gr.Accordion("Ink and nib", open=True):
                    stub = gr.Slider(0, 1, value=0.4, step=0.05, label="Stub nib intensity")
                    nib_angle = gr.Slider(0, 180, value=45, step=1, label="Nib angle (degrees)")
                    flow = gr.Slider(0, 2, value=1.0, step=0.05,
                                     label="Ink flow sensitivity to speed")
                    blob = gr.Slider(0, 2, value=0.6, step=0.05, label="Start and end ink blobs")
                    bleed = gr.Slider(0, 2, value=0.4, step=0.05, label="Paper bleed (blur sigma)")
                with gr.Accordion("CMYK wave angles (degrees)", open=False):
                    ang_c = gr.Slider(-90, 90, value=15, step=1, label="Cyan")
                    ang_m = gr.Slider(-90, 90, value=75, step=1, label="Magenta")
                    ang_y = gr.Slider(-90, 90, value=0, step=1, label="Yellow")
                    ang_k = gr.Slider(-90, 90, value=45, step=1, label="Black")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=1):
                out = gr.Image(label="Plotter output", type="numpy", format="png")

        inputs = [inp, mode, spacing, amplitude, mult, thickness, invert,
                  ang_c, ang_m, ang_y, ang_k,
                  mass, damping, backlash, jitter, stub, nib_angle, flow, blob,
                  bleed, seed]
        btn.click(render_waves, inputs, out)
    return demo


if __name__ == "__main__":
    create_demo().launch(inbrowser=True)
