#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Lenia Simulation with Taichi and Gradio

A continuous cellular automaton that creates life-like patterns.
Every physical parameter (growth rate, target density, growth window,
kernel radius) can be independently varied across the X axis and the
Y axis of the simulation grid, giving rich spatial diversity of behavior.

Requirements: pip install taichi gradio numpy
"""

import gradio as gr
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

# Size of the grid
WIDTH = 1024
HEIGHT = 1024

# The main Lenia field (values between 0 and 1)
field = ti.field(dtype=ti.f32, shape=(HEIGHT, WIDTH))

# A temporary field used for updates
next_field = ti.field(dtype=ti.f32, shape=(HEIGHT, WIDTH))

# A colour image (RGB) for display
image = ti.Vector.field(3, dtype=ti.f32, shape=(HEIGHT, WIDTH))

COLOR_SCHEMES = ["Rainbow", "Fire", "Ocean", "Monochrome", "Plasma"]


@ti.func
def wrap(i, j):
    """Wrap-around so edges connect (toroidal world)"""
    return i % HEIGHT, j % WIDTH


@ti.func
def kernel(r, m: ti.f32, s: ti.f32):
    """Smooth Gaussian kernel for neighbor weighting"""
    return ti.exp(-((r - m) ** 2) / (2.0 * s * s))


def init_field(seed=None):
    """Initialize with a single large Gaussian blob centered on the grid."""
    rng = np.random.default_rng(seed)
    cx, cy = WIDTH / 2.0, HEIGHT / 2.0
    sigma = min(WIDTH, HEIGHT) * 0.18

    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    noise = rng.uniform(0.85, 1.0, size=blob.shape)
    seed_field = (blob * noise).astype(np.float32)

    field.from_numpy(seed_field)


@ti.kernel
def step(
    dt_center: ti.f32, dt_x_range: ti.f32, dt_y_range: ti.f32,
    mu_center: ti.f32, mu_x_range: ti.f32, mu_y_range: ti.f32,
    sigma_center: ti.f32, sigma_x_range: ti.f32, sigma_y_range: ti.f32,
    radius_center: ti.f32, radius_x_range: ti.f32, radius_y_range: ti.f32,
):
    """Update every cell in the field, with every parameter varying across X and Y."""
    for i, j in field:
        x = float(j) / float(WIDTH - 1)
        y = float(i) / float(HEIGHT - 1)
        xd = x - 0.5
        yd = y - 0.5

        dt = dt_center + xd * dt_x_range + yd * dt_y_range
        mu = mu_center + xd * mu_x_range + yd * mu_y_range
        sigma = ti.max(sigma_center + xd * sigma_x_range + yd * sigma_y_range, 0.001)
        radius_f = radius_center + xd * radius_x_range + yd * radius_y_range
        radius = ti.max(ti.min(ti.i32(radius_f + 0.5), 64), 2)

        acc = 0.0
        norm = 0.0

        # Look at neighbours in a square around the cell
        for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
            ni, nj = wrap(i + di, j + dj)  # wrapped neighbour position

            # Distance from centre, scaled to 0..1
            r = ti.sqrt(float(di * di + dj * dj)) / float(radius)

            # Weight from the kernel (bell-shaped)
            w = kernel(r, 0.5, 0.15)

            # Add weighted neighbour value and accumulate normalization
            acc += field[ni, nj] * w
            norm += w

        # Normalize the accumulator
        if norm > 0.0:
            acc = acc / norm

        # Growth function: cells thrive when neighbors are in sweet spot
        growth = 2.0 * ti.exp(-((acc - mu) ** 2) / (2.0 * sigma * sigma)) - 1.0

        # Apply growth to the cell with time step
        val = field[i, j] + dt * growth * field[i, j]

        # Clamp between 0 and 1
        next_field[i, j] = ti.min(ti.max(val, 0.0), 1.0)

    # Copy updated values back into the main field
    for i, j in field:
        field[i, j] = next_field[i, j]


@ti.kernel
def make_color(scheme: ti.i32):
    """Convert the field values into RGB colours based on selected scheme"""
    for i, j in field:
        v = field[i, j]
        r = 0.0
        g = 0.0
        b = 0.0

        # Scheme 0: Rainbow (black -> blue -> cyan -> green -> yellow -> white)
        if scheme == 0:
            if v < 0.2:
                b = v * 5.0
            elif v < 0.4:
                b = 1.0
                g = (v - 0.2) * 5.0
            elif v < 0.6:
                b = 1.0 - (v - 0.4) * 5.0
                g = 1.0
            elif v < 0.8:
                g = 1.0
                r = (v - 0.6) * 5.0
            else:
                r = 1.0
                g = 1.0
                b = (v - 0.8) * 5.0

        # Scheme 1: Fire (black -> red -> orange -> yellow -> white)
        elif scheme == 1:
            if v < 0.33:
                r = v * 3.0
            elif v < 0.66:
                r = 1.0
                g = (v - 0.33) * 3.0
            else:
                r = 1.0
                g = 1.0
                b = (v - 0.66) * 3.0

        # Scheme 2: Ocean (black -> dark blue -> cyan -> white)
        elif scheme == 2:
            if v < 0.5:
                b = v * 2.0
            else:
                b = 1.0
                r = (v - 0.5) * 2.0
                g = (v - 0.5) * 2.0

        # Scheme 3: Monochrome (black -> white)
        elif scheme == 3:
            r = v
            g = v
            b = v

        # Scheme 4: Plasma (purple -> magenta -> orange -> yellow)
        else:
            if v < 0.33:
                r = v * 1.5
                b = 0.5 + v * 1.5
            elif v < 0.66:
                r = 0.5 + (v - 0.33) * 1.5
                g = (v - 0.33) * 1.5
                b = 1.0 - (v - 0.33) * 1.5
            else:
                r = 1.0
                g = 0.5 + (v - 0.66) * 1.5
                b = 0.0

        # Clamp colours
        r = ti.min(ti.max(r, 0.0), 1.0)
        g = ti.min(ti.max(g, 0.0), 1.0)
        b = ti.min(ti.max(b, 0.0), 1.0)

        image[i, j] = ti.Vector([r, g, b])


def render_frame(scheme_name):
    """Run color mapping and return an 8-bit RGB numpy image."""
    scheme_index = COLOR_SCHEMES.index(scheme_name)
    make_color(scheme_index)
    img_np = image.to_numpy()
    return (img_np * 255).astype(np.uint8)


init_field()


def simulate_tick(
    is_running, steps_per_tick, frame_count,
    dt_center, dt_x_range, dt_y_range,
    mu_center, mu_x_range, mu_y_range,
    sigma_center, sigma_x_range, sigma_y_range,
    radius_center, radius_x_range, radius_y_range,
    color_scheme,
):
    """Timer tick handler: advance the simulation and render a frame."""
    if is_running:
        for _ in range(int(steps_per_tick)):
            step(
                dt_center, dt_x_range, dt_y_range,
                mu_center, mu_x_range, mu_y_range,
                sigma_center, sigma_x_range, sigma_y_range,
                radius_center, radius_x_range, radius_y_range,
            )
            frame_count += 1

    frame = render_frame(color_scheme)
    status = f"Frame {frame_count}  |  {'Running' if is_running else 'Paused'}"
    return frame, frame_count, status


def toggle_running(is_running):
    new_state = not is_running
    return new_state, "Pause" if new_state else "Play"


def restart_simulation():
    init_field()
    frame = render_frame(COLOR_SCHEMES[0])
    return frame, 0, "Frame 0  |  Restarted"


CUSTOM_CSS = """
.gradio-container {
    background: radial-gradient(circle at 20% 20%, #1b1f2e 0%, #0d0f16 55%, #08090d 100%) !important;
}
#lenia-title {
    text-align: center;
    padding: 6px 0 2px 0;
}
#lenia-title h1 {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7dd3fc, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
#lenia-title p {
    color: #8b93a7;
    margin: 2px 0 0 0;
    font-size: 0.9rem;
}
#sim-display {
    overflow: auto !important;
}
#sim-display img {
    border-radius: 14px;
    border: 1px solid #2a2f3d;
    box-shadow: 0 0 40px rgba(124, 58, 237, 0.15);
    width: 2048px !important;
    height: 1600px !important;
    max-width: none !important;
    object-fit: none !important;
}
#status-bar {
    text-align: center;
    color: #9aa4bb;
    font-family: "JetBrains Mono", monospace;
    font-size: 0.85rem;
}
.control-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 14px;
    padding: 8px 12px !important;
}
#play-btn {
    background: linear-gradient(90deg, #7c3aed, #4f46e5) !important;
    border: none !important;
    color: white !important;
}
#restart-btn {
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #e5e7eb !important;
}
"""

AXIS_HELP = "Center is the base value. X-range/Y-range spread the value left-right / top-bottom (negative reverses the gradient)."


def param_group(label, default_center, center_range, default_x_range, default_y_range, spread_range, step):
    """Build a labeled group of Center / X-range / Y-range sliders for one parameter."""
    with gr.Group(elem_classes="control-card"):
        gr.Markdown(f"**{label}**")
        center = gr.Slider(*center_range, value=default_center, step=step, label="Center")
        with gr.Row():
            x_range = gr.Slider(*spread_range, value=default_x_range, step=step, label="X-range")
            y_range = gr.Slider(*spread_range, value=default_y_range, step=step, label="Y-range")
    return center, x_range, y_range


with gr.Blocks(title="Lenia Simulation") as demo:
    is_running = gr.State(True)
    frame_count = gr.State(0)

    with gr.Column(elem_id="lenia-title"):
        gr.Markdown("# Lenia Simulation\n<p>A continuous cellular automaton, fully tunable across space</p>")

    with gr.Row():
        with gr.Column(scale=3):
            display = gr.Image(
                value=render_frame(COLOR_SCHEMES[0]),
                label=None,
                show_label=False,
                elem_id="sim-display",
                interactive=False,
                image_mode="RGB",
                width=WIDTH,
                height=HEIGHT,
            )
            status_text = gr.Markdown("Frame 0  |  Running", elem_id="status-bar")
            with gr.Row():
                play_btn = gr.Button("Pause", elem_id="play-btn", scale=2)
                restart_btn = gr.Button("Restart", elem_id="restart-btn", scale=2)
                speed = gr.Slider(1, 10, value=1, step=1, label="Steps / tick", scale=3)
                color_scheme = gr.Dropdown(COLOR_SCHEMES, value=COLOR_SCHEMES[0], label="Colour scheme", scale=2)

        with gr.Column(scale=2):
            gr.Markdown("### Parameters — every value can vary across X and Y")
            with gr.Accordion("Growth rate (dt)", open=True):
                dt_center, dt_x_range, dt_y_range = param_group(
                    "How fast cells grow or shrink", 0.08, (0.0, 0.3), 0.14, 0.0, (-0.3, 0.3), 0.001
                )
            with gr.Accordion("Target density (mu)", open=True):
                mu_center, mu_x_range, mu_y_range = param_group(
                    "Neighbourhood density cells prefer", 0.125, (0.0, 0.3), 0.0, 0.03, (-0.3, 0.3), 0.001
                )
            with gr.Accordion("Growth window (sigma)", open=False):
                sigma_center, sigma_x_range, sigma_y_range = param_group(
                    "Tolerance around the target density", 0.081, (0.001, 0.3), 0.0, 0.0, (-0.2, 0.2), 0.001
                )
            with gr.Accordion("Kernel radius", open=False):
                radius_center, radius_x_range, radius_y_range = param_group(
                    "Size of the neighbourhood considered", 16, (2, 64), 0, 0, (-60, 60), 1
                )
            gr.Markdown(AXIS_HELP)

    param_inputs = [
        dt_center, dt_x_range, dt_y_range,
        mu_center, mu_x_range, mu_y_range,
        sigma_center, sigma_x_range, sigma_y_range,
        radius_center, radius_x_range, radius_y_range,
    ]

    timer = gr.Timer(0.05, active=True)
    timer.tick(
        fn=simulate_tick,
        inputs=[is_running, speed, frame_count, *param_inputs, color_scheme],
        outputs=[display, frame_count, status_text],
    )

    play_btn.click(fn=toggle_running, inputs=[is_running], outputs=[is_running, play_btn])
    restart_btn.click(fn=restart_simulation, inputs=None, outputs=[display, frame_count, status_text])

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"))
