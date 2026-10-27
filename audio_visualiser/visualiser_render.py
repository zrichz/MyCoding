"""Stage 2: render an mp4 from a band_timestamps datafile (PIL frames piped to ffmpeg)."""
import math
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle", "hexagon", "diamond"]
ACTIVE_COLS = 8
CENTRE_SCALE = 0.5  # white centre hexagon radius relative to the cell
GRID_COLOR = (40, 40, 40)
PALETTE = [(240, 35, 35), (35, 120, 245), (255, 80, 80), (70, 170, 255)]


def _polygon_points(cx, cy, r, sides, rotation):
    return [
        (cx + r * math.cos(rotation + 2 * math.pi * k / sides),
         cy + r * math.sin(rotation + 2 * math.pi * k / sides))
        for k in range(sides)
    ]


def _draw_shape(draw, shape, cx, cy, r, color, rotation):
    w = 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=w)
    elif shape == "square":
        draw.polygon(_polygon_points(cx, cy, r * 1.2, 4, rotation + math.pi / 4), outline=color, width=w)
    elif shape == "diamond":
        draw.polygon(_polygon_points(cx, cy, r * 1.2, 4, rotation), outline=color, width=w)
    elif shape == "triangle":
        draw.polygon(_polygon_points(cx, cy, r * 1.2, 3, rotation - math.pi / 2), outline=color, width=w)
    else:
        draw.polygon(_polygon_points(cx, cy, r, 6, rotation), outline=color, width=w)


def _hex_row(width, y, row, R):
    """All cell centres of one row of a pointy-top hex grid; odd rows are offset by half a cell."""
    dx = math.sqrt(3) * R
    n = int(width / dx - 0.5)
    x0 = (width - (n + 0.5) * dx) / 2 + dx / 2 + (dx / 2 if row % 2 else 0.0)
    return [(x0 + k * dx, y) for k in range(n)]


def _build_objects(data, width, height, life_s, seed):
    """One hexagon per event, snapped to a random cell of a shared hex grid."""
    rng = np.random.default_rng(seed)
    n_bands = len(data["bands"])
    band_h = height / n_bands
    R = band_h / 1.5  # grid row spacing (1.5 R) equals one band height
    per_band = []
    grid = []
    for bi, band in enumerate(data["bands"]):
        rgb = PALETTE[bi % len(PALETTE)]
        y_mid = (n_bands - 1 - bi + 0.5) * band_h  # lowest band at the bottom
        row_cells = _hex_row(width, y_mid, bi, R)
        grid.extend(row_cells)
        start = (len(row_cells) - ACTIVE_COLS) // 2
        cells = row_cells[start:start + ACTIVE_COLS]
        objs = []
        for ev in band["events"]:
            x, y = cells[rng.integers(len(cells))]
            objs.append({"t": ev["t"], "x": x, "y": y, "r": R, "s": ev["strength"]})
        per_band.append({"rgb": rgb, "objs": objs, "starts": np.array([o["t"] for o in objs])})
    return per_band, grid, R


def render_video(data, audio_path, out_path, width, height, fps, life_s, seed, progress=None):
    width = height = min(width, height) // 2 * 2  # square; yuv420p needs even dimensions
    per_band, grid, R = _build_objects(data, width, height, life_s, seed)
    background = Image.new("RGB", (width, height), (0, 0, 0))
    bg_draw = ImageDraw.Draw(background)
    for cx, cy in grid:
        _draw_shape(bg_draw, "hexagon", cx, cy, R, GRID_COLOR, math.pi / 6)
    duration = data["duration"]
    n_frames = int(math.ceil(duration * fps))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-i", audio_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-shortest", out_path,
    ]
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err_file)
    try:
        for f in range(n_frames):
            t = f / fps
            img = background.copy()
            draw = ImageDraw.Draw(img)
            for band in per_band:
                starts = band["starts"]
                if len(starts) == 0:
                    continue
                lo = np.searchsorted(starts, t - 2 * life_s, side="left")
                hi = np.searchsorted(starts, t, side="right")
                for o in band["objs"][lo:hi]:
                    age = min(1.0, max(0.0, (t - o["t"]) / life_s))
                    alpha = (1.0 - age) ** 1.5 * (0.4 + 0.6 * min(1.0, o["s"]))
                    color = tuple(int(c * alpha) for c in band["rgb"])
                    _draw_shape(draw, "hexagon", o["x"], o["y"], o["r"], color, math.pi / 6)
                    age2 = min(1.0, max(0.0, (t - o["t"]) / (2 * life_s)))
                    white = int(255 * (1.0 - age2) ** 1.5)
                    if white > 0:
                        draw.polygon(_polygon_points(o["x"], o["y"], o["r"] * CENTRE_SCALE, 6, math.pi / 6),
                                     fill=(white, white, white))
            proc.stdin.write(img.tobytes())
            if progress is not None and f % 15 == 0:
                progress((f + 1) / n_frames, desc="Rendering frames")
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        proc.wait()
    if proc.returncode != 0:
        err_file.seek(0)
        raise RuntimeError("ffmpeg failed: " + err_file.read().decode(errors="replace")[-500:])
    return out_path
