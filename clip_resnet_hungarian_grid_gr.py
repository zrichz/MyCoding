import os
import glob
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.models as models
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# Global cache for loaded models
MODEL_CACHE = {
    "clip_model": None,
    "clip_processor": None,
    "clip_name": None,
    "resnet_model": None,
    "resnet_name": None,
    "device": None,
}

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def get_safe_device():
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros((1, 1), device="cuda")
            _ = test_tensor + 1
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


def get_hf_auth_token():
    for env_var in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.getenv(env_var)
        if token and token.strip():
            return token.strip()

    token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(token_path):
        try:
            with open(token_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            if token:
                return token
        except Exception:
            pass

    return None


def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    device = get_safe_device()
    if (
        MODEL_CACHE["clip_model"] is None
        or MODEL_CACHE["clip_name"] != model_name
        or MODEL_CACHE["device"] != device
    ):
        hf_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        token = get_hf_auth_token()
        try:
            model = CLIPModel.from_pretrained(model_name, cache_dir=hf_cache_dir, token=token)
            processor = CLIPProcessor.from_pretrained(model_name, cache_dir=hf_cache_dir, token=token)
            model.to(device).eval()
            MODEL_CACHE["clip_model"] = model
            MODEL_CACHE["clip_processor"] = processor
            MODEL_CACHE["clip_name"] = model_name
            MODEL_CACHE["device"] = device
        except Exception as exc:
            raise RuntimeError(
                "CLIP download failed. This usually means the Hugging Face token is missing or expired.\n"
                "1) Check whether a token exists in your environment: echo $HF_TOKEN; echo $HUGGINGFACE_HUB_TOKEN\n"
                "2) Check the local auth file: ls -la ~/.huggingface; cat ~/.huggingface/token\n"
                "3) If missing, log in with: huggingface-cli login\n"
                "   or export HF_TOKEN=your_token_here\n"
                "4) Retry after the token is valid.\n"
                f"Model: {model_name}\n"
                f"Cache dir: {hf_cache_dir}\n"
                f"Original error: {exc}"
            ) from exc
    return MODEL_CACHE["clip_model"], MODEL_CACHE["clip_processor"], device


def load_resnet_model(resnet_type="resnet18"):
    device = get_safe_device()
    if (
        MODEL_CACHE["resnet_model"] is None
        or MODEL_CACHE["resnet_name"] != resnet_type
        or MODEL_CACHE["device"] != device
    ):
        if resnet_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        elif resnet_type == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT
            model = models.resnet34(weights=weights)
        else:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)

        model.fc = torch.nn.Identity()
        model.to(device).eval()
        MODEL_CACHE["resnet_model"] = model
        MODEL_CACHE["resnet_name"] = resnet_type
        MODEL_CACHE["device"] = device
    return MODEL_CACHE["resnet_model"], device


def extract_features(pil_images, resnet_type="resnet18", batch_size=16, progress=gr.Progress()):
    clip_model, clip_processor, device = load_clip_model()
    resnet_model, _ = load_resnet_model(resnet_type)

    resnet_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    clip_embeddings = []
    resnet_embeddings = []

    num_images = len(pil_images)
    for start_idx in range(0, num_images, batch_size):
        end_idx = min(start_idx + batch_size, num_images)
        batch = pil_images[start_idx:end_idx]

        if progress is not None:
            progress((start_idx + len(batch)) / num_images, desc=f"Extracting features {start_idx + len(batch)}/{num_images}")

        # ResNet batch extraction
        rgb_batch = [img.convert("RGB") for img in batch]
        resnet_tensors = torch.stack([resnet_preprocess(img) for img in rgb_batch]).to(device)

        with torch.no_grad():
            res_feats = resnet_model(resnet_tensors)
            res_feats = res_feats.cpu().numpy()
            resnet_embeddings.append(res_feats)

        # CLIP batch extraction
        clip_inputs = clip_processor(images=rgb_batch, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_out = clip_model.get_image_features(**clip_inputs)
            if isinstance(clip_out, torch.Tensor):
                clip_feats = clip_out
            elif hasattr(clip_out, "pooler_output") and clip_out.pooler_output is not None:
                clip_feats = clip_out.pooler_output
            elif hasattr(clip_out, "image_embeds") and clip_out.image_embeds is not None:
                clip_feats = clip_out.image_embeds
            else:
                clip_feats = clip_out[0]

            # L2 normalize
            clip_feats = clip_feats / clip_feats.norm(p=2, dim=-1, keepdim=True)
            clip_feats = clip_feats.cpu().numpy()
            clip_embeddings.append(clip_feats)

    clip_all = np.vstack(clip_embeddings)
    resnet_all = np.vstack(resnet_embeddings)

    # Normalize ResNet embeddings
    res_norms = np.linalg.norm(resnet_all, axis=1, keepdims=True)
    res_norms[res_norms == 0] = 1e-8
    resnet_all = resnet_all / res_norms

    return clip_all, resnet_all


def reduce_to_1d(features, random_seed=42):
    n_samples = features.shape[0]
    if n_samples == 1:
        return np.array([0.5])

    reducer = PCA(n_components=1, random_state=random_seed)
    coords = reducer.fit_transform(features).ravel()

    # Min-max normalization to [0, 1]
    min_val = coords.min()
    max_val = coords.max()
    if max_val - min_val > 1e-8:
        coords_norm = (coords - min_val) / (max_val - min_val)
    else:
        coords_norm = np.full_like(coords, 0.5)

    return coords_norm


def compute_grid_dimensions(n_items, mode="Auto (Square-like)", custom_cols=4, custom_rows=4):
    if mode == "Custom":
        cols = max(1, int(custom_cols))
        rows = max(1, int(custom_rows))
        if rows * cols < n_items:
            rows = math.ceil(n_items / cols)

        #return the custom grid dimensions if mode is "Custom"
        return rows, cols

    # Auto layout: prefer exact square counts, falling back to the next perfect square.
    target = math.ceil(math.sqrt(n_items))
    cols = target
    rows = target
    while rows * cols < n_items:
        if rows <= cols:
            rows += 1
        else:
            cols += 1
    return rows, cols


def get_square_grid_options(max_items=400):
    options = []
    for n in range(2, int(math.isqrt(max_items)) + 1):
        square = n * n
        if square <= max_items:
            options.append(square)
    return options


def fit_thumbnail(img, cell_size, fit_mode="crop", bg_color=(20, 24, 30)):
    rgb_img = img.convert("RGB")
    w, h = rgb_img.size

    if fit_mode == "crop":
        scale = max(cell_size / w, cell_size / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = rgb_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - cell_size) // 2
        top = (new_h - cell_size) // 2
        return resized.crop((left, top, left + cell_size, top + cell_size))
    else:
        scale = min(cell_size / w, cell_size / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = rgb_img.resize((new_w, new_h), Image.LANCZOS)
        thumb = Image.new("RGB", (cell_size, cell_size), bg_color)
        paste_x = (cell_size - new_w) // 2
        paste_y = (cell_size - new_h) // 2
        thumb.paste(resized, (paste_x, paste_y))
        return thumb


def render_quantized_grid(
    images,
    grid_assignments,
    rows,
    cols,
    cell_size=160,
    padding=8,
    bg_color_hex="#14181e",
    border_color_hex="#2a3342",
    fit_mode="crop",
    show_labels=True,
    x_axis_name="CLIP Semantic (1D)",
    y_axis_name="ResNet Features (1D)",
):
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    bg_color = hex_to_rgb(bg_color_hex)
    border_color = hex_to_rgb(border_color_hex)

    margin_left = 60 if show_labels else padding
    margin_bottom = 50 if show_labels else padding
    margin_top = 40 if show_labels else padding
    margin_right = padding

    grid_w = cols * cell_size + (cols + 1) * padding
    grid_h = rows * cell_size + (rows + 1) * padding

    canvas_w = margin_left + grid_w + margin_right
    canvas_h = margin_top + grid_h + margin_bottom

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    try:
        font_large = ImageFont.truetype("DejaVuSans.ttf", 15)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw grid cell slots
    for r in range(rows):
        for c in range(cols):
            x = margin_left + padding + c * (cell_size + padding)
            y = margin_top + padding + r * (cell_size + padding)
            draw.rectangle(
                [x - 1, y - 1, x + cell_size, y + cell_size],
                outline=border_color,
                width=1,
            )

    # Place assigned images
    for img_idx, slot_idx in enumerate(grid_assignments):
        r = slot_idx // cols
        c = slot_idx % cols
        x = margin_left + padding + c * (cell_size + padding)
        y = margin_top + padding + r * (cell_size + padding)

        thumb = fit_thumbnail(images[img_idx], cell_size, fit_mode, bg_color)
        canvas.paste(thumb, (x, y))

    # Optional Axis labels and title
    if show_labels:
        title_text = "Quantized Hungarian 2D Grid"
        draw.text((margin_left + padding, 12), title_text, fill=(220, 225, 235), font=font_large)

        x_label = f"Horizontal (X): {x_axis_name} ->"
        draw.text((margin_left + padding, canvas_h - margin_bottom + 14), x_label, fill=(170, 185, 205), font=font_small)

        y_label = f"Vertical (Y): {y_axis_name}"
        draw.text((10, margin_top + padding), y_label, fill=(170, 185, 205), font=font_small)

    return canvas


def render_continuous_canvas(
    images,
    coords_2d,
    canvas_size=900,
    thumb_size=80,
    bg_color_hex="#14181e",
    x_axis_name="CLIP Semantic (1D)",
    y_axis_name="ResNet Features (1D)",
):
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    bg_color = hex_to_rgb(bg_color_hex)
    margin = thumb_size // 2 + 30
    draw_w = canvas_size - 2 * margin
    draw_h = canvas_size - 2 * margin

    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(canvas)

    try:
        font_large = ImageFont.truetype("DejaVuSans.ttf", 15)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw subtle background grid lines
    for step in np.linspace(0, 1, 5):
        gx = int(margin + step * draw_w)
        gy = int(margin + step * draw_h)
        draw.line([(gx, margin), (gx, canvas_size - margin)], fill=(35, 42, 54), width=1)
        draw.line([(margin, gy), (canvas_size - margin, gy)], fill=(35, 42, 54), width=1)

    # Render thumbnails at continuous (x, y)
    for idx, (x_norm, y_norm) in enumerate(coords_2d):
        center_x = int(margin + x_norm * draw_w)
        center_y = int(margin + y_norm * draw_h)

        thumb = fit_thumbnail(images[idx], thumb_size, fit_mode="crop", bg_color=bg_color)
        paste_x = center_x - thumb_size // 2
        paste_y = center_y - thumb_size // 2

        draw.rectangle(
            [paste_x - 1, paste_y - 1, paste_x + thumb_size, paste_y + thumb_size],
            outline=(70, 85, 110),
            width=1,
        )
        canvas.paste(thumb, (paste_x, paste_y))

    # Axis text
    draw.text((margin, 12), "Continuous 2D Projection (Before Quantization)", fill=(220, 225, 235), font=font_large)
    draw.text((margin, canvas_size - 24), f"X-axis: {x_axis_name} ->", fill=(170, 185, 205), font=font_small)
    draw.text((12, margin), f"Y-axis: {y_axis_name}", fill=(170, 185, 205), font=font_small)

    return canvas


def load_images_from_input(dir_path, max_images=64):
    image_paths = []

    if dir_path and os.path.isdir(dir_path.strip()):
        clean_dir = dir_path.strip()
        for root, _, files in os.walk(clean_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    image_paths.append(os.path.join(root, file))

    image_paths = sorted(image_paths)

    if not image_paths:
        return [], []

    if max_images > 0 and len(image_paths) > max_images:
        image_paths = image_paths[:int(max_images)]

    loaded_images = []
    valid_paths = []
    for p in image_paths:
        try:
            img = Image.open(p)
            img.load()
            loaded_images.append(img)
            valid_paths.append(p)
        except Exception:
            continue

    return loaded_images, valid_paths


def process_image_grid(
    dir_path,
    max_images,
    resnet_type,
    cell_size,
    cell_padding,
    fit_mode,
    show_labels,
    progress=gr.Progress(),
):
    if progress is not None:
        progress(0.05, desc="Scanning and loading images")

    images, paths = load_images_from_input(dir_path, max_images)
    n_images = len(images)

    if n_images < 2:
        raise gr.Error("Please provide at least 2 valid images in the directory or upload list")

    # Step 1: Feature Extraction
    clip_feats, resnet_feats = extract_features(
        images,
        resnet_type=resnet_type,
        batch_size=16,
        progress=progress,
    )

    if progress is not None:
        progress(0.70, desc="Running dimensionality reduction")

    # Step 2: 1D Dimensionality Reduction for CLIP (X) and ResNet (Y)
    clip_1d = reduce_to_1d(clip_feats)
    resnet_1d = reduce_to_1d(resnet_feats)

    x_coords = clip_1d
    y_coords = resnet_1d
    x_name = "CLIP Semantic"
    y_name = f"ResNet ({resnet_type})"

    coords_2d = np.column_stack([x_coords, y_coords])

    if progress is not None:
        progress(0.80, desc="Computing Hungarian optimal grid assignment")

    # Step 3: Quantized Grid Generation (Auto Square-like)
    rows, cols = compute_grid_dimensions(n_images, mode="Auto (Square-like)")
    total_slots = rows * cols

    grid_x_vals = np.linspace(0, 1, cols) if cols > 1 else np.array([0.5])
    grid_y_vals = np.linspace(0, 1, rows) if rows > 1 else np.array([0.5])
    grid_mesh_x, grid_mesh_y = np.meshgrid(grid_x_vals, grid_y_vals)
    grid_slot_coords = np.vstack([grid_mesh_x.ravel(), grid_mesh_y.ravel()]).T

    # Step 4: Hungarian Matching (Linear Sum Assignment)
    cost_matrix = np.linalg.norm(coords_2d[:, None, :] - grid_slot_coords[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # col_ind[i] is the matched grid slot index for image i
    matched_slots = col_ind[np.argsort(row_ind)]

    if progress is not None:
        progress(0.90, desc="Rendering outputs")

    # Step 5: Rendering
    fit_mode_str = "crop" if "Crop" in fit_mode else "pad"
    bg_color = "#777777"  # Mid-gray background

    quantized_img = render_quantized_grid(
        images=images,
        grid_assignments=matched_slots,
        rows=rows,
        cols=cols,
        cell_size=int(cell_size),
        padding=int(cell_padding),
        bg_color_hex=bg_color,
        border_color_hex="#d0d0d0",
        fit_mode=fit_mode_str,
        show_labels=show_labels,
        x_axis_name=f"{x_name} (PCA)",
        y_axis_name=f"{y_name} (PCA)",
    )

    continuous_img = render_continuous_canvas(
        images=images,
        coords_2d=coords_2d,
        canvas_size=max(800, cols * int(cell_size) // 2 + 200),
        thumb_size=max(64, int(cell_size) // 2),
        bg_color_hex=bg_color,
        x_axis_name=f"{x_name} (PCA)",
        y_axis_name=f"{y_name} (PCA)",
    )

    status_msg = (
        f"Processed {n_images} images successfully. "
        f"Grid size: {rows} rows x {cols} columns ({total_slots} total slots). "
        f"X-axis: {x_name}, Y-axis: {y_name} using PCA 1D projection."
    )

    if progress is not None:
        progress(1.0, desc="Completed")

    return quantized_img, continuous_img, status_msg


# Build Gradio UI
with gr.Blocks(title="CLIP & ResNet 2D Image Grid") as demo:
    gr.Markdown("CLIP & ResNet 2D Image Grid")
    gr.Markdown(
        "Extract semantic features with CLIP and visual/structural features with ResNet, "
        "reduce each embedding to 1D axis, and arrange thumbnails onto a regular quantized screen grid "
        "using Hungarian algorithm for minimal topological distortion."
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Input Images")
                dir_input = gr.Textbox(
                    label="Image Directory Path",
                    value="/home/rich/MyCoding/images_general",
                    placeholder="/path/to/image/folder",
                )
                max_images_slider = gr.Dropdown(
                    choices=get_square_grid_options(400),
                    value=36,
                    label="Max Images to Process (Exact Square Counts)",
                )

            with gr.Group():
                gr.Markdown("### Embedding & Dimensionality Reduction")
                resnet_choice = gr.Dropdown(
                    choices=["resnet18", "resnet34", "resnet50"],
                    value="resnet18",
                    label="ResNet Architecture",
                )

            with gr.Group():
                gr.Markdown("### Grid & Layout Configuration")
                with gr.Row():
                    cell_size_slider = gr.Slider(minimum=64, maximum=320, value=160, step=16, label="Cell Size (px)")
                    padding_slider = gr.Slider(minimum=0, maximum=32, value=4, step=2, label="Cell Padding (px)")

                fit_mode_radio = gr.Radio(
                    choices=["Crop to Fill (Square)", "Fit with Padding (Letterbox)"],
                    value="Crop to Fill (Square)",
                    label="Thumbnail Fit Mode",
                )
                show_labels_box = gr.Checkbox(label="Show Axis Labels & Title", value=False)

            run_btn = gr.Button("Generate Grid", variant="primary")

        with gr.Column(scale=2):
            status_text = gr.Textbox(label="Status", interactive=False)

            with gr.Tabs():
                with gr.TabItem("Quantized Grid"):
                    grid_output = gr.Image(label="2D Grid", format="png", type="pil")
                with gr.TabItem("Continuous 2D Canvas"):
                    continuous_output = gr.Image(label="Continuous 2D Projection", format="png", type="pil")

    run_btn.click(
        fn=process_image_grid,
        inputs=[
            dir_input,
            max_images_slider,
            resnet_choice,
            cell_size_slider,
            padding_slider,
            fit_mode_radio,
            show_labels_box,
        ],
        outputs=[grid_output, continuous_output, status_text],
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
