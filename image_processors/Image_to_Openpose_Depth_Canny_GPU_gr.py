import os
import numpy as np
import gradio as gr
from PIL import Image

# controlnet_aux imports mediapipe.solutions at import time, which newer mediapipe lacks
import mediapipe as mp
if not hasattr(mp, "solutions"):
    class _Stub:
        def __getattr__(self, name):
            return _Stub()

        def __iter__(self):
            return iter(())

        def __call__(self, *a, **k):
            return _Stub()

    mp.solutions = _Stub()

import cv2
import torch
from controlnet_aux import OpenposeDetector, MidasDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    return _detector


def run_openpose(image, include_body, include_hands, include_face, detect_res):
    if image is None:
        raise gr.Error("Please upload an image.")
    img = image.convert("RGB")
    w, h = img.size
    det = get_detector()
    res = det(
        img,
        detect_resolution=int(detect_res),
        image_resolution=max(w, h),
        include_body=include_body,
        include_hand=include_hands,
        include_face=include_face,
        output_type="pil",
    )
    res = res.resize((w, h), Image.NEAREST)
    return res


_midas = None


def run_depth(img, detect_res):
    global _midas
    if _midas is None:
        _midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(DEVICE)
    w, h = img.size
    res = _midas(img, detect_resolution=int(detect_res), image_resolution=max(w, h), output_type="pil")
    return res.convert("RGB").resize((w, h), Image.BICUBIC)


def run_canny(img, low, high):
    arr = np.array(img)
    edges = cv2.Canny(arr, int(low), int(high))
    return Image.fromarray(edges).convert("RGB")


def process(image, do_pose, do_depth, do_canny, include_body, include_hands, include_face, detect_res, low, high):
    if image is None:
        raise gr.Error("Please upload an image.")
    img = image.convert("RGB")
    pose = run_openpose(img, include_body, include_hands, include_face, detect_res) if do_pose else None
    depth = run_depth(img, detect_res) if do_depth else None
    canny = run_canny(img, low, high) if do_canny else None
    return pose, depth, canny


def run_batch(in_dir, out_dir, do_pose, do_depth, do_canny, include_body, include_hands, include_face,
              detect_res, low, high, progress=gr.Progress()):
    in_dir = (in_dir or "").strip()
    if not os.path.isdir(in_dir):
        raise gr.Error("Input directory does not exist.")
    out_dir = (out_dir or "").strip() or in_dir
    if not (do_pose or do_depth or do_canny):
        raise gr.Error("Select at least one extraction.")
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
    if not files:
        raise gr.Error("No images found in the input directory.")
    jobs = []
    if do_pose:
        jobs.append(("openpose", lambda im: run_openpose(im, include_body, include_hands, include_face, detect_res)))
    if do_depth:
        jobs.append(("Depth", lambda im: run_depth(im, detect_res)))
    if do_canny:
        jobs.append(("Canny", lambda im: run_canny(im, low, high)))
    for name, _ in jobs:
        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    errors = []
    for f in progress.tqdm(files, desc="Processing"):
        try:
            img = Image.open(os.path.join(in_dir, f)).convert("RGB")
            stem = os.path.splitext(f)[0]
            for name, fn in jobs:
                fn(img).save(os.path.join(out_dir, name, stem + ".png"), format="PNG")
        except Exception as e:
            errors.append(f"{f}: {e}")
    msg = f"Processed {len(files) - len(errors)} of {len(files)} images. Output in {out_dir}"
    if errors:
        msg += "\nErrors:\n" + "\n".join(errors)
    return msg


with gr.Blocks(title="OpenPose Detailed") as demo:
    gr.Markdown("# Pose, Depth and Canny Extraction\nOutputs are the same size as the input.")
    with gr.Tab("Single image"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Input image", format="png", height=400)
                do_pose = gr.Checkbox(True, label="Extract OpenPose")
                body = gr.Checkbox(True, label="Body")
                hands = gr.Checkbox(True, label="Hands")
                face = gr.Checkbox(True, label="Face and head shape")
                do_depth = gr.Checkbox(True, label="Extract depth map")
                do_canny = gr.Checkbox(True, label="Extract canny edges")
                dres = gr.Slider(256, 1536, value=768, step=64, label="Detection resolution (pose and depth)")
                low = gr.Slider(0, 255, value=100, step=1, label="Canny low threshold")
                high = gr.Slider(0, 255, value=200, step=1, label="Canny high threshold")
                btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=3):
                with gr.Row():
                    out_pose = gr.Image(type="pil", label="OpenPose", format="png", height=400)
                    out_depth = gr.Image(type="pil", label="Depth map", format="png", height=400)
                    out_canny = gr.Image(type="pil", label="Canny", format="png", height=400)
        btn.click(process, [inp, do_pose, do_depth, do_canny, body, hands, face, dres, low, high],
                  [out_pose, out_depth, out_canny])

    with gr.Tab("Batch directory"):
        b_in = gr.Textbox(label="Input directory")
        b_out = gr.Textbox(label="Output directory (blank uses the input directory)")
        b_pose = gr.Checkbox(True, label="Extract OpenPose")
        b_body = gr.Checkbox(True, label="Body")
        b_hands = gr.Checkbox(True, label="Hands")
        b_face = gr.Checkbox(True, label="Face and head shape")
        b_depth = gr.Checkbox(True, label="Extract depth map")
        b_canny = gr.Checkbox(True, label="Extract canny edges")
        b_dres = gr.Slider(256, 1536, value=768, step=64, label="Detection resolution (pose and depth)")
        b_low = gr.Slider(0, 255, value=100, step=1, label="Canny low threshold")
        b_high = gr.Slider(0, 255, value=200, step=1, label="Canny high threshold")
        b_btn = gr.Button("Run batch", variant="primary")
        b_status = gr.Textbox(label="Status", lines=4)
        b_btn.click(run_batch, [b_in, b_out, b_pose, b_depth, b_canny, b_body, b_hands, b_face, b_dres, b_low, b_high], b_status)

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
