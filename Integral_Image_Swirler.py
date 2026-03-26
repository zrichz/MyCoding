import cv2
import numpy as np
import gradio as gr

def integral(img):
    return cv2.integral(img)[1:, 1:]  # remove padded row/col

def rect_sum(ii, x1, y1, x2, y2):
    # safe rectangle sum using integral image
    A = ii[y1, x1]
    B = ii[y1, x2]
    C = ii[y2, x1]
    D = ii[y2, x2]
    return D - B - C + A

def haar_feature(ii, x, y, w, h, mode):
    # compute simple 2-rectangle Haar features
    if mode == "horizontal":
        mid = y + h // 2
        top = rect_sum(ii, x, y, x + w, mid)
        bottom = rect_sum(ii, x, mid, x + w, y + h)
        return bottom - top

    if mode == "vertical":
        mid = x + w // 2
        left = rect_sum(ii, x, y, mid, y + h)
        right = rect_sum(ii, mid, y, x + w, y + h)
        return right - left

    if mode == "diagonal":
        midx = x + w // 2
        midy = y + h // 2
        tl = rect_sum(ii, x, y, midx, midy)
        br = rect_sum(ii, midx, midy, x + w, y + h)
        return br - tl

def swirliness_map(image, block=8):
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ii = integral(gray)

    h, w = gray.shape

    H = np.zeros_like(gray, dtype=np.float32)
    V = np.zeros_like(gray, dtype=np.float32)
    D = np.zeros_like(gray, dtype=np.float32)

    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            H[y:y+block, x:x+block] = haar_feature(ii, x, y, block, block, "horizontal")
            V[y:y+block, x:x+block] = haar_feature(ii, x, y, block, block, "vertical")
            D[y:y+block, x:x+block] = haar_feature(ii, x, y, block, block, "diagonal")

    # structure energy
    energy = np.sqrt(H**2 + V**2 + D**2)
    energy = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return energy

def process(img, block):
    return swirliness_map(img, block)

with gr.Blocks() as demo:
    gr.Markdown("# 🌀 Swirliness Map Generator\nUpload an image to compute Haar-based structure energy.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input Image")
            block_slider = gr.Slider(minimum=1, maximum=16, step=1, value=8, label="Block Size")
            btn = gr.Button("Generate Swirliness Map")
        with gr.Column():
            out = gr.Image(type="numpy", label="Swirliness Map")
            gr.Markdown("You can right-click → Save Image As to download the result.")
    btn.click(process, [inp, block_slider], out)

demo.launch(inbrowser=True)
