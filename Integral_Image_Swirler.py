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

def circle_similarity_map(image, kernel_size=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    size = max(3, int(kernel_size))
    if size % 2 == 0:
        size += 1

    kernel = np.zeros((size, size), dtype=np.uint8)
    radius = size // 2
    cv2.circle(kernel, (radius, radius), radius, 1, -1)

    response = cv2.matchTemplate(gray, kernel.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    pad = size // 2
    response = cv2.copyMakeBorder(response, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    response = cv2.resize(response, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)

    return response

def process(img, filter_type, block, circle_size, normalise):
    if filter_type == "Filled Circle Similarity":
        result = circle_similarity_map(img, circle_size)
    else:
        result = swirliness_map(img, block)

    if normalise:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)
    return result

with gr.Blocks() as demo:
    gr.Markdown("# 🌀 Pattern Energy Mapper\nUse Haar swirliness or filled-circle similarity filters.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input Image")
            filter_type = gr.Radio(
                choices=["Haar Swirliness", "Filled Circle Similarity"],
                value="Haar Swirliness",
                label="Filter Type",
            )
            block_slider = gr.Slider(minimum=2, maximum=16, step=2, value=8, label="Block Size")
            circle_size_slider = gr.Slider(minimum=3, maximum=65, step=2, value=15, label="Circle Kernel Size")
            normalise_toggle = gr.Checkbox(value=False, label="Enhance contrast (CLAHE)")
            btn = gr.Button("Generate Output Map")
        with gr.Column():
            out = gr.Image(type="numpy", label="Output Map")
            gr.Markdown("You can right-click → Save Image As to download the result.")
    btn.click(process, [inp, filter_type, block_slider, circle_size_slider, normalise_toggle], out)

demo.launch(inbrowser=True)
