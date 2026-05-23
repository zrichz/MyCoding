#!/home/rich/MyCoding/venvmycoding313/bin/python
import numpy as np
import gradio as gr
import time

# Fixed seeds and parameters for reproducibility
FIXED_SEED = 12345
ARNOLD_ITERATIONS = 10
BLOCK_SIZE = 16
COLOR_SHIFT = 85

def arnold_cat_map(img, iterations=ARNOLD_ITERATIONS):
    """Arnold's Cat Map - scrambles pixel positions reversibly."""
    h, w = img.shape[:2]
    result = img.copy()
    for _ in range(iterations):
        new_img = np.zeros_like(result)
        for y in range(h):
            for x in range(w):
                new_x = (2*x + y) % w
                new_y = (x + y) % h
                new_img[new_y, new_x] = result[y, x]
        result = new_img
    return result

def arnold_cat_map_inverse(img, iterations=ARNOLD_ITERATIONS):
    """Inverse Arnold's Cat Map."""
    h, w = img.shape[:2]
    result = img.copy()
    for _ in range(iterations):
        new_img = np.zeros_like(result)
        for y in range(h):
            for x in range(w):
                new_x = (x - y) % w
                new_y = (-x + 2*y) % h
                new_img[new_y, new_x] = result[y, x]
        result = new_img
    return result

def shuffle_bits(img, reverse=False):
    """Shuffle bit positions. Same function reverses itself."""
    bit_map = [7, 6, 5, 4, 3, 2, 1, 0] if not reverse else [7, 6, 5, 4, 3, 2, 1, 0]
    result = np.zeros_like(img, dtype=np.uint8)
    for i, bit_pos in enumerate(bit_map):
        result |= ((img >> i) & 1) << bit_pos
    return result

def color_rotation(img, reverse=False):
    """Rotate color values using modular arithmetic."""
    shift = COLOR_SHIFT if not reverse else (256 - COLOR_SHIFT)
    return ((img.astype(np.int16) + shift) % 256).astype(np.uint8)

def block_transpose(img, reverse=False):
    """Shuffle image blocks using fixed seed."""
    h, w, c = img.shape
    blocks_y, blocks_x = h // BLOCK_SIZE, w // BLOCK_SIZE
    
    # Truncate to block-aligned size
    h_aligned = blocks_y * BLOCK_SIZE
    w_aligned = blocks_x * BLOCK_SIZE
    working = img[:h_aligned, :w_aligned].copy()
    
    # Create block grid
    blocks = []
    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = working[by*BLOCK_SIZE:(by+1)*BLOCK_SIZE,
                           bx*BLOCK_SIZE:(bx+1)*BLOCK_SIZE]
            blocks.append(block)
    
    # Create permutation
    np.random.seed(FIXED_SEED)
    indices = np.random.permutation(len(blocks))
    
    if reverse:
        # Inverse permutation
        inverse_indices = np.argsort(indices)
        shuffled = [blocks[i] for i in inverse_indices]
    else:
        shuffled = [blocks[i] for i in indices]
    
    # Reconstruct
    result = np.zeros_like(working)
    for idx, block in enumerate(shuffled):
        by, bx = idx // blocks_x, idx % blocks_x
        result[by*BLOCK_SIZE:(by+1)*BLOCK_SIZE,
               bx*BLOCK_SIZE:(bx+1)*BLOCK_SIZE] = block
    
    # Restore original size
    output = img.copy()
    output[:h_aligned, :w_aligned] = result
    return output

def mix_rgb_channels(img1, img2, img3, mode, use_rgb_mix, use_arnold, use_bit_shuffle, use_color_rot, use_block_transpose, progress=gr.Progress()):
    """
    Mixes RGB channels from 3 input images to create 3 output images.
    Applies optional obfuscation operations based on user selection.
    Mode: 'encode' or 'decode'
    """
    if img1 is None or img2 is None or img3 is None:
        return None, None, None, ""
    
    # Convert to uint8 if needed
    def ensure_uint8(img):
        if img.dtype != np.uint8:
            return (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img
    
    # Convert to RGB if grayscale
    def ensure_rgb(img):
        if len(img.shape) == 2:
            return np.stack([img, img, img], axis=2)
        return img
    
    img1 = ensure_rgb(ensure_uint8(img1))
    img2 = ensure_rgb(ensure_uint8(img2))
    img3 = ensure_rgb(ensure_uint8(img3))
    
    # Find minimum dimensions to crop all images to same size
    min_h = min(img1.shape[0], img2.shape[0], img3.shape[0])
    min_w = min(img1.shape[1], img2.shape[1], img3.shape[1])
    
    # Crop all images to same size
    img1 = img1[:min_h, :min_w]
    img2 = img2[:min_h, :min_w]
    img3 = img3[:min_h, :min_w]
    
    if mode == "encode":
        # ENCODE MODE: Channel mixing (optional) then obfuscation
        if use_rgb_mix:
            output1 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            output2 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            output3 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            
            # Extract channels
            r1, g1, b1 = img1[:, :, 0], img1[:, :, 1], img1[:, :, 2]
            r2, g2, b2 = img2[:, :, 0], img2[:, :, 1], img2[:, :, 2]
            r3, g3, b3 = img3[:, :, 0], img3[:, :, 1], img3[:, :, 2]
            
            total_pixels = min_h * min_w
            update_interval = max(total_pixels // 50, min_h * 2)
            pixel_count = 0
            last_update = time.time()
            
            # Channel mixing
            for y in range(min_h):
                for x in range(min_w):
                    output1[y, x, 0] = r1[y, x]
                    output1[y, x, 1] = r2[y, x]
                    output1[y, x, 2] = r3[y, x]
                    
                    output2[y, x, 0] = g1[y, x]
                    output2[y, x, 1] = g2[y, x]
                    output2[y, x, 2] = g3[y, x]
                    
                    output3[y, x, 0] = b1[y, x]
                    output3[y, x, 1] = b2[y, x]
                    output3[y, x, 2] = b3[y, x]
                    
                    pixel_count += 1
                    
                    if pixel_count % update_interval == 0:
                        current_time = time.time()
                        if current_time - last_update >= 2.0:
                            progress(pixel_count / total_pixels)
                            last_update = current_time
            
            progress(1.0)
            outputs = [output1, output2, output3]
        else:
            # No RGB mixing - just pass through the images
            outputs = [img1.copy(), img2.copy(), img3.copy()]
        
        # Apply obfuscation operations in order
        if use_arnold:
            outputs = [arnold_cat_map(out) for out in outputs]
        if use_bit_shuffle:
            outputs = [shuffle_bits(out) for out in outputs]
        if use_color_rot:
            outputs = [color_rotation(out) for out in outputs]
        if use_block_transpose:
            outputs = [block_transpose(out) for out in outputs]
        
        status = "Encoded with:"
        if use_rgb_mix: status += " RGB mixing,"
        if use_arnold: status += " Arnold Cat Map,"
        if use_bit_shuffle: status += " Bit Shuffle,"
        if use_color_rot: status += " Color Rotation,"
        if use_block_transpose: status += " Block Transpose,"
        status = status.rstrip(',')
        
        return outputs[0], outputs[1], outputs[2], status
    
    else:
        # DECODE MODE: Reverse obfuscation then unmix channels (if needed)
        images = [img1, img2, img3]
        
        # Apply reverse operations in reverse order
        if use_block_transpose:
            images = [block_transpose(img, reverse=True) for img in images]
        if use_color_rot:
            images = [color_rotation(img, reverse=True) for img in images]
        if use_bit_shuffle:
            images = [shuffle_bits(img, reverse=True) for img in images]
        if use_arnold:
            images = [arnold_cat_map_inverse(img) for img in images]
        
        if use_rgb_mix:
            # Unmix channels
            output1 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            output2 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            output3 = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            
            for y in range(min_h):
                for x in range(min_w):
                    # Reconstruct original images from mixed channels
                    output1[y, x, 0] = images[0][y, x, 0]  # R from output1
                    output1[y, x, 1] = images[1][y, x, 0]  # G from output2
                    output1[y, x, 2] = images[2][y, x, 0]  # B from output3
                    
                    output2[y, x, 0] = images[0][y, x, 1]  # R from output1
                    output2[y, x, 1] = images[1][y, x, 1]  # G from output2
                    output2[y, x, 2] = images[2][y, x, 1]  # B from output3
                    
                    output3[y, x, 0] = images[0][y, x, 2]  # R from output1
                    output3[y, x, 1] = images[1][y, x, 2]  # G from output2
                    output3[y, x, 2] = images[2][y, x, 2]  # B from output3
            
            status = "Decoded successfully - original images restored"
            return output1, output2, output3, status
        else:
            # No RGB mixing to undo - just return the deobfuscated images
            status = "Decoded successfully - images restored"
            return images[0], images[1], images[2], status

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# RGB Channel Mixer with Obfuscation\nMixes RGB channels from 3 input images and applies reversible obfuscation operations.")
    gr.Markdown("**Fixed parameters:** Arnold Cat Map (10 iterations), Block Size (16px), Color Shift (85), Seed (12345)")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Images")
            input1 = gr.Image(type="numpy", label="Input Image 1", height=300, format="png")
            input2 = gr.Image(type="numpy", label="Input Image 2", height=300, format="png")
            input3 = gr.Image(type="numpy", label="Input Image 3", height=300, format="png")
            
            gr.Markdown("### Mode and Operations")
            mode = gr.Radio(
                choices=["encode", "decode"],
                value="encode",
                label="Mode"
            )
            
            with gr.Column():
                rgb_mix_check = gr.Checkbox(value=True, label="RGB Channel Mixing (channel separation)")
                arnold_check = gr.Checkbox(value=True, label="Arnold Cat Map (pixel scrambling)")
                bit_shuffle_check = gr.Checkbox(value=True, label="Bit Shuffle (bit reordering)")
                color_rot_check = gr.Checkbox(value=True, label="Color Rotation (palette shift)")
                block_transpose_check = gr.Checkbox(value=True, label="Block Transpose (mosaic shuffle)")
            
            process_btn = gr.Button("Process", variant="primary")
            gr.Markdown("**Note:** Images will be cropped to the smallest common dimensions.\n\n**To reconstruct:** Save the 3 output images, reload them as inputs, select same operations, and choose decode mode.")
        
        with gr.Column():
            gr.Markdown("### Output Images")
            output1 = gr.Image(type="numpy", label="Output 1", height=300, format="png")
            output2 = gr.Image(type="numpy", label="Output 2", height=300, format="png")
            output3 = gr.Image(type="numpy", label="Output 3", height=300, format="png")
            status_text = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("Right-click any output → Save Image As... to save as PNG")
    
    process_btn.click(
        mix_rgb_channels,
        inputs=[input1, input2, input3, mode, rgb_mix_check, arnold_check, bit_shuffle_check, color_rot_check, block_transpose_check],
        outputs=[output1, output2, output3, status_text]
    )

demo.launch(inbrowser=True)
