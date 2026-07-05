#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
TI Changer SDXL - Convert and manipulate SDXL textual inversion files
Gradio version - Simple web interface
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import gradio as gr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.ndimage import zoom
import safetensors.torch as st
import tempfile
import shutil

# Configure matplotlib for better display
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 200

# Heatmap dimensions
HEATMAP_HEIGHT_CLIP_L = 24
HEATMAP_WIDTH_CLIP_L = 32
HEATMAP_HEIGHT_CLIP_G = 32
HEATMAP_WIDTH_CLIP_G = 40


def analyze_safetensors_file(filepath):
    """Analyze a .safetensors file and return analysis results"""
    try:
        data = st.load_file(filepath)
        
        if not isinstance(data, dict):
            return None, f"File contains {type(data)}, expected dictionary"
        
        # Check for CLIP-L and CLIP-G embeddings
        clip_l_key = None
        clip_g_key = None
        
        for key in data.keys():
            if 'clip_l' in key.lower() or 'emb_l' in key.lower():
                clip_l_key = key
            elif 'clip_g' in key.lower() or 'emb_g' in key.lower():
                clip_g_key = key
        
        if not (clip_l_key and clip_g_key):
            return None, f"Missing CLIP embeddings. Keys found: {list(data.keys())}"
        
        clip_l_tensor = data[clip_l_key]
        clip_g_tensor = data[clip_g_key]
        
        if len(clip_l_tensor.shape) != 2 or len(clip_g_tensor.shape) != 2:
            return None, "Unexpected tensor shapes"
        
        info = {
            'data': data,
            'clip_l': clip_l_tensor,
            'clip_g': clip_g_tensor,
            'clip_l_key': clip_l_key,
            'clip_g_key': clip_g_key,
            'num_vectors': clip_l_tensor.shape[0],
            'message': f"Valid SDXL TI: {clip_l_tensor.shape[0]} vectors, CLIP-L {clip_l_tensor.shape}, CLIP-G {clip_g_tensor.shape}"
        }
        
        return info, info['message']
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_stats_plot(clip_l_np, clip_g_np, filename):
    """Create statistics visualization"""
    numvectors = clip_l_np.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    vector_numbers = list(range(1, numvectors + 1))
    x_pos = np.arange(len(vector_numbers))
    
    # CLIP-L stats
    min_l = [np.min(clip_l_np[i]) for i in range(numvectors)]
    max_l = [np.max(clip_l_np[i]) for i in range(numvectors)]
    ax1.bar(x_pos, min_l, 0.7, label='Min', color='blue', alpha=0.6)
    ax1.bar(x_pos, max_l, 0.7, label='Max', color='red', alpha=0.6)
    ax1.set_title(f'CLIP-L Statistics ({filename})')
    ax1.set_xlabel('Vector Number')
    ax1.set_ylabel('Value')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(vector_numbers)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # CLIP-G stats
    min_g = [np.min(clip_g_np[i]) for i in range(numvectors)]
    max_g = [np.max(clip_g_np[i]) for i in range(numvectors)]
    ax2.bar(x_pos, min_g, 0.7, label='Min', color='blue', alpha=0.6)
    ax2.bar(x_pos, max_g, 0.7, label='Max', color='red', alpha=0.6)
    ax2.set_title(f'CLIP-G Statistics ({filename})')
    ax2.set_xlabel('Vector Number')
    ax2.set_ylabel('Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(vector_numbers)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    return fig


def save_file(data, clip_l_np, clip_g_np, original_name, suffix, clip_l_key, clip_g_key):
    """Save processed tensors to a new file"""
    output_dir = "textual_inversions_output"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = original_name.replace('.safetensors', '')
    output_name = f"{base_name}{suffix}.safetensors"
    output_path = os.path.join(output_dir, output_name)
    
    new_data = {
        clip_l_key: torch.from_numpy(clip_l_np.astype(np.float32)),
        clip_g_key: torch.from_numpy(clip_g_np.astype(np.float32))
    }
    
    st.save_file(new_data, output_path)
    
    return output_path


# === OPERATION: Analyze File ===
def analyze_file(file):
    if file is None:
        return "Please upload a file", None
    
    info, message = analyze_safetensors_file(file.name)
    
    if info is None:
        return message, None
    
    # Create plot
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    
    fig = create_stats_plot(clip_l_np, clip_g_np, os.path.basename(file.name))
    
    return message, fig


# === OPERATION: Smoothing ===
def apply_smoothing(file, kernel_size):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    numvectors = info['num_vectors']
    
    if kernel_size == 1:
        processed_l = clip_l_np.copy()
        processed_g = clip_g_np.copy()
    else:
        # Smooth CLIP-L
        np_l_flat = clip_l_np.flatten()
        smooth_l = np.convolve(np_l_flat, np.ones(kernel_size)/kernel_size, mode='full')
        smooth_l = smooth_l[kernel_size//2:len(smooth_l)-kernel_size//2]
        processed_l = smooth_l.reshape(numvectors, -1)
        
        # Smooth CLIP-G
        np_g_flat = clip_g_np.flatten()
        smooth_g = np.convolve(np_g_flat, np.ones(kernel_size)/kernel_size, mode='full')
        smooth_g = smooth_g[kernel_size//2:len(smooth_g)-kernel_size//2]
        processed_g = smooth_g.reshape(numvectors, -1)
    
    output_path = save_file(
        info['data'], processed_l, processed_g,
        os.path.basename(file.name), f"_sm{kernel_size}",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Smoothing applied with kernel size {kernel_size}. Saved to {output_path}", output_path


# === OPERATION: Mean Vector ===
def create_mean_vector(file):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    numvectors = info['num_vectors']
    
    # Calculate mean vectors
    sd_values_l = [np.std(clip_l_np[i]) for i in range(numvectors)]
    meanSD_l = np.mean(sd_values_l)
    Xmean_l = np.mean(clip_l_np, axis=0)
    sd_val_l = np.std(Xmean_l)
    Xmean_l = Xmean_l * meanSD_l / sd_val_l
    
    sd_values_g = [np.std(clip_g_np[i]) for i in range(numvectors)]
    meanSD_g = np.mean(sd_values_g)
    Xmean_g = np.mean(clip_g_np, axis=0)
    sd_val_g = np.std(Xmean_g)
    Xmean_g = Xmean_g * meanSD_g / sd_val_g
    
    processed_l = Xmean_l.reshape(1, -1)
    processed_g = Xmean_g.reshape(1, -1)
    
    output_path = save_file(
        info['data'], processed_l, processed_g,
        os.path.basename(file.name), "_mean",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Created mean vector from {numvectors} vectors. Saved to {output_path}", output_path


# === OPERATION: Divide by Scalar ===
def divide_by_scalar(file, divisor):
    if file is None:
        return "Please upload a file", None
    
    if divisor == 0:
        return "Cannot divide by zero", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    
    processed_l = clip_l_np / divisor
    processed_g = clip_g_np / divisor
    
    output_path = save_file(
        info['data'], processed_l, processed_g,
        os.path.basename(file.name), f"_div{divisor}",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Divided by {divisor}. Saved to {output_path}", output_path


# === OPERATION: Extract Individual Vectors ===
def extract_vectors(file):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    numvectors = info['num_vectors']
    base_name = os.path.basename(file.name).replace('.safetensors', '')
    
    output_dir = "textual_inversions_output"
    os.makedirs(output_dir, exist_ok=True)
    
    files_created = []
    
    for i in range(numvectors):
        individual_clip_l = info['clip_l'][i:i+1].clone()
        individual_clip_g = info['clip_g'][i:i+1].clone()
        
        individual_data = {
            info['clip_l_key']: individual_clip_l,
            info['clip_g_key']: individual_clip_g
        }
        
        vector_filename = f"{base_name}_v_{i+1:02d}.safetensors"
        filepath = os.path.join(output_dir, vector_filename)
        
        st.save_file(individual_data, filepath)
        files_created.append(vector_filename)
    
    return f"Extracted {numvectors} vectors to {output_dir}: {', '.join(files_created)}", None


# === OPERATION: Top N Vectors ===
def select_top_n(file, n):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    numvectors = info['num_vectors']
    
    if n < 1 or n > numvectors:
        return f"N must be between 1 and {numvectors}", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    
    # Calculate combined magnitude
    vector_magnitudes = []
    for i in range(numvectors):
        mag_l = np.linalg.norm(clip_l_np[i])
        mag_g = np.linalg.norm(clip_g_np[i])
        combined_mag = mag_l + mag_g
        vector_magnitudes.append((i, combined_mag))
    
    # Sort by magnitude
    vector_magnitudes.sort(key=lambda x: x[1], reverse=True)
    
    # Select top N
    top_indices = [idx for idx, _ in vector_magnitudes[:n]]
    top_indices.sort()
    
    selected_clip_l = clip_l_np[top_indices]
    selected_clip_g = clip_g_np[top_indices]
    
    output_path = save_file(
        info['data'], selected_clip_l, selected_clip_g,
        os.path.basename(file.name), f"_top{n}",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Selected top {n} vectors by magnitude. Saved to {output_path}", output_path


# === OPERATION: L2 Normalization ===
def l2_normalize(file):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    
    # L2 normalize each vector
    normalized_l = clip_l_np / (np.linalg.norm(clip_l_np, axis=1, keepdims=True) + 1e-8)
    normalized_g = clip_g_np / (np.linalg.norm(clip_g_np, axis=1, keepdims=True) + 1e-8)
    
    output_path = save_file(
        info['data'], normalized_l, normalized_g,
        os.path.basename(file.name), "_l2norm",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Applied L2 normalization. Saved to {output_path}", output_path


# === OPERATION: Tanh Squashing ===
def tanh_squash(file, scale):
    if file is None:
        return "Please upload a file", None
    
    info, _ = analyze_safetensors_file(file.name)
    if info is None:
        return "Invalid file", None
    
    clip_l_np = info['clip_l'].cpu().detach().numpy()
    clip_g_np = info['clip_g'].cpu().detach().numpy()
    
    squashed_l = np.tanh(clip_l_np / scale) * scale
    squashed_g = np.tanh(clip_g_np / scale) * scale
    
    output_path = save_file(
        info['data'], squashed_l, squashed_g,
        os.path.basename(file.name), f"_tanh{scale}",
        info['clip_l_key'], info['clip_g_key']
    )
    
    return f"Applied tanh squashing with scale {scale}. Saved to {output_path}", output_path


# === BUILD GRADIO INTERFACE ===
def build_interface():
    with gr.Blocks(title="TI Changer SDXL") as demo:
        gr.Markdown("# TI Changer SDXL")
        gr.Markdown("Manipulate SDXL textual inversion files with dual embeddings (CLIP-L and CLIP-G)")
        
        with gr.Tab("Analyze"):
            with gr.Row():
                with gr.Column():
                    file_analyze = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    analyze_btn = gr.Button("Analyze File")
                with gr.Column():
                    analyze_output = gr.Textbox(label="Analysis Results", lines=5)
                    analyze_plot = gr.Plot(label="Statistics")
            
            analyze_btn.click(
                analyze_file,
                inputs=[file_analyze],
                outputs=[analyze_output, analyze_plot]
            )
        
        with gr.Tab("Smoothing"):
            with gr.Row():
                with gr.Column():
                    file_smooth = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    kernel_size = gr.Slider(1, 11, value=3, step=2, label="Kernel Size (odd numbers)")
                    smooth_btn = gr.Button("Apply Smoothing")
                with gr.Column():
                    smooth_output = gr.Textbox(label="Result", lines=3)
                    smooth_file = gr.File(label="Download Processed File")
            
            smooth_btn.click(
                apply_smoothing,
                inputs=[file_smooth, kernel_size],
                outputs=[smooth_output, smooth_file]
            )
        
        with gr.Tab("Mean Vector"):
            with gr.Row():
                with gr.Column():
                    file_mean = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    mean_btn = gr.Button("Create Mean Vector")
                with gr.Column():
                    mean_output = gr.Textbox(label="Result", lines=3)
                    mean_file = gr.File(label="Download Processed File")
            
            mean_btn.click(
                create_mean_vector,
                inputs=[file_mean],
                outputs=[mean_output, mean_file]
            )
        
        with gr.Tab("Divide by Scalar"):
            with gr.Row():
                with gr.Column():
                    file_div = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    divisor = gr.Number(value=2.0, label="Divisor")
                    div_btn = gr.Button("Divide")
                with gr.Column():
                    div_output = gr.Textbox(label="Result", lines=3)
                    div_file = gr.File(label="Download Processed File")
            
            div_btn.click(
                divide_by_scalar,
                inputs=[file_div, divisor],
                outputs=[div_output, div_file]
            )
        
        with gr.Tab("Extract Vectors"):
            with gr.Row():
                with gr.Column():
                    file_extract = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    extract_btn = gr.Button("Extract Individual Vectors")
                with gr.Column():
                    extract_output = gr.Textbox(label="Result", lines=5)
            
            extract_btn.click(
                extract_vectors,
                inputs=[file_extract],
                outputs=[extract_output]
            )
        
        with gr.Tab("Top N Vectors"):
            with gr.Row():
                with gr.Column():
                    file_topn = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    n_vectors = gr.Slider(1, 20, value=5, step=1, label="Number of Top Vectors")
                    topn_btn = gr.Button("Select Top N")
                with gr.Column():
                    topn_output = gr.Textbox(label="Result", lines=3)
                    topn_file = gr.File(label="Download Processed File")
            
            topn_btn.click(
                select_top_n,
                inputs=[file_topn, n_vectors],
                outputs=[topn_output, topn_file]
            )
        
        with gr.Tab("L2 Normalization"):
            with gr.Row():
                with gr.Column():
                    file_l2 = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    l2_btn = gr.Button("Apply L2 Normalization")
                with gr.Column():
                    l2_output = gr.Textbox(label="Result", lines=3)
                    l2_file = gr.File(label="Download Processed File")
            
            l2_btn.click(
                l2_normalize,
                inputs=[file_l2],
                outputs=[l2_output, l2_file]
            )
        
        with gr.Tab("Tanh Squashing"):
            with gr.Row():
                with gr.Column():
                    file_tanh = gr.File(label="Upload .safetensors file", file_types=[".safetensors"])
                    tanh_scale = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Scale Factor")
                    tanh_btn = gr.Button("Apply Tanh Squashing")
                with gr.Column():
                    tanh_output = gr.Textbox(label="Result", lines=3)
                    tanh_file = gr.File(label="Download Processed File")
            
            tanh_btn.click(
                tanh_squash,
                inputs=[file_tanh, tanh_scale],
                outputs=[tanh_output, tanh_file]
            )
    
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(inbrowser=True)
