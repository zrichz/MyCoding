"""
Music Visualizer - Non-realtime audio analysis and visualization
Creates spectrogram visualizations from WAV/MP3 files with transient detection.
Outputs 512x512 60fps MP4 videos using Gradio interface.
"""

import gradio as gr
import numpy as np
import librosa
import cv2
import os
from pathlib import Path
from datetime import datetime
import PIL.Image

# Custom colormap implementations (replacing matplotlib)
def apply_colormap(data, colormap_name):
    """Apply colormap to normalized 0-255 uint8 data"""
    # OpenCV built-in colormaps
    cv2_colormaps = {
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
    }
    
    if colormap_name in cv2_colormaps:
        # Apply OpenCV colormap (returns BGR)
        colored = cv2.applyColorMap(data, cv2_colormaps[colormap_name])
        return colored
    else:
        # Fallback to viridis
        colored = cv2.applyColorMap(data, cv2.COLORMAP_VIRIDIS)
        return colored

def precompute_colormap_lut(colormap_name):
    """Pre-compute colormap lookup table for faster application"""
    # Create a 256x1 gradient
    gradient = np.arange(256, dtype=np.uint8).reshape(256, 1)
    # Apply colormap once to create LUT
    lut = apply_colormap(gradient, colormap_name)
    lut = lut.reshape(256, 3)
    # Ensure lowest value is solid black for maximum contrast
    lut[0] = [0, 0, 0]
    return lut

def create_circular_mask(size):
    """Create a circular mask for the given size"""
    center = (size // 2, size // 2)
    radius = size // 2
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    return mask

def apply_hemisphere_symmetry(frame_hemisphere, size):
    """Take a PI (180 degree) hemisphere and mirror it vertically for top-bottom symmetry"""
    # frame_hemisphere is the transformed 0 to PI section (top half)
    half_size = size // 2
    
    # Create full frame
    full_frame = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Top hemisphere (original)
    full_frame[0:half_size, :] = frame_hemisphere[0:half_size, :]
    
    # Bottom hemisphere (vertical flip of top)
    full_frame[half_size:size, :] = np.flipud(frame_hemisphere[0:half_size, :])
    
    return full_frame

def analyze_audio(audio_file, n_mels=32):
    """Analyze audio file and generate spectrogram preview"""
    if audio_file is None:
        return None, "Please upload an audio file"
    
    # Load audio
    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    
    # Compute mel spectrogram
    hop_length = int(sr / 30)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048,
        hop_length=hop_length, n_mels=n_mels, fmax=8000
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Detect transients
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='frames')
    transients = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    
    # Extract 5 seconds from the middle of the audio
    preview_duration = min(5.0, duration)  # Max 5 seconds
    middle_time = duration / 2.0
    start_time = max(0, middle_time - preview_duration / 2.0)
    end_time = min(duration, start_time + preview_duration)
    
    # Find frame indices for the time window
    start_idx = np.argmin(np.abs(times - start_time))
    end_idx = np.argmin(np.abs(times - end_time))
    
    # Slice spectrogram to show only middle 5 seconds
    S_db_preview = S_db[:, start_idx:end_idx]
    
    # Create preview image using CV2 (no matplotlib)
    spec_normalized = np.clip((S_db_preview + 80) / 80 * 255, 0, 255).astype(np.uint8)
    spec_colored = apply_colormap(spec_normalized, 'viridis')
    spec_colored = np.flipud(spec_colored)  # Flip for correct orientation
    
    # Resize for preview (800x300)
    preview_img = cv2.resize(spec_colored, (800, 300), interpolation=cv2.INTER_NEAREST)
    
    # Draw transient markers as red vertical lines (only within preview window)
    preview_transients = [t for t in transients if start_time <= t <= end_time]
    time_per_pixel = (end_time - start_time) / preview_img.shape[1]
    for t in preview_transients:
        x_pos = int((t - start_time) / time_per_pixel)
        if 0 <= x_pos < preview_img.shape[1]:
            cv2.line(preview_img, (x_pos, 0), (x_pos, preview_img.shape[0]), (0, 0, 255), 1)
    
    # Convert BGR to RGB for Gradio display
    preview_img = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
    
    # Analysis info
    info = f"""Duration: {duration:.2f}s | Sample Rate: {sr} Hz | Transients: {len(transients)}
Frequency bins: {len(S)} | Time frames: {len(times)} | Preview: {start_time:.1f}s - {end_time:.1f}s"""
    
    return preview_img, info

def create_spectrogram_frame(S_db, times, current_time, time_window, size, colormap_lut, y, sr, cache, show_waveform):
    """Create single video frame with optional waveform overlay - optimized version"""
    # Time window slice - maintain constant window size
    start_time = current_time
    end_time = current_time + time_window
    start_idx = np.searchsorted(times, start_time)
    end_idx = np.searchsorted(times, end_time)
    
    if end_idx <= start_idx:
        end_idx = min(start_idx + 6, len(times))
    
    # Calculate desired window width in frames
    desired_width = cache['desired_width']
    
    # If we're near the end and don't have enough frames, shift the window backward
    if end_idx >= len(times):
        end_idx = len(times)
        start_idx = max(0, end_idx - desired_width)
    
    # Create spectrogram using direct array to image conversion
    spec_slice = S_db[:, start_idx:end_idx]
    
    # Normalize to 0-255 range (vectorized)
    spec_normalized = np.clip((spec_slice + 80) * 3.1875, 0, 255).astype(np.uint8)
    
    # Apply colormap using pre-computed LUT (much faster)
    spec_colored = colormap_lut[spec_normalized]
    
    # Flip vertically
    spec_colored = np.flipud(spec_colored)
    
    # Resize to target size
    frame = cv2.resize(spec_colored, (size, size), interpolation=cv2.INTER_NEAREST)
    
    # Apply polar transformation with PI range (0 to 180 degrees)
    # This creates a hemisphere (top half) that we'll mirror vertically
    center = cache['center']
    max_radius = cache['max_radius']
    
    # Use WARP_FILL_OUTLIERS to handle the PI mapping properly
    frame = cv2.warpPolar(frame, (size, size), center, max_radius, 
                          cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    
    # Apply top-bottom symmetry by mirroring the hemisphere
    frame = apply_hemisphere_symmetry(frame, size)
    
    # Overlay waveform AFTER polar transform (if enabled)
    if show_waveform:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        waveform_slice = y[start_sample:end_sample]
        
        if len(waveform_slice) > 500:
            # Downsample waveform for faster drawing (take every Nth sample)
            step = len(waveform_slice) // 500
            waveform_slice = waveform_slice[::step]
        
        if len(waveform_slice) > 0:
            # Normalize waveform
            max_val = np.max(np.abs(waveform_slice))
            if max_val > 1e-8:
                waveform_normalized = waveform_slice / max_val
            else:
                waveform_normalized = waveform_slice
            
            # Vectorized point generation
            center_y = cache['center_y']
            waveform_height = cache['waveform_height']  # Now 25% = half screen peak-to-peak
            num_samples = len(waveform_slice)
            
            x_coords = (np.arange(num_samples) * size / num_samples).astype(np.int32)
            y_coords = (center_y + waveform_normalized * waveform_height).astype(np.int32)
            
            # Create points array for polylines (much faster than individual line calls)
            points = np.column_stack((x_coords, y_coords)).reshape((-1, 1, 2))
            
            # Draw waveform as polyline
            cv2.polylines(frame, [points], False, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Apply circular mask to remove artifacts outside the circle
    mask = cache['circular_mask']
    frame[~mask] = 0  # Set everything outside the circle to black
    
    return frame

def generate_video(audio_file, size, fps, n_mels, time_window, colormap, interpolation, show_waveform, progress=gr.Progress()):
    """Generate video from audio file"""
    if audio_file is None:
        return None, "Please upload an audio file first"
    
    progress(0, desc="Loading audio...")
    
    # Load and analyze
    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    
    progress(0.05, desc="Computing spectrogram...")
    hop_length = int(sr / 30)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length, 
        n_mels=n_mels, fmax=8000
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    
    # Pre-compute colormap LUT for faster application
    colormap_lut = precompute_colormap_lut(colormap)
    
    # Pre-calculate values to cache
    initial_window_samples = int(time_window * len(times) / duration)
    circular_mask = create_circular_mask(size)
    cache = {
        'desired_width': initial_window_samples,
        'center': (size // 2, size // 2),
        'max_radius': size // 2,
        'center_y': size // 2,
        'waveform_height': int(size * 0.25),  # 25% for half screen peak-to-peak
        'circular_mask': circular_mask
    }
    
    # Setup video (temporary file without audio)
    base_name = Path(audio_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_video = f"temp_{base_name}_{timestamp}.mp4"
    output_file = f"{base_name}_spectrogram_{size}x{size}_{fps}fps_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (size, size))
    
    total_frames = int(duration * fps)
    
    progress(0.1, desc="Generating frames...")
    
    # Generate frames
    for frame_num in range(total_frames):
        current_time = (frame_num / total_frames) * duration
        
        frame = create_spectrogram_frame(S_db, times, current_time, time_window, 
                                        size, colormap_lut, y, sr, cache, show_waveform)
        out.write(frame)
        
        if frame_num % 30 == 0:
            progress(0.1 + 0.85 * frame_num / total_frames)
    
    out.release()
    
    # Add audio to video using ffmpeg
    progress(0.95, desc="Adding audio...")
    import subprocess
    
    # Use ffmpeg to combine video with audio
    cmd = [
        'ffmpeg', '-y', '-i', temp_video, '-i', audio_file,
        '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental',
        '-shortest', output_file
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(temp_video)  # Clean up temp file
    except subprocess.CalledProcessError:
        # If ffmpeg fails, rename temp file as output
        os.rename(temp_video, output_file)
    
    progress(1.0, desc="Complete!")
    
    return output_file, f"✓ Video saved: {output_file}"

# Build Gradio interface (optimized for 1080p: 1920x1080)
with gr.Blocks(title="Music Visualizer") as app:
    gr.Markdown("# 🎵 Music Visualizer")
    gr.Markdown("Generate MP4 spectrogram visualizations with audio from WAV/MP3 files")
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(label="Upload Audio", type="filepath")
            
            with gr.Row():
                with gr.Column():
                    size = gr.Dropdown([128, 256, 512, 720, 1024], value=512, label="Size (px)")
                    n_mels = gr.Dropdown([8, 16, 32, 64], value=32, label="Freq Bands")
                    colormap = gr.Dropdown(["plasma", "viridis", "magma", "inferno"], 
                                          value="plasma", label="Colors")
                with gr.Column():
                    fps = gr.Dropdown([24, 30, 60], value=60, label="FPS")
                    time_window = gr.Dropdown([0.2, 0.4, 0.8, 1.6], value=0.8, label="Window (s)")
                    interpolation = gr.Dropdown(["nearest", "bilinear"], 
                                              value="nearest", label="Interp")
            
            with gr.Row():
                show_waveform = gr.Checkbox(label="Show Waveform Overlay", value=True)
            
            with gr.Row():
                analyze_btn = gr.Button("📊 Analyze", variant="secondary")
                generate_btn = gr.Button("🎬 Generate Video", variant="primary")
            
            analysis_info = gr.Textbox(label="Analysis Info", lines=2)
        
        with gr.Column(scale=3):
            preview_plot = gr.Image(label="Spectrogram Preview", type="numpy")
            video_output = gr.Video(label="Generated Video", height=300)
            status = gr.Textbox(label="Status", lines=1, show_label=False)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input, n_mels],
        outputs=[preview_plot, analysis_info]
    )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[audio_input, size, fps, n_mels, time_window, colormap, interpolation, show_waveform],
        outputs=[video_output, status]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)