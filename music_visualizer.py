"""
Music Visualizer - Non-realtime audio analysis and visualization
Creates spectrogram visualizations from WAV/MP3 files with transient detection.
Outputs 512x512 60fps MP4 videos using Gradio interface.
"""

import gradio as gr
import numpy as np
import librosa
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from datetime import datetime
import PIL.Image
import io

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
    
    # Create preview plot (smaller for 1080p layout)
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('white')
    librosa.display.specshow(S_db, x_axis='time', y_axis='hz', sr=sr, hop_length=512, ax=ax, cmap='viridis')
    
    # Mark transients
    for t in transients:
        ax.axvline(x=t, color='red', alpha=0.7, linewidth=0.8)
    
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    
    # Analysis info
    info = f"""Duration: {duration:.2f}s | Sample Rate: {sr} Hz | Transients: {len(transients)}
Frequency bins: {len(S)} | Time frames: {len(times)}"""
    
    return fig, info

def create_spectrogram_frame(S_db, times, current_time, time_window, size, colormap, interpolation, y, sr):
    """Create single video frame with waveform overlay"""
    fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
    fig.patch.set_facecolor('black')
    
    # Time window slice - maintain constant window size
    start_time = current_time
    end_time = current_time + time_window
    start_idx = np.argmin(np.abs(times - start_time))
    end_idx = np.argmin(np.abs(times - end_time))
    
    if end_idx <= start_idx:
        end_idx = min(start_idx + 6, len(times))
    
    # Calculate desired window width in frames
    desired_width = end_idx - start_idx
    
    # If we're near the end and don't have enough frames, shift the window backward
    if end_idx >= len(times):
        end_idx = len(times)
        start_idx = max(0, end_idx - desired_width)
    
    time_slice = slice(start_idx, end_idx)
    
    # Display spectrogram with no interpolation
    ax.imshow(S_db[:, time_slice], aspect='auto', origin='lower', 
              cmap=colormap, vmax=0, vmin=-80, interpolation='nearest',
              extent=[0, 1, 0, 1])
    
    # Overlay waveform for the same time slice
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    waveform_slice = y[start_sample:end_sample]
    
    if len(waveform_slice) > 0:
        # Normalize waveform to fit in bottom portion of plot
        waveform_normalized = waveform_slice / (np.max(np.abs(waveform_slice)) + 1e-8)
        # Scale to bottom 15% of plot
        waveform_y = 0.075 + waveform_normalized * 0.075
        waveform_x = np.linspace(0, 1, len(waveform_slice))
        ax.plot(waveform_x, waveform_y, color='white', linewidth=0.5, alpha=0.8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    fig.tight_layout(pad=0)
    
    # Convert to image with no filtering
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                facecolor='black', edgecolor='none')
    buf.seek(0)
    
    pil_img = PIL.Image.open(buf)
    # Use NEAREST neighbor only - no bilinear filtering
    pil_img = pil_img.resize((size, size), PIL.Image.Resampling.NEAREST)
    frame_rgb = np.array(pil_img)
    
    if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:
        frame_rgb = frame_rgb[:, :, :3]
    
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    buf.close()
    plt.close(fig)
    
    # Apply polar transformation: map width (time) to 0 to 2π (Tau)
    # Width becomes angular coordinate (0 to 2π), height becomes radial coordinate
    center = (size // 2, size // 2)
    max_radius = size // 2
    frame = cv2.warpPolar(frame, (size, size), center, max_radius, 
                          cv2.WARP_POLAR_LINEAR + cv2.WARP_INVERSE_MAP)
    
    return frame

def normalize_frame(frame):
    """Apply contrast normalization"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def generate_video(audio_file, size, fps, n_mels, time_window, colormap, interpolation, progress=gr.Progress()):
    """Generate video from audio file"""
    if audio_file is None:
        return None, "Please upload an audio file first"
    
    progress(0, desc="Loading audio...")
    
    # Load and analyze
    y, sr = librosa.load(audio_file, sr=None)
    duration = len(y) / sr
    
    progress(0.1, desc="Computing spectrogram...")
    hop_length = int(sr / 30)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=hop_length, 
        n_mels=n_mels, fmax=8000
    )
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    times = librosa.times_like(S_db, sr=sr, hop_length=hop_length)
    
    # Setup video (temporary file without audio)
    base_name = Path(audio_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_video = f"temp_{base_name}_{timestamp}.mp4"
    output_file = f"{base_name}_spectrogram_{size}x{size}_{fps}fps_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (size, size))
    
    total_frames = int(duration * fps)
    
    progress(0.2, desc="Generating frames...")
    
    # Generate frames
    for frame_num in range(total_frames):
        current_time = (frame_num / total_frames) * duration
        
        frame = create_spectrogram_frame(S_db, times, current_time, time_window, 
                                        size, colormap, interpolation, y, sr)
        frame = normalize_frame(frame)
        out.write(frame)
        
        if frame_num % 30 == 0:
            progress((0.2 + 0.75 * frame_num / total_frames), 
                    desc=f"Frame {frame_num}/{total_frames}")
    
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
                    time_window = gr.Dropdown([0.2, 0.4, 0.8, 1.6], value=0.4, label="Window (s)")
                    interpolation = gr.Dropdown(["nearest", "bilinear"], 
                                              value="nearest", label="Interp")
            
            with gr.Row():
                analyze_btn = gr.Button("📊 Analyze", variant="secondary")
                generate_btn = gr.Button("🎬 Generate Video", variant="primary")
            
            analysis_info = gr.Textbox(label="Analysis Info", lines=2)
        
        with gr.Column(scale=3):
            preview_plot = gr.Plot(label="Spectrogram Preview")
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
        inputs=[audio_input, size, fps, n_mels, time_window, colormap, interpolation],
        outputs=[video_output, status]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)