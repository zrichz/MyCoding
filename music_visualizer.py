"""
Music Visualizer - Non-realtime audio analysis and visualization
Creates spectrogram visualizations from WAV/MP3 files with transient detection.
Outputs 512x512 60fps MP4 videos.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os
from pathlib import Path
import threading
from datetime import datetime
import PIL.Image

class MusicVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Music Visualizer - Spectrogram Generator")
        self.root.geometry("900x700")
        
        # Audio data
        self.audio_file = None
        self.y = None  # audio time series
        self.sr = None  # sample rate
        self.duration = None
        self.S = None  # spectrogram
        self.times = None
        self.freqs = None
        self.transients = None
        
        # Video parameters
        self.output_size = 512
        self.fps = 60
        self.is_processing = False
        
        # Processing parameters (will be set from GUI)
        self.n_mels = 32
        self.time_window = 0.4
        self.colormap = 'plasma'
        self.interpolation = 'bilinear'
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(file_frame, text="Browse", 
                  command=self.browse_file).grid(row=0, column=0, padx=(0, 10))
        
        self.file_var = tk.StringVar(value="No file selected")
        file_label = ttk.Label(file_frame, textvariable=self.file_var)
        file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Analysis info frame
        info_frame = ttk.LabelFrame(main_frame, text="Audio Analysis", padding="10")
        info_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=10, width=30, wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        info_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Spectrogram preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Spectrogram Preview", padding="10")
        preview_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Matplotlib figure for preview
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.patch.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, preview_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Controls frame
        controls_frame = ttk.LabelFrame(main_frame, text="Video Generation", padding="10")
        controls_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        controls_frame.columnconfigure(1, weight=1)
        
        # Video parameters
        ttk.Label(controls_frame, text="Output Size:").grid(row=0, column=0, sticky=tk.W)
        self.size_var = tk.StringVar(value="512x512")
        size_combo = ttk.Combobox(controls_frame, textvariable=self.size_var, 
                                 values=["256x256", "512x512", "720x720", "1024x1024"], 
                                 state="readonly", width=10)
        size_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 20))
        
        ttk.Label(controls_frame, text="FPS:").grid(row=0, column=2, sticky=tk.W)
        self.fps_var = tk.StringVar(value="60")
        fps_combo = ttk.Combobox(controls_frame, textvariable=self.fps_var,
                                values=["24", "30", "60"], state="readonly", width=5)
        fps_combo.grid(row=0, column=3, sticky=tk.W, padx=(10, 20))
        
        # Second row of parameters
        ttk.Label(controls_frame, text="Freq Bands:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.bands_var = tk.StringVar(value="32")
        bands_combo = ttk.Combobox(controls_frame, textvariable=self.bands_var,
                                  values=["16", "32", "64"], state="readonly", width=10)
        bands_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 20), pady=(10, 0))
        
        ttk.Label(controls_frame, text="Time Window:").grid(row=1, column=2, sticky=tk.W, pady=(10, 0))
        self.window_var = tk.StringVar(value="0.4")
        window_combo = ttk.Combobox(controls_frame, textvariable=self.window_var,
                                   values=["0.2", "0.4", "0.8"], state="readonly", width=5)
        window_combo.grid(row=1, column=3, sticky=tk.W, padx=(10, 20), pady=(10, 0))
        
        # Third row of parameters
        ttk.Label(controls_frame, text="Color Scheme:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.colormap_var = tk.StringVar(value="plasma")
        colormap_combo = ttk.Combobox(controls_frame, textvariable=self.colormap_var,
                                     values=["plasma", "viridis", "magma"], state="readonly", width=10)
        colormap_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 20), pady=(10, 0))
        
        ttk.Label(controls_frame, text="Interpolation:").grid(row=2, column=2, sticky=tk.W, pady=(10, 0))
        self.interp_var = tk.StringVar(value="bilinear")
        interp_combo = ttk.Combobox(controls_frame, textvariable=self.interp_var,
                                   values=["nearest", "bilinear"], state="readonly", width=5)
        interp_combo.grid(row=2, column=3, sticky=tk.W, padx=(10, 20), pady=(10, 0))
        
        # Generate button
        self.generate_btn = ttk.Button(controls_frame, text="Generate Video", 
                                      command=self.generate_video, state=tk.DISABLED)
        self.generate_btn.grid(row=0, column=4, padx=(20, 0))
        
        # Progress bar
        self.progress = ttk.Progressbar(controls_frame, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Status label
        self.status_var = tk.StringVar(value="Load an audio file to begin")
        status_label = ttk.Label(controls_frame, textvariable=self.status_var)
        status_label.grid(row=4, column=0, columnspan=5, pady=(5, 0))
        
    def browse_file(self):
        """Browse for audio file"""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if filename:
            self.audio_file = filename
            self.file_var.set(os.path.basename(filename))
            self.status_var.set("Audio file loaded. Analyzing...")
            self.clear_analysis()
            # Auto-analyze the file
            self.analyze_audio()
            
    def clear_analysis(self):
        """Clear previous analysis results"""
        self.y = None
        self.sr = None
        self.S = None
        self.transients = None
        self.info_text.delete(1.0, tk.END)
        self.ax.clear()
        self.canvas.draw()
        self.generate_btn.config(state=tk.DISABLED)
        
    def analyze_audio(self):
        """Analyze the selected audio file"""
        if not self.audio_file:
            messagebox.showerror("Error", "Please select an audio file first")
            return
            
        # Run analysis in separate thread to avoid GUI freezing
        threading.Thread(target=self._analyze_audio_worker, daemon=True).start()
        
    def _analyze_audio_worker(self):
        """Worker thread for audio analysis"""
        try:
            self.status_var.set("Loading audio file...")
            self.root.update()
            
            # Load audio file
            self.y, self.sr = librosa.load(self.audio_file, sr=None)
            self.duration = len(self.y) / self.sr
            
            self.status_var.set("Computing spectrogram...")
            self.root.update()
            
            # Compute mel spectrogram with user-selected bands and 30fps update rate
            # Mel spectrograms look much better for music visualization
            hop_length = int(self.sr / 30)  # 1/30 second time resolution
            n_fft = 2048  # Higher FFT for better frequency resolution
            n_mels = int(self.bands_var.get())  # User-selected mel frequency bands
            
            # Compute mel spectrogram
            self.S = librosa.feature.melspectrogram(
                y=self.y, 
                sr=self.sr, 
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmax=8000  # Focus on lower frequencies where most music content is
            )
            
            # Convert to dB scale
            S_db = librosa.power_to_db(self.S, ref=np.max)
            
            # Generate time axis
            self.times = librosa.times_like(S_db, sr=self.sr, hop_length=hop_length)
            self.freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=8000)
            
            self.status_var.set("Detecting transients...")
            self.root.update()
            
            # Detect transients (sudden changes in energy)
            onset_frames = librosa.onset.onset_detect(
                y=self.y, 
                sr=self.sr, 
                hop_length=hop_length,
                units='frames'
            )
            self.transients = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=hop_length)
            
            # Update GUI on main thread
            self.root.after(0, self._update_analysis_display, S_db)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Error analyzing audio: {str(e)}"))
            self.status_var.set("Analysis failed")
            
    def _update_analysis_display(self, S_db):
        """Update the analysis display with results"""
        # Update info text
        self.info_text.delete(1.0, tk.END)
        
        info = f"""Audio Analysis Results

File: {os.path.basename(self.audio_file)}
Duration: {self.duration:.2f} seconds
Sample Rate: {self.sr} Hz
Samples: {len(self.y):,}

Spectrogram:
• Frequency bins: {len(self.freqs)}
• Time frames: {len(self.times)}
• Max frequency: {self.freqs[-1]:.0f} Hz
• Time resolution: {self.times[1] - self.times[0]:.4f}s

Transients Detected: {len(self.transients)}
• Average interval: {np.mean(np.diff(self.transients)):.2f}s

Ready for video generation
"""
        
        self.info_text.insert(1.0, info)
        
        # Update spectrogram preview
        self.ax.clear()
        
        # Display spectrogram
        img = librosa.display.specshow(
            S_db, 
            x_axis='time', 
            y_axis='hz',
            sr=self.sr,
            hop_length=512,
            ax=self.ax,
            cmap='viridis'
        )
        
        # Mark transients
        for t in self.transients:
            self.ax.axvline(x=t, color='red', alpha=0.7, linewidth=0.8)
        
        # Clean visualizer - no titles or labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Remove colorbar for cleaner look
        # if hasattr(self, 'colorbar'):
        #     self.colorbar.remove()
        # self.colorbar = plt.colorbar(img, ax=self.ax, format='%+2.0f dB')
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Enable video generation
        self.generate_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Analysis complete. Found {len(self.transients)} transients")
        
    def generate_video(self):
        """Generate MP4 video from spectrogram"""
        if self.S is None:
            messagebox.showerror("Error", "Please analyze an audio file first")
            return
            
        # Get parameters from GUI
        size = int(self.size_var.get().split('x')[0])
        fps = int(self.fps_var.get())
        self.time_window = float(self.window_var.get())
        self.colormap = self.colormap_var.get()
        self.interpolation = self.interp_var.get()
        
        # Generate filename
        base_name = Path(self.audio_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{base_name}_spectrogram_{size}x{size}_{fps}fps_{timestamp}.mp4"
        
        # Run generation in separate thread
        threading.Thread(
            target=self._generate_video_worker, 
            args=(size, fps, output_file),
            daemon=True
        ).start()
        
    def _generate_video_worker(self, size, fps, output_file):
        """Worker thread for video generation"""
        try:
            self.is_processing = True
            self.root.after(0, lambda: self.generate_btn.config(state=tk.DISABLED))
            self.root.after(0, lambda: self.status_var.set("Generating video..."))
            
            # Convert spectrogram to dB scale
            S_db = librosa.amplitude_to_db(self.S, ref=np.max)
            
            # Calculate video parameters
            video_duration = self.duration
            total_frames = int(video_duration * fps)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, (size, size))
            
            # Generate frames
            for frame_num in range(total_frames):
                # Update progress
                progress = (frame_num / total_frames) * 100
                self.root.after(0, lambda p=progress: self.progress.config(value=p))
                
                # Calculate current time
                current_time = (frame_num / total_frames) * video_duration
                
                # Find closest time index in spectrogram
                time_idx = np.argmin(np.abs(self.times - current_time))
                
                # Create frame
                frame = self._create_spectrogram_frame(S_db, time_idx, current_time, size)
                
                # Apply post-processing normalization for better contrast
                frame = self._normalize_frame(frame)
                
                # Write frame
                out.write(frame)
                
                # Update status occasionally
                if frame_num % (fps // 4) == 0:  # 4 times per second
                    self.root.after(0, lambda t=current_time, f=frame_num: 
                                   self.status_var.set(f"Generating frame {f}/{total_frames} (time: {t:.1f}s)"))
            
            # Clean up
            out.release()
            
            # Update GUI
            self.root.after(0, self._video_generation_complete, output_file)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Video Error", f"Error generating video: {str(e)}"))
            self.root.after(0, lambda: self.status_var.set("Video generation failed"))
        finally:
            self.is_processing = False
            self.root.after(0, lambda: self.generate_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress.config(value=0))
            
    def _create_spectrogram_frame(self, S_db, time_idx, current_time, size):
        """Create a single frame of the spectrogram visualization"""
        # Create figure for this frame
        fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
        fig.patch.set_facecolor('black')
        
        # Display spectrogram window for current time slice
        # Each frame shows the next sequential window
        window_duration = self.time_window  # User-selected duration
        
        # Find the time indices for this window
        # Start at current_time, end at current_time + window_duration
        start_time = current_time
        end_time = current_time + window_duration
        
        # Find corresponding indices in the spectrogram
        start_idx = np.argmin(np.abs(self.times - start_time))
        end_idx = np.argmin(np.abs(self.times - end_time))
        
        # Ensure we have at least a few frames to display
        if end_idx <= start_idx:
            end_idx = min(start_idx + 6, len(self.times))  # At least 6 frames (~0.1s at 60fps)
        
        time_slice = slice(start_idx, end_idx)
        
        # Display the spectrogram slice without axis labels - just the data
        img = ax.imshow(
            S_db[:, time_slice], 
            aspect='auto',
            origin='lower',
            cmap=self.colormap,
            vmax=0,
            vmin=-80,
            interpolation=self.interpolation,
            extent=[0, 1, 0, 1]  # Normalize to fill entire frame
        )
        
        # Remove all axes elements for full-screen display
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Tight layout
        fig.tight_layout(pad=0)  # Remove padding
        
        # Convert to OpenCV format using a more reliable method
        fig.canvas.draw()
        
        # Save to temporary buffer and read back
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        
        # Read back as image
        import PIL.Image
        pil_img = PIL.Image.open(buf)
        pil_img = pil_img.resize((size, size), PIL.Image.Resampling.LANCZOS)
        
        # Convert to numpy array and then to BGR for OpenCV
        frame_rgb = np.array(pil_img)
        if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:  # RGBA
            frame_rgb = frame_rgb[:, :, :3]  # Remove alpha channel
        elif len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3:  # RGB
            pass  # Already RGB
        else:  # Grayscale
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        buf.close()
        
        # Close figure to free memory
        plt.close(fig)
        
        return frame
        
    def _normalize_frame(self, frame):
        """Apply contrast normalization to improve visibility"""
        # Convert to LAB color space for better luminance control
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels back
        lab = cv2.merge([l, a, b])
        
        # Convert back to BGR
        normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return normalized
        
    def _video_generation_complete(self, output_file):
        """Called when video generation is complete"""
        self.status_var.set(f"Video saved: {output_file}")
        
        # Show success message
        result = messagebox.askyesno(
            "Video Complete", 
            f"Video generated successfully\n\nFile: {output_file}\n\nWould you like to open the containing folder?"
        )
        
        if result:
            import subprocess
            import platform
            
            folder_path = os.path.dirname(os.path.abspath(output_file))
            
            if platform.system() == "Windows":
                os.startfile(folder_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux
                subprocess.run(["xdg-open", folder_path])

def main():
    """Main function"""
    root = tk.Tk()
    app = MusicVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()