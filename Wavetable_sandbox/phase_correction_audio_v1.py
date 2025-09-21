import os
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

def fractional_phase_correction(X, N, prev_frame=None):
    """
    Given the FFT spectrum X (of length N) of a frame,
    search for a fractional delay (applied as a linear phase shift)
    that minimizes the discontinuity between the first and last samples
    of the reconstructed waveform.
    
    If prev_frame is provided, minimize discontinuity with the end of that frame.
    """
    k = np.arange(N)

    # Define a function that applies a linear phase (fractional delay) of tau (in samples)
    def apply_tau(tau):
        # Multiply each bin k by exp(-i*2π*k*tau/N)
        return X * np.exp(-1j * 2 * np.pi * k * tau / N)
    
    # Define an error measure based on the context
    def discontinuity(tau):
        X_corr = apply_tau(tau)
        y_corr = np.fft.ifft(X_corr).real
        
        if prev_frame is not None:
            # Minimize discontinuity with previous frame's end
            return np.abs(y_corr[0] - prev_frame[-1])
        else:
            # Default: minimize discontinuity between endpoints of current frame
            return np.abs(y_corr[0] - y_corr[-1])
    
    # Brute-force search for the optimal tau over a 1-sample span
    taus = np.linspace(0, 1, 1000)
    errors = np.array([discontinuity(t) for t in taus])
    best_tau = taus[np.argmin(errors)]
    best_error = np.min(errors)
    
    # Apply the optimal phase correction
    X_corr = apply_tau(best_tau)
    return X_corr, best_tau, best_error

def select_audio_file():
    """Select audio file with proper Tkinter cleanup"""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    try:
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac")]
        )
        if not file_path:
            raise ValueError("No file selected")
        return file_path
    finally:
        # Ensure root is properly destroyed
        root.destroy()

def select_output_directory():
    """Select output directory with proper Tkinter cleanup"""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    try:
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if not dir_path:
            raise ValueError("No directory selected")
        return dir_path
    finally:
        # Ensure root is properly destroyed
        root.destroy()

def process_audio_with_phase_correction(audio_file, output_dir, num_chunks=64, frame_length=2048, visualize=False):
    """
    Process an audio file by:
    1. Loading it
    2. Splitting it into evenly spaced chunks
    3. Applying phase correction for seamless transitions
    4. Saving the result
    """
    # Load audio file
    print(f"Loading {audio_file}...")
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # Trim silence and normalize
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    audio_normalized = audio_trimmed / np.max(np.abs(audio_trimmed))
    
    # Determine chunk positions (evenly spaced)
    audio_length = len(audio_normalized)
    chunk_positions = np.linspace(0, audio_length - frame_length, num_chunks, dtype=int)
    
    # Extract chunks with and without Hamming window
    chunks_windowed = []
    chunks_no_window = []
    
    for pos in chunk_positions:
        if pos + frame_length <= audio_length:
            # Extract chunk
            chunk = audio_normalized[pos:pos + frame_length].copy()
            
            # Store non-windowed version
            chunks_no_window.append(chunk)
            
            # Apply Hamming window for windowed version
            window = np.hamming(len(chunk))
            windowed_chunk = chunk * window
            chunks_windowed.append(windowed_chunk)
    
    print(f"Extracted {len(chunks_windowed)} chunks from audio (with and without windowing)")
    
    # Process both versions (windowed and non-windowed)
    for use_window in [True, False]:
        chunks = chunks_windowed if use_window else chunks_no_window
        window_text = "windowed" if use_window else "no_window"
        
        print(f"\nProcessing {window_text} version:")
        
        # Apply phase correction to each chunk
        corrected_chunks = []
        prev_chunk = None
        
        for i, chunk in enumerate(chunks):
            # FFT of current chunk
            X = np.fft.fft(chunk)
            
            # Apply phase correction
            if i > 0:
                X_corr, tau, error = fractional_phase_correction(X, frame_length, prev_chunk)
                print(f"Chunk {i}: Tau = {tau:.6f}, Error = {error:.6f}")
            else:
                # For first chunk, just ensure it loops properly
                X_corr, tau, error = fractional_phase_correction(X, frame_length)
                print(f"Chunk {i}: Tau = {tau:.6f}, Error = {error:.6f}")
            
            # Convert back to time domain
            chunk_corr = np.fft.ifft(X_corr).real
            
            # Normalize each chunk individually
            chunk_max = np.max(np.abs(chunk_corr))
            if chunk_max > 0:  # Avoid division by zero
                chunk_corr = chunk_corr / chunk_max
            
            corrected_chunks.append(chunk_corr)
            prev_chunk = chunk_corr
        
        # Save as individual files and a concatenated file
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Create output filenames
        output_filename = f"{base_name}_64chunks_phased_{window_text}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # Concatenate all normalized chunks and save
        flattened_chunks = np.concatenate(corrected_chunks)
        sf.write(output_path, flattened_chunks.astype(np.float32), sr, subtype='FLOAT')
        print(f"Saved concatenated normalized chunks to {output_path}")
        
        # Optionally visualize
        if visualize:
            plt.figure(figsize=(12, 8))
            
            # Plot original waveform
            plt.subplot(2, 1, 1)
            plt.plot(audio_normalized)
            plt.title("Original Audio Waveform")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            
            # Plot corrected chunks
            plt.subplot(2, 1, 2)
            plt.plot(flattened_chunks)
            for i in range(1, len(corrected_chunks)):
                chunk_start = i * frame_length
                plt.axvline(x=chunk_start, color='r', linestyle='--', alpha=0.5)
            plt.title(f"Phase-Corrected Normalized Chunks ({window_text})")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{base_name}_visualization_{window_text}.png"))
            # Show both plots (windowed and non-windowed)
            plt.show()
    
    return output_path

def main():
    print("Audio Phase Correction Utility")
    print("------------------------------")
    
    # Select input file and output directory
    audio_file = select_audio_file()
    output_dir = select_output_directory()
    
    # Process the audio file
    process_audio_with_phase_correction(
        audio_file=audio_file,
        output_dir=output_dir,
        num_chunks=64,
        frame_length=2048,
        visualize=True  # Set to False to disable visualization
    )
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
