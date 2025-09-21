# # Detailed Program Summary: Wavetable Generator

# ## Program Flow

# 1. **User Interface Initialization**:
#    - Uses Tkinter to create a minimal GUI
#    - Prompts user to select input and output directories
#    - Sets up logging for operation tracking

# 2. **Audio File Processing Pipeline** (for each WAV file):
#    - Loads audio file with librosa
#    - Preprocesses audio (trimming silence, normalizing amplitude)
#    - Extracts cycles through FFT-based pitch detection
#    - Aligns and normalizes cycles
#    - Generates multiple wavetables (with 4, 16, and 64 frames)
#    - Saves the resulting wavetables as WAV files

# ## Key Functions

# ### `select_directory(prompt)`
# - Creates a Tkinter dialog for directory selection
# - Properly manages Tkinter resources to prevent memory leaks
# - Returns the selected directory path or raises an error if none selected

# ### `extract_cycles_fft(audio, sr, fmin=60, fmax=1000)`
# - Uses librosa's pYIN algorithm for pitch detection
# - Identifies voiced regions with sufficient probability
# - Calculates cycle length based on median fundamental frequency
# - Locates positive-going zero crossings for cycle boundaries
# - Extracts multiple cycles of consistent length

# ### `phase_align_cycles(cycles, target_length=2048)`
# - Resamples each cycle to the target length (2048 samples)
# - Applies Hanning window to smooth cycle edges
# - Normalizes amplitude for consistent output
# - Returns an array of aligned cycles

# ### `generate_wavetable_frames(cycles, num_frames, target_length=2048)`
# - Creates wavetable frames using FFT-based spectral interpolation
# - For sufficient cycles: selects representative cycles evenly
# - For insufficient cycles: creates frames through phase shifting
# - Applies phase vocoder for smooth transitions between frames
# - Normalizes all frames

# ### `load_and_preprocess(file_path)`
# - Loads audio file and converts to mono if needed
# - Trims silence using librosa
# - Normalizes audio amplitude
# - Extracts and aligns cycles
# - Returns aligned cycles and sample rate

# ### `process_file(file_path)`
# - Orchestrates the processing pipeline for a single audio file
# - Generates multiple wavetables with different frame counts
# - Creates appropriate output filenames
# - Saves wavetables as WAV files using soundfile library
# - Handles errors gracefully with logging

# ### `main()`
# - Scans the input directory for WAV files
# - Processes each file sequentially
# - Includes commented code for threading if needed in future

# ## Technical Approach

# The program uses advanced DSP techniques:
# - FFT analysis for pitch detection and spectral manipulation
# - Phase vocoding for ensuring smooth transitions between frames
# - Zero-crossing detection for precise cycle extraction
# - Spectral interpolation for frame generation with consistent timbre
# - Progressive phase shifting for creating variations in sound character

# This approach ensures the generated wavetables are usable in synthesizers, with smooth transitions between frames and consistent amplitude across the wavetable.





import os                           # functions for interacting with the OS, such as file and directory manipulation
import glob                         # Used for finding all file paths matching a specified pattern
import numpy as np                  # Provides support for numerical operations and handling arrays
import librosa                      # A library for audio and music analysis, used for loading, processing, and analyzing audio
import logging                      # Used for tracking events and debug info during program execution
import tkinter as tk                # Provides tools for creating GUIs
from tkinter import filedialog      # Used for creating file and directory selection dialogs in the GUI
from scipy.fft import fft, ifft     # functions for Fast Fourier Transform (FFT) and its inverse (IFFT)
import soundfile as sf              # read and write audio files in various formats


# ----- PARAMETERS -----
# V2: FFT-based wavetable generation with phase alignment

# Create a GUI for directory selection
def select_directory(prompt):
    """Select directory with proper Tkinter cleanup"""
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    
    try:
        selected_dir = filedialog.askdirectory(title=prompt)
        if not selected_dir:
            raise ValueError(f"No directory selected for {prompt}")
        return selected_dir
    finally:
        # Ensure root is properly destroyed
        root.destroy()

# Prompt the user to select input and output directories
input_directory = select_directory("Select the Input Directory")
output_directory = select_directory("Select the Output Directory")

target_frame_length = 2048
num_frames_list = [4, 16, 64]  # Number of frames to create for each wavetable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FFT and Phase Alignment Functions
# ---------------------------------
def extract_cycles_fft(audio, sr, fmin=60, fmax=1000):
    """Extract cycles from audio using FFT-based pitch detection"""
    # Get pitch information
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y=audio, 
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=2048,
        hop_length=512
    )
    
    # Find regions with stable pitch
    where_voiced = np.where(voiced_prob > 0.6)[0]
    if len(where_voiced) == 0:
        logger.warning("No voiced regions detected in audio")
        return []
    
    # Calculate average cycle length based on fundamental frequency
    voiced_f0 = f0[where_voiced]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    
    if len(voiced_f0) == 0:
        logger.warning("No valid f0 values detected")
        return []
    
    # Use median frequency for more stability
    median_f0 = np.median(voiced_f0)
    cycle_length_samples = int(sr / median_f0)
    
    logger.info(f"Detected median f0: {median_f0:.2f} Hz, cycle length: {cycle_length_samples} samples")
    
    # Extract cycles at zero-crossings for better alignment
    zero_crossings = librosa.zero_crossings(audio)
    zero_crossing_indices = np.where(zero_crossings)[0]
    
    # Only use positive-going zero crossings
    pos_crossings = []
    for i in range(1, len(zero_crossing_indices)):
        idx = zero_crossing_indices[i]
        if idx > 0 and idx < len(audio) - cycle_length_samples:
            if audio[idx] > audio[idx-1]:  # positive slope
                pos_crossings.append(idx)
    
    # Extract cycles of consistent length
    cycles = []
    for start_idx in pos_crossings:
        if start_idx + cycle_length_samples < len(audio):
            cycle = audio[start_idx:start_idx + cycle_length_samples]
            cycles.append(cycle)
    
    logger.info(f"Extracted {len(cycles)} initial cycles at zero crossings")
    return cycles

def phase_align_cycles(cycles, target_length=2048):
    """Align and resample cycles for phase coherence"""
    if not cycles:
        return np.array([])
    
    aligned_cycles = []
    
    for cycle in cycles:
        # Resample to target length
        resampled = librosa.resample(cycle, orig_sr=len(cycle), target_sr=target_length)
        
        # Apply window to smooth edges
        window = np.hanning(len(resampled))
        windowed_cycle = resampled * window
        
        # Normalize amplitude
        normalized = windowed_cycle / np.max(np.abs(windowed_cycle))
        
        aligned_cycles.append(normalized)
    
    return np.array(aligned_cycles)

def generate_wavetable_frames(cycles, num_frames, target_length=2048):
    """Generate wavetable frames using FFT-based spectral interpolation"""
    if len(cycles) < 2:
        logger.warning("Not enough cycles to generate wavetable")
        return None
    
    # Analyze cycles in frequency domain
    ffts = np.array([fft(cycle) for cycle in cycles])
    
    # Calculate average magnitude and phase
    avg_mag = np.abs(ffts).mean(axis=0)
    
    # For phase, we'll create a progressive phase shift across frames
    frames = []
    
    # Pick representative cycles from the extracted set
    if len(cycles) >= num_frames:
        # If we have enough cycles, pick evenly spaced ones
        indices = np.linspace(0, len(cycles)-1, num_frames, dtype=int)
        representative_cycles = [cycles[i] for i in indices]
    else:
        # Otherwise, use spectral interpolation to create frames
        for i in range(num_frames):
            # Create progressive phase shift
            phase_shift = 2 * np.pi * i / num_frames
            phase = np.angle(ffts[0]) + phase_shift
            
            # Reconstruct with average magnitude and progressive phase
            frame_spectrum = avg_mag * np.exp(1j * phase)
            frame = np.real(ifft(frame_spectrum))
            
            # Normalize
            frame = frame / np.max(np.abs(frame))
            frames.append(frame[:target_length])
        
        return np.array(frames)
    
    # If using representative cycles, apply phase vocoder for smooth transitions
    frames = representative_cycles
    
    # Apply phase vocoder to align phases between frames
    for i in range(1, len(frames)):
        # Get FFTs
        prev_fft = fft(frames[i-1])
        curr_fft = fft(frames[i])
        
        # Extract magnitude and phase
        prev_mag, prev_phase = np.abs(prev_fft), np.angle(prev_fft)
        curr_mag, curr_phase = np.abs(curr_fft), np.angle(curr_fft)
        
        # Calculate phase advancement
        phase_adv = curr_phase - prev_phase
        
        # Apply smooth phase transition
        new_phase = prev_phase + phase_adv * 0.5
        
        # Reconstruct with adjusted phase
        new_fft = curr_mag * np.exp(1j * new_phase)
        frames[i] = np.real(ifft(new_fft))
        
        # Normalize
        frames[i] = frames[i] / np.max(np.abs(frames[i]))
    
    return np.array(frames)

def load_and_preprocess(file_path):
    """Load and preprocess audio file for wavetable generation"""
    logger.info(f"\nLoading {file_path}...")
    
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    logger.info(f"Loaded {file_path} with {len(audio)} samples at {sr} Hz.")
    
    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
    logger.info(f"After trimming trailing silence, {len(audio_trimmed)} samples remain.")
    
    # Normalize
    audio_normalized = audio_trimmed / np.max(np.abs(audio_trimmed))
    logger.info(f"Trimmed audio normalized.")
    
    # Extract cycles
    cycles = extract_cycles_fft(audio_normalized, sr)
    if not cycles:
        logger.warning(f"No usable cycles found in {file_path}")
        return None, sr
    
    # Align and prepare cycles
    aligned_cycles = phase_align_cycles(cycles, target_length=target_frame_length)
    
    return aligned_cycles, sr

def process_file(file_path):
    """Process a single audio file to generate wavetables"""
    try:
        # Load and preprocess file
        aligned_cycles, sr = load_and_preprocess(file_path)
        if aligned_cycles is None or len(aligned_cycles) == 0:
            logger.warning(f"Could not extract usable cycles from {file_path}")
            return
        
        logger.info(f"Successfully aligned {len(aligned_cycles)} cycles")
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Generate wavetables for different numbers of frames
        for num_frames in num_frames_list:
            # Generate wavetable frames using spectral interpolation
            frames = generate_wavetable_frames(aligned_cycles, num_frames, target_length=target_frame_length)
            
            if frames is None or len(frames) == 0:
                logger.warning(f"Failed to generate wavetable with {num_frames} frames")
                continue
            
            # Create output filename
            output_filename = f"{base_name}_{num_frames}fr.wav"
            output_path = os.path.join(output_directory, output_filename)
            
            # Save wavetable
            # Ensure proper format for soundfile
            frames_float32 = frames.astype(np.float32)

            # Check if we need to reshape the data
            if frames_float32.ndim == 1:
                # Single frame, save as mono
                sf.write(output_path, frames_float32, sr, subtype='FLOAT')
            elif frames_float32.ndim == 2:
                # Multiple frames (2D array)
                # First flatten frames into a single audio file with frame boundaries
                flattened_frames = np.concatenate(frames_float32)
                sf.write(output_path, flattened_frames, sr, subtype='FLOAT')
            else:
                # Log the unexpected shape
                logger.error(f"Unexpected frame data shape: {frames_float32.shape}")
        logger.info(f"Saved wavetable with {num_frames} frames to {output_path}")        
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")

def main():
    # List all WAV files in the input directory
    wav_files = glob.glob(os.path.join(input_directory, "*.wav"))
    logger.info(f"Scanning for .wav files in {input_directory} ...")
    logger.info(f"Found {len(wav_files)} .wav files.")
    
    # Process each file one by one (no threading)
    for file_path in wav_files:
        process_file(file_path)
    
    # If you need threading later, use this instead:
    # with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = [executor.submit(process_file, file_path) for file_path in wav_files]
    #     concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()
