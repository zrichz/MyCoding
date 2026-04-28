import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.signal import butter, filtfilt, savgol_filter, lfilter
from scipy.io.wavfile import write, read
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.widgets import Slider, Button
import os
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
# Add tkinter for file dialog
import tkinter as tk
from tkinter import filedialog

# Constants
num_frames = 64             # Number of frames in the wavetable
samples_per_frame = 2048    # Number of samples per frame
sample_rate = 44100         # Audio sample rate in Hz
fundamental_freq = 1      # Base frequency of the waveform in Hz

# Vowel Formant Frequencies (F1, F2, F3) for different vowel sounds
# These are the characteristic frequencies for vowel sounds in human speech
formant_data = {
    "A": (730, 1090, 2440),
    "E": (660, 1700, 2400),
    "I": (440, 1900, 2800),
    "O": (460, 1150, 2800),
    "U": (350, 900, 2600)
}

# ----------------- Utility Functions -----------------

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to a signal.

    Args:
        signal (np.ndarray): The input signal to filter.
        lowcut (float): The lower cutoff frequency in Hz.
        highcut (float): The upper cutoff frequency in Hz.
        fs (int): The sampling rate in Hz.
        order (int): The order of the filter (default is 4).

    Returns:
        np.ndarray: The filtered signal.
    """
    nyq = 0.5 * fs          # Nyquist frequency
    low = lowcut / nyq      # Normalize the lower cutoff frequency
    high = highcut / nyq    # Normalize the upper cutoff frequency
    b, a = butter(order, [low, high], btype='band')  # Design the bandpass filter
    return filtfilt(b, a, signal)  # Apply the filter to the signal

def save_and_visualize_wavetable(wavetable, filename):
    """Save a preview frame as a WAV file and display 3D visualization."""
    # Save a preview frame as a WAV file
    write(filename, sample_rate, wavetable[32].astype(np.float32))
    
    # Create a 3D visualization of the wavetable
    fig = plt.figure(figsize=(15, 10))  # 25% larger than (12, 8)
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid for 3D surface - ensure X and Z downsampling factors match
    downsample_factor = 2  # Consistent downsampling factor
    X = np.arange(0, samples_per_frame, downsample_factor)
    Y = np.arange(0, num_frames)
    X, Y = np.meshgrid(X, Y)
    Z = wavetable[:, ::downsample_factor]  # Use the same downsample factor for consistency
    
    # Create the 3D surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='black', alpha=0.8)
    
    # Add color bar to show amplitude values
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # Set z-axis limits to "squash" the visualization vertically
    ax.set_zlim(-5, 5)  # Changed from (-3, 3) to (-5, 5)
    
    # Set labels and title
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Frame')
    ax.set_zlabel('Amp')
    ax.set_title('Wavetable')
    
    plt.tight_layout()
    plt.show()
    
    

# ----------------- Wavetable Generation Functions -----------------

def generate_formant_wavetable(selected_vowels):
    """
    Generate a wavetable dynamically based on selected vowels.

    Args:
        selected_vowels (dict): A dictionary of selected vowels and their formant frequencies.

    Returns:
        np.ndarray: A 2D array representing the wavetable (num_frames x samples_per_frame).
    """
    wavetable = np.zeros((num_frames, samples_per_frame))  # Initialize the wavetable

    # Determine the number of frames allocated to each vowel
    vowel_keys = list(selected_vowels.keys())   # List of selected vowel keys
    num_vowels = len(vowel_keys)                # Number of selected vowels
    steps_per_vowel = num_frames // num_vowels  # Frames allocated per vowel

    for frame in range(num_frames):
        # Generate a base sine wave for the current frame
        t = np.linspace(0, 1, samples_per_frame, endpoint=False)  # Time vector
        wave = np.sin(2 * np.pi * fundamental_freq * t)  # Base sine wave

        # Add harmonics to the base wave
        num_harmonics = int(frame / num_frames * 10)  # Number of harmonics increases over time
        for harmonic in range(2, num_harmonics + 2):
            wave += (1 / harmonic) * np.sin(2 * np.pi * fundamental_freq * harmonic * t)

        # Apply spectral shaping using FFT
        spectrum = fft(wave)  # Compute the FFT of the wave
        freqs = np.linspace(0, 1, len(spectrum))  # Frequency bins (normalized)
        envelope = np.exp(-3 * freqs) * (1 - freqs)  # Spectral envelope
        spectrum *= envelope  # Apply the envelope to the spectrum
        wave_shaped = np.real(ifft(spectrum))  # Compute the inverse FFT to get the shaped wave

        # Determine the current vowel based on the frame index
        vowel_idx = min(frame // steps_per_vowel, num_vowels - 1)  # Index of the current vowel
        vowel = vowel_keys[vowel_idx]  # Current vowel
        f1, f2, f3 = selected_vowels[vowel]  # Formant frequencies for the current vowel

        # Apply bandpass filters to simulate the formants
        wave_shaped = bandpass_filter(wave_shaped, f1, f2, sample_rate)  # First formant filter
        wave_shaped = bandpass_filter(wave_shaped, f2, f3, sample_rate)  # Second formant filter

        # Normalize the waveform to the range [-1, 1]
        wave_shaped /= np.max(np.abs(wave_shaped))
        wavetable[frame] = wave_shaped  # Store the waveform in the wavetable

    return wavetable

def generate_harmonic_wavetable(num_harmonics, harmonic_amplitudes=None, falloff_power=1.0):
    """
    Generate a wavetable with a specified number of harmonics.
    
    Args:
        num_harmonics (int): Number of harmonics to include
        harmonic_amplitudes (list, optional): List of amplitudes for each harmonic
        falloff_power (float): Power to which the harmonic number is raised in the falloff formula
                              (amp = 1/h^falloff_power)
    
    Returns:
        np.ndarray: A 2D array representing the wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    if harmonic_amplitudes is None:
        harmonic_amplitudes = [1/(n**falloff_power) for n in range(1, num_harmonics + 1)]
    
    # Create frames with varying harmonic content
    for frame in range(num_frames):
        t = np.linspace(0, 1, samples_per_frame, endpoint=False)
        wave = np.zeros(samples_per_frame)
        
        # Scale factor controls how many harmonics are present in each frame
        scale_factor = (frame + 1) / num_frames
        max_harmonics = max(1, int(scale_factor * num_harmonics))
        
        for h in range(1, max_harmonics + 1):
            if h <= len(harmonic_amplitudes):
                amp = harmonic_amplitudes[h-1]
            else:
                amp = 1/(h**falloff_power)  # Apply user-defined falloff power
            wave += amp * np.sin(2 * np.pi * fundamental_freq * h * t)
        
        # Normalize the waveform
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        wavetable[frame] = wave
    
    return wavetable

def generate_additive_wavetable(harmonic_profiles):
    """
    Generate a wavetable using additive synthesis with precise control over harmonics.
    
    Args:
        harmonic_profiles (list): List of dicts with amplitude and phase for each harmonic
    
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Create frames with varying harmonic content
    for frame in range(num_frames):
        t = np.linspace(0, 1, samples_per_frame, endpoint=False)
        wave = np.zeros(samples_per_frame)
        
        # Calculate the current profile index based on frame position
        profile_idx = min(int(frame / num_frames * len(harmonic_profiles)), len(harmonic_profiles) - 1)
        profile = harmonic_profiles[profile_idx]
        
        # Add all harmonics in the current profile
        for h, (amp, phase) in enumerate(profile, 1):
            wave += amp * np.sin(2 * np.pi * fundamental_freq * h * t + phase)
        
        # Normalize the waveform
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        wavetable[frame] = wave
    
    return wavetable

def generate_subtractive_wavetable(filter_type='lowpass', start_freq=10000, end_freq=500):
    """
    Generate a wavetable using subtractive synthesis.
    
    Args:
        filter_type (str): Type of filter to use ('lowpass', 'highpass', 'bandpass')
        start_freq (float): Starting filter frequency
        end_freq (float): Ending filter frequency
    
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Create a rich source sound (noise or sawtooth)
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    
    # Generate source waves for each frame
    for frame in range(num_frames):
        # Create a rich harmonic source (sawtooth wave)
        source = np.zeros(samples_per_frame)
        for h in range(1, 50):
            source += (1.0/h) * np.sin(2 * np.pi * h * t)
        
        # Calculate filter frequency for this frame
        alpha = frame / (num_frames - 1)
        cutoff_freq = start_freq * (1 - alpha) + end_freq * alpha
        
        # Apply the filter
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        if filter_type == 'lowpass':
            b, a = butter(4, normalized_cutoff, btype='low')
            filtered = lfilter(b, a, source)
        elif filter_type == 'highpass':
            b, a = butter(4, normalized_cutoff, btype='high')
            filtered = lfilter(b, a, source)
        else:  # bandpass
            b, a = butter(4, [normalized_cutoff * 0.5, normalized_cutoff], btype='band')
            filtered = lfilter(b, a, source)
        
        # Normalize
        if np.max(np.abs(filtered)) > 0:
            filtered /= np.max(np.abs(filtered))
        
        wavetable[frame] = filtered
    
    return wavetable

def generate_noise_wavetable(noise_types=None):
    """
    Generate a wavetable with various types of noise.
    
    Args:
        noise_types (list): List of noise types to include
                          ('white', 'pink', 'brown', 'filtered')
    
    Returns:
        np.ndarray: The generated wavetable
    """
    if noise_types is None:
        noise_types = ['white', 'pink', 'brown', 'filtered']
    
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Each frame will have a variation of noise
    for frame in range(num_frames):
        # Determine which noise type to use
        noise_idx = min(int(frame / num_frames * len(noise_types)), len(noise_types) - 1)
        noise_type = noise_types[noise_idx]
        
        # Generate the appropriate noise
        if noise_type == 'white':
            # White noise (flat spectrum)
            noise = np.random.uniform(-1, 1, samples_per_frame)
        
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            white_noise = np.random.uniform(-1, 1, samples_per_frame)
            # Convert to frequency domain
            X = fft(white_noise)
            # Create pink filter (1/f)
            freqs = np.linspace(0, 1, len(X))
            # Avoid division by zero
            freqs[0] = freqs[1]
            pink_filter = 1 / np.sqrt(freqs)
            # Apply filter and convert back to time domain
            X_pink = X * pink_filter
            noise = np.real(ifft(X_pink))
        
        elif noise_type == 'brown':
            # Brown noise (1/f^2 spectrum)
            white_noise = np.random.uniform(-1, 1, samples_per_frame)
            # Convert to frequency domain
            X = fft(white_noise)
            # Create brown filter (1/f^2)
            freqs = np.linspace(0, 1, len(X))
            # Avoid division by zero
            freqs[0] = freqs[1]
            brown_filter = 1 / (freqs * freqs)
            # Apply filter and convert back to time domain
            X_brown = X * brown_filter
            noise = np.real(ifft(X_brown))
        
        else:  # filtered
            # Filtered noise
            white_noise = np.random.uniform(-1, 1, samples_per_frame)
            # Apply a resonant filter
            filter_freq = 1000 * (frame / num_frames + 0.5)  # Vary filter freq with frame
            nyquist = sample_rate / 2
            normalized_cutoff = filter_freq / nyquist
            b, a = butter(2, normalized_cutoff, btype='low') # Lowpass filter
            noise = lfilter(b, a, white_noise)
        
        # Normalize
        if np.max(np.abs(noise)) > 0:
            noise /= np.max(np.abs(noise))
        
        wavetable[frame] = noise
    
    return wavetable

def generate_fm_wavetable(carrier_freq=1.0, modulator_ratio=(0.5, 2.0), mod_index_range=(0.1, 10.0)):
    """
    Generate a wavetable using frequency modulation synthesis.
    
    Args:
        carrier_freq (float): Carrier frequency (normalized)
        modulator_ratio (tuple): Range of modulator/carrier frequency ratios
        mod_index_range (tuple): Range of modulation indices
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Time vector
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    
    # Generate frames with varying FM parameters
    for frame in range(num_frames):
        # Interpolate modulator ratio and modulation index
        frame_ratio = modulator_ratio[0] + (modulator_ratio[1] - modulator_ratio[0]) * (frame / num_frames)
        mod_index = mod_index_range[0] + (mod_index_range[1] - mod_index_range[0]) * (frame / num_frames)
        
        # Calculate modulator frequency
        modulator_freq = carrier_freq * frame_ratio
        
        # Generate FM signal
        modulator = mod_index * np.sin(2 * np.pi * modulator_freq * t)
        wave = np.sin(2 * np.pi * carrier_freq * t + modulator)
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_wavefolding_wavetable(input_freq=1.0, fold_intensity_range=(1, 10)):
    """
    Generate a wavetable using wavefolding distortion.
    
    Args:
        input_freq (float): Input frequency
        fold_intensity_range (tuple): Range of folding intensities
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Time vector
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    
    # Generate sine input wave
    input_wave = np.sin(2 * np.pi * input_freq * t)
    
    for frame in range(num_frames):
        # Interpolate folding intensity
        fold_intensity = fold_intensity_range[0] + (fold_intensity_range[1] - fold_intensity_range[0]) * (frame / num_frames)
        
        # Apply wavefolding (using sinusoidal folder)
        wave = np.sin(input_wave * fold_intensity * np.pi)
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_granular_wavetable(grain_count_range=(5, 50), grain_size_range=(0.01, 0.2)):
    """
    Generate a wavetable using granular synthesis techniques.
    
    Args:
        grain_count_range (tuple): Range of grain counts
        grain_size_range (tuple): Range of grain sizes (in normalized time)
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Generate some source material (could be anything)
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    source_material = np.sin(2 * np.pi * t) + 0.5 * np.sin(6 * np.pi * t)
    
    for frame in range(num_frames):
        # Interpolate grain parameters
        grain_count = int(grain_count_range[0] + (grain_count_range[1] - grain_count_range[0]) * (frame / num_frames))
        grain_size = grain_size_range[0] + (grain_size_range[1] - grain_size_range[0]) * (frame / num_frames)
        
        # Initialize this frame's waveform
        wave = np.zeros(samples_per_frame)
        
        # Create grains
        for g in range(grain_count):
            # Random position in the source material
            pos = random.random()
            
            # Convert grain size to samples
            grain_samples = int(grain_size * samples_per_frame)
            if grain_samples < 2:
                grain_samples = 2  # Ensure minimum grain size
            
            # Calculate grain window
            grain_window = np.hanning(grain_samples)
            
            # Extract grain from source material with wrapping
            grain_indices = [(int(pos * samples_per_frame) + i) % samples_per_frame 
                            for i in range(grain_samples)]
            grain = source_material[grain_indices] * grain_window
            
            # Place grain in the output at a random position
            placement_pos = random.randint(0, samples_per_frame - grain_samples)
            wave[placement_pos:placement_pos + grain_samples] += grain
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_spectral_morphing_wavetable(num_spectra=4):
    """
    Generate a wavetable that morphs between different spectra.
    
    Args:
        num_spectra (int): Number of spectral targets
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Add validation for num_spectra
    if num_spectra < 2:
        print("Warning: At least 2 spectra needed for morphing. Using default.")
        num_spectra = 4
    
    # Create several target spectra
    spectra = []
    for i in range(num_spectra):
        # Create a random spectrum with different characteristics
        spectrum = np.zeros(samples_per_frame // 2 + 1, dtype=complex)
        
        # Different spectral shapes
        if i == 0:
            # Low frequencies dominant
            freqs = np.exp(-np.linspace(0, 10, len(spectrum)))
        elif i == 1:
            # Mid frequencies dominant
            freqs = np.exp(-(np.linspace(0, len(spectrum), len(spectrum)) - len(spectrum)//2)**2 / (len(spectrum)//4)**2)
        elif i == 2:
            # High frequencies dominant
            freqs = np.exp(-np.linspace(10, 0, len(spectrum)))
        else:
            # Random resonant peaks
            freqs = np.zeros_like(spectrum, dtype=float)
            for j in range(5):
                peak_pos = random.randint(0, len(freqs)-1)
                peak_width = len(freqs) // 20 # Width of the peak
                freqs += np.exp(-((np.arange(len(freqs)) - peak_pos) / peak_width)**2)
        
        # Add random phases
        phases = np.random.uniform(0, 2*np.pi, len(spectrum))
        spectrum = freqs * np.exp(1j * phases)
        spectra.append(spectrum)
    
    # Generate frames
    for frame in range(num_frames):
        # Determine which spectra to interpolate between
        spectrum_idx = frame / num_frames * (num_spectra - 1)
        spectrum1_idx = int(spectrum_idx)
        spectrum2_idx = min(spectrum1_idx + 1, num_spectra - 1)
        alpha = spectrum_idx - spectrum1_idx
        
        # Interpolate between spectra
        interp_spectrum = (1 - alpha) * spectra[spectrum1_idx] + alpha * spectra[spectrum2_idx]
        
        # Ensure symmetry for real output (complex conjugate symmetry)
        full_spectrum = np.zeros(samples_per_frame, dtype=complex)
        full_spectrum[:len(interp_spectrum)] = interp_spectrum
        
        # Use complex conjugate symmetry for the second half
        if len(interp_spectrum) > 1:
            full_spectrum[len(interp_spectrum):] = np.conj(interp_spectrum[1:])[::-1]
        
        # Convert to time domain
        wave = np.real(ifft(full_spectrum))
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_chaotic_wavetable(equation_type='logistic'):
    """
    Generate a wavetable using chaotic equations.
    
    Args:
        equation_type (str): Type of chaotic equation to use
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    for frame in range(num_frames):
        # Generate chaotic sequence
        if equation_type == 'logistic':
            # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
            r = 3.7 + 0.3 * (frame / num_frames)  # r parameter (3.7 - 4.0)
            x = 0.5  # Initial condition
            sequence = []
            
            # Generate twice as many values and discard first half (remove transients)
            for _ in range(samples_per_frame * 2):
                x = r * x * (1 - x)
                sequence.append(x)
            
            wave = np.array(sequence[samples_per_frame:])
            
        elif equation_type == 'henon':
            # Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b*x_n
            a = 1.2 + 0.2 * (frame / num_frames)
            b = 0.3
            x, y = 0.1, 0.1  # Initial conditions
            sequence = []
            
            for _ in range(samples_per_frame * 2):
                x_new = 1 - a * x * x + y
                y_new = b * x
                x, y = x_new, y_new
                sequence.append(x)
            
            wave = np.array(sequence[samples_per_frame:])
            
        else:  # Lorenz
            # Approximate Lorenz system with Euler method
            dt = 0.01
            sigma = 10.0
            rho = 28.0 
            beta = 8.0/3.0
            
            # Initial conditions
            x, y, z = 0.1, 0.0, 0.0
            sequence = []
            
            for _ in range(samples_per_frame * 2):
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
                x += dx
                y += dy
                z += dz
                sequence.append(x / 30)  # Scale down to reasonable range
            
            wave = np.array(sequence[samples_per_frame:])
        
        # Normalize to [-1, 1]
        if np.max(wave) != np.min(wave):
            wave = 2 * (wave - np.min(wave)) / (np.max(wave) - np.min(wave)) - 1
        wavetable[frame] = wave
    
    return wavetable

def generate_physical_modeling_wavetable(model_type='string'):
    """
    Generate a wavetable using simple physical modeling techniques.
    
    Args:
        model_type (str): Type of physical model ('string', 'tube', 'membrane')
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    for frame in range(num_frames):
        # Parameters varying with frame
        damping = 0.95 + 0.04 * (frame / num_frames)  # Damping factor
        excitation_pos = 0.1 + 0.8 * (frame / num_frames)  # Excitation position
        
        if model_type == 'string':
            # Simple Karplus-Strong string model
            # Initialize with noise burst at excitation point
            buffer_size = samples_per_frame
            buffer = np.zeros(buffer_size)
            
            # Excite string with short noise burst
            excite_pos = int(excitation_pos * buffer_size)
            excite_width = buffer_size // 20
            start = max(0, excite_pos - excite_width // 2)
            end = min(buffer_size, excite_pos + excite_width // 2)
            buffer[start:end] = np.random.uniform(-1, 1, end-start)
            
            # Run Karplus-Strong algorithm
            wave = np.zeros(samples_per_frame)
            for i in range(samples_per_frame):
                # Read current value
                wave[i] = buffer[0]
                
                # Calculate next value with averaging filter
                new_val = damping * 0.5 * (buffer[0] + buffer[1])
                
                # Shift buffer and insert new value
                buffer = np.roll(buffer, -1)
                buffer[-1] = new_val
        
        elif model_type == 'tube':
            # Simple waveguide model simulating a tube/pipe
            # Two delay lines for traveling waves in both directions
            forward_buffer = np.zeros(samples_per_frame // 2)
            backward_buffer = np.zeros(samples_per_frame // 2)
            
            # Excite with pulse at one end
            forward_buffer[0] = 1.0
            
            # Process waveguide
            wave = np.zeros(samples_per_frame)
            for i in range(samples_per_frame):
                # Output is sum of forward and backward waves
                wave[i] = forward_buffer[0] + backward_buffer[0]
                
                # Calculate reflections at the ends
                left_reflection = -damping * backward_buffer[0]  # Closed end
                right_reflection = damping * forward_buffer[-1]  # Open end
                
                # Update forward and backward buffers
                forward_buffer = np.roll(forward_buffer, -1)
                forward_buffer[-1] = left_reflection
                
                backward_buffer = np.roll(backward_buffer, -1)
                backward_buffer[-1] = right_reflection
        
        else:  # membrane
            # Simple 2D membrane simulation
            size = int(np.sqrt(samples_per_frame))
            membrane = np.zeros((size, size))
            last_membrane = np.zeros((size, size))
            
            # Excite membrane at a point
            exc_x = int(size * excitation_pos)
            exc_y = int(size * 0.5)
            membrane[exc_x, exc_y] = 1.0
            
            # Run finite difference method simulation
            c = 0.5  # Wave speed
            wave_2d = []
            for _ in range(samples_per_frame):
                new_membrane = np.zeros((size, size))
                
                # Update inner points using finite difference equation
                for i in range(1, size-1):
                    for j in range(1, size-1):
                        laplacian = (membrane[i+1,j] + membrane[i-1,j] + 
                                    membrane[i,j+1] + membrane[i,j-1] - 
                                    4 * membrane[i,j])
                        new_membrane[i,j] = 2*membrane[i,j] - last_membrane[i,j] + c*c*laplacian
                
                # Apply boundary conditions (fixed edges)
                new_membrane[0,:] = new_membrane[-1,:] = new_membrane[:,0] = new_membrane[:,-1] = 0
                
                # Apply damping
                new_membrane *= damping
                
                # Store current state for output
                wave_2d.append(np.mean(membrane))
                
                # Update for next iteration
                last_membrane = membrane.copy()
                membrane = new_membrane.copy()
            
            wave = np.array(wave_2d[:samples_per_frame])
        
        # Normalize to [-1, 1]
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        wavetable[frame] = wave
    
    return wavetable

def generate_vocal_wavetable():
    """
    Generate a wavetable that simulates vocal sounds using formant synthesis.
    
    Returns:
        np.ndarray: The generated wavetable
    """
    # Create a sequence of vowels for morphing
    vowel_sequence = {
        "A": formant_data["A"],
        "E": formant_data["E"],
        "I": formant_data["I"],
        "O": formant_data["O"],
        "U": formant_data["U"]
    }
    
    return generate_formant_wavetable(vowel_sequence)

def generate_wave_morphing_wavetable():
    """
    Generate a wavetable that morphs between basic waveforms.
    
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Create basic waveforms
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    
    # Sine wave
    sine = np.sin(2 * np.pi * t)
    
    # Square wave (using Fourier synthesis for smoothness)
    square = np.zeros_like(t)
    for k in range(1, 20, 2):
        square += (4/np.pi) * (1/k) * np.sin(2 * np.pi * k * t)
    
    # Sawtooth wave (using Fourier synthesis for smoothness)
    saw = np.zeros_like(t)
    for k in range(1, 20):
        saw += (2/np.pi) * ((-1)**(k+1)) * (1/k) * np.sin(2 * np.pi * k * t)
    
    # Triangle wave (using Fourier synthesis for smoothness)
    tri = np.zeros_like(t)
    for k in range(1, 20, 2):
        tri += (8/(np.pi*np.pi)) * (1/(k*k)) * np.cos(2 * np.pi * k * t)
    
    # Morph between waveforms
    waveforms = [sine, square, saw, tri]
    sections = len(waveforms) - 1
    frames_per_section = num_frames // sections
    
    for frame in range(num_frames):
        section = min(frame // frames_per_section, sections - 1)
        alpha = (frame % frames_per_section) / frames_per_section
        
        # Linear interpolation between adjacent waveforms
        wave = (1 - alpha) * waveforms[section] + alpha * waveforms[section + 1]
        
        # Normalize
        wave /= np.max(np.abs(wave))
        wavetable[frame] = wave
    
    return wavetable

def generate_fractal_wavetable(fractal_type='mandelbrot'):
    """
    Generate a wavetable using fractal mathematics.
    
    Args:
        fractal_type (str): Type of fractal algorithm to use
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    for frame in range(num_frames):
        if fractal_type == 'mandelbrot':
            # Extract audio from Mandelbrot set
            wave = np.zeros(samples_per_frame)
            
            # Center and zoom factors that change with frame
            center_x = -0.5
            center_y = 0
            zoom = 0.5 * (1 - frame / num_frames) + 0.1
            
            # Maximum iteration count increases with frame
            max_iter = 50 + int(frame / num_frames * 150)
            
            for i in range(samples_per_frame):
                # Map sample index to a point in the complex plane
                angle = 2 * np.pi * i / samples_per_frame
                dist = (i % (samples_per_frame // 8)) / (samples_per_frame // 8)
                re = center_x + zoom * dist * np.cos(angle)
                im = center_y + zoom * dist * np.sin(angle)
                
                # Compute Mandelbrot iteration
                z = 0
                c = complex(re, im)
                iteration = 0
                
                while abs(z) <= 2 and iteration < max_iter:
                    z = z*z + c
                    iteration += 1
                
                # Convert iteration count to amplitude
                if iteration < max_iter:
                    # Smooth coloring formula
                    wave[i] = iteration + 1 - np.log(np.log(abs(z))) / np.log(2)
                else:
                    wave[i] = 0
        
        elif fractal_type == 'julia':
            # Extract audio from Julia set
            wave = np.zeros(samples_per_frame)
            
            # Parameter for Julia set varies with frame
            angle = 2 * np.pi * frame / num_frames
            c = complex(-0.7 + 0.2*np.cos(angle), 0.3*np.sin(angle))
            
            # Maximum iteration count
            max_iter = 100
            
            for i in range(samples_per_frame):
                # Map sample index to a point in the complex plane
                x = (i / samples_per_frame) * 4 - 2
                z = complex(x, x/4)
                
                # Compute Julia iteration
                iteration = 0
                while abs(z) <= 2 and iteration < max_iter:
                    z = z*z + c
                    iteration += 1
                
                # Convert iteration count to amplitude
                if iteration < max_iter:
                    wave[i] = iteration / max_iter
                else:
                    wave[i] = 0
        
        else:  # IFS (Iterated Function System)
            # Use an IFS to generate points
            num_points = 10000
            points = np.zeros(num_points)
            
            # IFS parameters (simplified example)
            funcs = [
                lambda x, p: 0.5*x,
                lambda x, p: 0.5*x + 0.5,
                lambda x, p: 0.5*x - 0.5
            ]
            probs = [0.33, 0.33, 0.34]
            
            # Iterate the IFS
            x = 0
            for n in range(num_points):
                # Choose a function based on probabilities
                f = np.random.choice(range(len(funcs)), p=probs)
                # Apply the function to current point
                x = funcs[f](x, frame/num_frames)
                # Store the point
                points[n] = x
            
            # Convert points to a waveform using histogram
            hist, _ = np.histogram(points, bins=samples_per_frame, range=(-1, 1))
            wave = hist / np.max(hist) if np.max(hist) > 0 else hist
        
        # Normalize to [-1, 1]
        if np.max(wave) != np.min(wave):
            wave = 2 * (wave - np.min(wave)) / (np.max(wave) - np.min(wave)) - 1
        wavetable[frame] = wave
    
    return wavetable

def generate_sample_based_wavetable(sample_path=None):
    """
    Generate a wavetable from an audio sample.
    
    Args:
        sample_path (str): Path to audio sample. If None, uses a built-in sample.
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))

    def synthetic_sample():
        t = np.linspace(0, 0.5, sample_rate // 2)  # Reduce duration to 0.5 seconds
        sample = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t) + 0.25 * np.sin(2 * np.pi * 1320 * t)
        sample *= np.exp(-t)
        return sample

    # If sample path not provided or file does not exist, use synthetic sample
    if sample_path is None or not os.path.exists(sample_path):
        sample = synthetic_sample()
    else:
        try:
            sample_rate_file, sample = read(sample_path)
            if len(sample.shape) > 1:
                sample = np.mean(sample, axis=1) # Convert to mono
            sample = sample.astype(float) / max(np.max(np.abs(sample)), 1)
        except Exception:
            sample = synthetic_sample()

    # ...existing code for slicing and processing sample...
    sample_length = len(sample)
    slices = []
    if sample_length < samples_per_frame:
        repetitions = int(np.ceil(samples_per_frame / sample_length))
        sample = np.tile(sample, repetitions)[:samples_per_frame]
        slices = [sample]
    else:
        for i in range(num_frames):
            start_idx = int((sample_length - samples_per_frame) * i / max(1, num_frames - 1))
            slices.append(sample[start_idx:start_idx + samples_per_frame])
            if len(slices[-1]) < samples_per_frame:
                padding = samples_per_frame - len(slices[-1])
                slices[-1] = np.pad(slices[-1], (0, padding), 'constant')

    # Precompute the four operations at anchor frames
    anchor_frames = [0, 21, 42, 63]
    anchor_ops = []
    for idx, frame in enumerate(anchor_frames):
        if frame < len(slices):
            wave = slices[frame].copy()
        else:
            wave = slices[0].copy()
        if idx == 0:
            indices = np.linspace(0, len(wave) - 1, samples_per_frame)
            op_wave = np.interp(indices, np.arange(len(wave)), wave)
        elif idx == 1:
            nyquist = sample_rate / 2
            cutoff = (0.1 + 0.8 * (frame / num_frames)) * nyquist
            b, a = butter(4, cutoff / nyquist, btype='low')
            op_wave = lfilter(b, a, wave)
        elif idx == 2:
            spectrum = fft(wave)
            shift_amount = int(len(spectrum) * 0.1 * (frame / num_frames))
            spectrum = np.roll(spectrum, shift_amount)
            op_wave = np.real(ifft(spectrum))
        elif idx == 3:
            env = np.exp(-5 * np.linspace(0, 1, samples_per_frame) ** (frame / num_frames))
            op_wave = wave * env
        if np.max(np.abs(op_wave)) > 0:
            op_wave /= np.max(np.abs(op_wave))
        anchor_ops.append(op_wave)

    # For each frame, interpolate quadratically between the four anchor operations
    for frame in range(num_frames):
        if frame in anchor_frames:
            op_idx = anchor_frames.index(frame)
            wave = anchor_ops[op_idx].copy()
        else:
            # Quadratic interpolation between anchor frames
            # Find which three anchors this frame is between
            if frame < anchor_frames[1]:
                # Between 0 and 21
                f0, f1, f2 = anchor_frames[0], anchor_frames[1], anchor_frames[2]
                w0, w1, w2 = anchor_ops[0], anchor_ops[1], anchor_ops[2]
            elif frame < anchor_frames[2]:
                # Between 21 and 42
                f0, f1, f2 = anchor_frames[1], anchor_frames[2], anchor_frames[3]
                w0, w1, w2 = anchor_ops[1], anchor_ops[2], anchor_ops[3]
            else:
                # Between 42 and 63
                f0, f1, f2 = anchor_frames[2], anchor_frames[3], anchor_frames[3]
                w0, w1, w2 = anchor_ops[2], anchor_ops[3], anchor_ops[3]
            # Quadratic Lagrange interpolation weights
            x = frame
            denom0 = (f0 - f1) * (f0 - f2)
            denom1 = (f1 - f0) * (f1 - f2)
            denom2 = (f2 - f0) * (f2 - f1)
            l0 = ((x - f1) * (x - f2)) / denom0 if denom0 != 0 else 0
            l1 = ((x - f0) * (x - f2)) / denom1 if denom1 != 0 else 0
            l2 = ((x - f0) * (x - f1)) / denom2 if denom2 != 0 else 0
            wave = l0 * w0 + l1 * w1 + l2 * w2
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        wavetable[frame] = wave

    return wavetable

def generate_bitcrushed_wavetable(bit_depth_range=(2, 16), base_waveform='sine'):
    """
    Generate a wavetable with varying levels of bit depth reduction.
    
    Args:
        bit_depth_range (tuple): Range of bit depths to use
        base_waveform (str): Base waveform type to bitcrush
        
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Generate base waveform
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    
    if base_waveform == 'sine':
        base = np.sin(2 * np.pi * t)
    elif base_waveform == 'square':
        base = np.sign(np.sin(2 * np.pi * t))
    elif base_waveform == 'sawtooth':
        base = 2 * (t - np.floor(t + 0.5))
    else:  # triangle
        base = 1 - 4 * np.abs(np.round(t - 0.25) - (t - 0.25))
    
    # Apply bitcrushing with varying bit depth
    min_bit_depth, max_bit_depth = bit_depth_range
    
    for frame in range(num_frames):
        # Calculate bit depth for this frame
        bit_depth = min_bit_depth + (max_bit_depth - min_bit_depth) * (1 - frame / num_frames)
        
        # Create bitcrushing quantization
        levels = 2 ** bit_depth
        wave = np.round(base * (levels/2)) / (levels/2)
        
        # Also apply sample rate reduction (downsampling)
        downsample_factor = max(1, int(2 + 30 * (frame / num_frames)**2))
        
        # Downsample and then upsample to create "stair step" effect
        downsampled = wave[::downsample_factor]
        if len(downsampled) < 2:
            downsampled = np.array([0, 0])
        
        # Use numpy's interp for zero-order hold (sample and hold)
        indices = np.arange(len(downsampled))
        new_indices = np.linspace(0, len(downsampled) - 1, samples_per_frame)
        wave = np.interp(new_indices, indices, downsampled)
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_user_drawn_wavetable():
    """
    Allow user to draw waveforms using matplotlib interactive interface.
    
    Returns:
        np.ndarray: The generated wavetable
    """
    wavetable = np.zeros((num_frames, samples_per_frame))
    
    # Create initial waveform (sine wave)
    t = np.linspace(0, 1, samples_per_frame, endpoint=False)
    initial_wave = np.sin(2 * np.pi * t)
    
    # Set up the figure for drawing
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(t, initial_wave)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Draw Your Wavetable (Close window when done)")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    
    # Add text instructions
    plt.figtext(0.5, 0.01, "Click and drag to modify the wave. Close window when done.", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    
    # Store drawn coordinates
    drawn_x = []
    drawn_y = []
    
    # Function to handle mouse dragging
    def on_move(event):
        if event.inaxes != ax:
            return
        if event.button != 1:  # Only respond to left mouse button
            return
        
        # Add the current position to our lists
        drawn_x.append(event.xdata)
        drawn_y.append(event.ydata)
        
        # Update plot with all points
        if drawn_x:
            # Sort points by x-coordinate for proper drawing
            points = sorted(zip(drawn_x, drawn_y), key=lambda p: p[0])
            if points:
                sorted_x, sorted_y = zip(*points)
                line.set_data(sorted_x, sorted_y)
                fig.canvas.draw_idle()
    
    # Connect event handler
    fig.canvas.mpl_connect('motion_notify_event', on_move)
    
    # Show the figure and wait for user input
    plt.tight_layout()
    plt.show()
    
    # Process the drawn waveform
    if drawn_x and drawn_y:
        # Sort points by x-coordinate
        points = sorted(zip(drawn_x, drawn_y), key=lambda p: p[0])
        sorted_x, sorted_y = zip(*points)
        
        # Interpolate to get values at the required sample points
        user_wave = np.interp(t, sorted_x, sorted_y)
        
        # Normalize
        user_wave /= max(1.0, np.max(np.abs(user_wave)))
    else:
        # Use sine wave if no input provided
        user_wave = initial_wave
    
    # Populate wavetable frames with variations of user wave
    for frame in range(num_frames):
        # Apply different processing to create variation
        wave = user_wave.copy()
        
        # Apply varying spectral modifications
        spectrum = fft(wave)  # Perform Fast Fourier Transform (FFT) to convert the waveform to the frequency domain
        
        # Modify spectrum based on frame
        if frame < num_frames // 3:
            # Emphasize low frequencies in the first third of the frames
            freqs = np.linspace(0, 1, len(spectrum))  # Create a normalized frequency array (0 to 1)
            mod = np.exp(-5 * freqs * (frame / (num_frames // 3)))  # Exponential decay to emphasize low frequencies
            spectrum *= mod  # Apply the modification to the spectrum
        elif frame < 2 * (num_frames // 3):
            # Shift spectral components in the middle third of the frames
            shift = int(len(spectrum) * 0.1 * ((frame - num_frames // 3) / (num_frames // 3)))  
            # Calculate the amount to shift the spectrum, proportional to the frame's position in this range
            spectrum = np.roll(spectrum, shift)  # Circularly shift the spectrum by the calculated amount
        else:
            # Add harmonics in the last third of the frames
            for harm in range(2, 5):  # Loop through harmonic multipliers (2nd, 3rd, 4th harmonics)
                harm_shift = int(len(spectrum) / harm)  # Calculate the shift amount for the current harmonic
            spectrum = np.roll(spectrum, harm_shift)  # Circularly shift the spectrum by the harmonic shift
            spectrum[:harm_shift] = 0  # Zero out the lower frequencies to avoid aliasing artifacts
        
        # Convert back to time domain
        wave = np.real(ifft(spectrum))  # Perform Inverse FFT (IFFT) to convert the modified spectrum back to a waveform
        
        # Normalize
        if np.max(np.abs(wave)) > 0:
            wave /= np.max(np.abs(wave))
        
        wavetable[frame] = wave
    
    return wavetable

def generate_image_wavetable(image_path):
    """
    Generate a wavetable from an image file.
    The image is resized to match the wavetable dimensions.
    Luminance values determine amplitude: white=+1, black=-1.
    
    Args:
        image_path (str): Path to the image file (.jpg or .png)
        
    Returns:
        np.ndarray: The generated wavetable
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: This function requires the PIL/Pillow library.")
        print("Please install it with: pip install Pillow")
        return np.zeros((num_frames, samples_per_frame))
    
    try:
        # Load the image and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to match wavetable dimensions (width=samples_per_frame, height=num_frames)
        # Using BILINEAR filtering as specified
        img = img.resize((samples_per_frame, num_frames), Image.BILINEAR)
        
        # Convert to numpy array and normalize to [-1, 1]
        # In a grayscale image, 0 is black and 255 is white
        # We want black=-1 and white=+1, so we do: 2 * (pixel/255) - 1
        img_array = np.array(img).astype(float)
        img_array = 2 * (img_array / 255) - 1
        
        # Image array has shape (height, width), but we want (num_frames, samples_per_frame)
        # The image is already in this orientation, but we need to flip it vertically
        # because image coordinates have origin at top-left, but we want bottom-up frames
        wavetable = np.flip(img_array, axis=0)
        
        return wavetable
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return np.zeros((num_frames, samples_per_frame))

# ----------------- Menu and Control Functions -----------------

def create_formant_wavetable():
    """Create a wavetable based on vowel formant frequencies."""
    print("Available vowels: A, E, I, O, U")
    vowel_input = input("Enter vowels to include (e.g., A E O): ").strip().upper().split()
    
    selected_vowels = {}
    for vowel in vowel_input:
        if vowel in formant_data:
            selected_vowels[vowel] = formant_data[vowel]
    
    if not selected_vowels:
        print("No valid vowels selected! Using default (A, E, O)")
        selected_vowels = {"A": formant_data["A"], "E": formant_data["E"], "O": formant_data["O"]}
    
    wavetable = generate_formant_wavetable(selected_vowels)
    save_and_visualize_wavetable(wavetable, "formant_wavetable.wav")
    return wavetable

def create_harmonic_wavetable():
    """Create a wavetable with specified harmonic content."""
    try:
        num_harmonics = int(input("Enter number of harmonics (1-50): "))
        num_harmonics = max(1, min(50, num_harmonics))
        
        # Get falloff power parameter
        falloff_power = 1.0
        try:
            falloff_power = float(input("Enter harmonic falloff power (default=1.0): "))
            if falloff_power <= 0:
                print("Invalid falloff power. Using default of 1.0.")
                falloff_power = 1.0
        except ValueError:
            print("Invalid input! Using default falloff power of 1.0.")
        
        custom_amplitudes = input("Do you want to specify harmonic amplitudes? (y/n): ").lower()
        
        harmonic_amplitudes = None
        if custom_amplitudes == 'y':
            print(f"Enter {num_harmonics} amplitude values (0-1), separated by spaces:")
            amp_input = input().strip().split()
            if len(amp_input) >= num_harmonics:
                harmonic_amplitudes = [float(amp_input[i]) for i in range(num_harmonics)]
        
        wavetable = generate_harmonic_wavetable(num_harmonics, harmonic_amplitudes, falloff_power)
        save_and_visualize_wavetable(wavetable, "harmonic_wavetable.wav")
        return wavetable
    
    except ValueError:
        print("Invalid input! Using default of 10 harmonics and falloff power of 1.0.")
        wavetable = generate_harmonic_wavetable(10, None, 1.0)
        save_and_visualize_wavetable(wavetable, "harmonic_wavetable.wav")
        return wavetable

def create_additive_wavetable():
    """Create a wavetable using additive synthesis with precise control."""
    try:
        num_harmonics = int(input("Enter number of harmonics (1-20): "))
        num_harmonics = max(1, min(20, num_harmonics))
        
        print("\nDefine harmonic profiles:")
        print("You'll create profiles that the wavetable will morph between")
        
        num_profiles = int(input("How many harmonic profiles (2-5)? "))
        num_profiles = max(2, min(5, num_profiles))
        
        harmonic_profiles = []
        
        for p in range(num_profiles):
            print(f"\nProfile {p+1}:")
            profile = []
            
            # Simplified input for better usability
            print(f"Enter amplitudes for {num_harmonics} harmonics (0-1), separated by spaces:")
            amps = input().strip().split()
            
            print(f"Enter phases for {num_harmonics} harmonics (0-6.28), separated by spaces:")
            phases = input().strip().split()
            
            for h in range(num_harmonics):
                try:
                    amp = float(amps[h]) if h < len(amps) else 1.0/(h+1)
                    amp = max(0, min(1, amp))
                    
                    phase = float(phases[h]) if h < len(phases) else 0.0
                    phase = phase % (2*np.pi)
                    
                    profile.append((amp, phase))
                except (ValueError, IndexError):
                    # Default values if input is invalid
                    profile.append((1.0/(h+1), 0.0))
            
            harmonic_profiles.append(profile)
        
        # After collecting profiles, verify we have at least one valid profile
        if not harmonic_profiles:
            print("No valid profiles created. Using defaults.")
            harmonic_profiles = [
                [(1.0, 0), (0.5, 0), (0.25, 0), (0.125, 0)],  # First profile
                [(0.8, np.pi/4), (0.6, np.pi/2), (0.4, np.pi), (0.2, 3*np.pi/2)]  # Second profile
            ]
        
        wavetable = generate_additive_wavetable(harmonic_profiles)
        save_and_visualize_wavetable(wavetable, "additive_wavetable.wav")
        return wavetable
        
    except ValueError:
        print("Invalid input! Using default harmonic profiles.")
        # Create default profiles
        harmonic_profiles = [
            [(1.0, 0), (0.5, 0), (0.25, 0), (0.125, 0)],  # First profile
            [(0.8, np.pi/4), (0.6, np.pi/2), (0.4, np.pi), (0.2, 3*np.pi/2)]  # Second profile
        ]
        
        wavetable = generate_additive_wavetable(harmonic_profiles)
        save_and_visualize_wavetable(wavetable, "additive_wavetable.wav")
        return wavetable

def create_subtractive_wavetable():
    """Create a wavetable using subtractive synthesis."""
    try:
        print("Filter types: lowpass, highpass, bandpass")
        filter_type = input("Enter filter type: ").lower()
        if filter_type not in ['lowpass', 'highpass', 'bandpass']:
            filter_type = 'lowpass'
        
        start_freq = float(input("Enter starting cutoff frequency in Hz (100-10000): "))
        start_freq = max(100, min(10000, start_freq))
        
        end_freq = float(input("Enter ending cutoff frequency in Hz (100-10000): "))
        end_freq = max(100, min(10000, end_freq))
        
        wavetable = generate_subtractive_wavetable(filter_type, start_freq, end_freq)
        save_and_visualize_wavetable(wavetable, "subtractive_wavetable.wav")
        return wavetable
        
    except ValueError:
        print("Invalid input! Using default settings.")
        wavetable = generate_subtractive_wavetable()
        save_and_visualize_wavetable(wavetable, "subtractive_wavetable.wav")
        return wavetable

def create_noise_wavetable():
    """Create a wavetable with various types of noise."""
    try:
        print("Available noise types:")
        print("1. white")
        print("2. pink")
        print("3. brown")
        print("4. filtered")
        print("Enter numbers for noise types to include (e.g., 1 2 4)")
        print("Leave blank to use all types")
        
        noise_input = input("Your selection: ").strip().split()
        
        noise_types = []
        noise_options = ['white', 'pink', 'brown', 'filtered']
        
        for item in noise_input:
            try:
                idx = int(item) - 1
                if 0 <= idx < len(noise_options):
                    noise_types.append(noise_options[idx])
            except ValueError:
                # Ignore invalid inputs
                pass
        
        if not noise_types:
            print("No valid noise types selected! Using all types.")
            noise_types = ['white', 'pink', 'brown', 'filtered']
        
        wavetable = generate_noise_wavetable(noise_types)
        save_and_visualize_wavetable(wavetable, "noise_wavetable.wav")
        return wavetable
        
    except ValueError:
        print("Invalid input! Using default noise types.")
        wavetable = generate_noise_wavetable()
        save_and_visualize_wavetable(wavetable, "noise_wavetable.wav")
        return wavetable

def create_fm_wavetable():
    """Create a wavetable using FM synthesis."""
    try:
        carrier_freq = float(input("Enter carrier frequency multiplier (0.5-2.0): "))
        carrier_freq = max(0.5, min(2.0, carrier_freq))
        
        min_mod_ratio = float(input("Enter minimum modulator/carrier ratio (0.1-10.0): "))
        min_mod_ratio = max(0.1, min(10.0, min_mod_ratio))
        
        max_mod_ratio = float(input("Enter maximum modulator/carrier ratio (0.1-10.0): "))
        max_mod_ratio = max(min_mod_ratio, min(10.0, max_mod_ratio))
        
        min_mod_index = float(input("Enter minimum modulation index (0.1-20.0): "))
        min_mod_index = max(0.1, min(20.0, min_mod_index))
        
        max_mod_index = float(input("Enter maximum modulation index (0.1-20.0): "))
        max_mod_index = max(min_mod_index, min(20.0, max_mod_index))
        
        wavetable = generate_fm_wavetable(carrier_freq, 
                                         (min_mod_ratio, max_mod_ratio),
                                         (min_mod_index, max_mod_index))
        save_and_visualize_wavetable(wavetable, "fm_wavetable.wav")
        return wavetable
        
    except ValueError:
        print("Invalid input! Using default FM parameters.")
        wavetable = generate_fm_wavetable()
        save_and_visualize_wavetable(wavetable, "fm_wavetable.wav")
        return wavetable

def create_image_wavetable():
    """Create a wavetable from an image file using a file dialog."""
    print("Would you like to use a file dialog to select the image?")
    print("1. Yes")
    print("2. No")
    
    choice = input("Enter your choice (1-2, default: 1): ")
    try:
        use_dialog = int(choice) == 1
    except ValueError:
        use_dialog = True  # Default to using dialog
    
    image_path = None
    if use_dialog:
        print("Opening file dialog...")
        # Create and immediately hide a tkinter root window
        root = tk.Tk()
        root.withdraw()
        
        # Show the file dialog and get the selected file path
        image_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[
                ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"),
                ("All files", "*.*")
            ],
            initialdir=os.path.expanduser("~")  # Start in user's home directory
        )
        
        # If user closed dialog without selecting a file
        if not image_path:
            print("No file selected.")
    else:
        # Use the original text input method
        image_path = input("Enter path to image file (.jpg or .png): ")
    
    # Process the image path (whether from dialog or text input)
    if not image_path or not os.path.exists(image_path):
        print("No valid file provided. Using a generated gradient image instead.")
        
        # Create a gradient image as a fallback
        gradient = np.zeros((num_frames, samples_per_frame))
        for i in range(num_frames):
            gradient[i] = np.linspace(-1, 1, samples_per_frame) * ((i / num_frames) * 2 - 1)
        
        save_and_visualize_wavetable(gradient, "image_wavetable.wav")
        return gradient
    
    print(f"Processing image: {image_path}")
    wavetable = generate_image_wavetable(image_path)
    save_and_visualize_wavetable(wavetable, "image_wavetable.wav")
    return wavetable

def create_wavetable(wavetable_type):
    """Create a wavetable of the specified type."""
    
    if wavetable_type == 1:
        return create_formant_wavetable()
    elif wavetable_type == 2:
        return create_harmonic_wavetable()
    elif wavetable_type == 3:
        return create_additive_wavetable()
    elif wavetable_type == 4:
        return create_subtractive_wavetable()
    elif wavetable_type == 5:
        return create_noise_wavetable()
    elif wavetable_type == 6:
        return create_fm_wavetable()
    elif wavetable_type == 7:
        print("Creating wavefolding wavetable...")
        wavetable = generate_wavefolding_wavetable()
        save_and_visualize_wavetable(wavetable, "wavefolding_wavetable.wav")
        return wavetable
    elif wavetable_type == 8:
        print("Creating granular wavetable...")
        wavetable = generate_granular_wavetable()
        save_and_visualize_wavetable(wavetable, "granular_wavetable.wav")
        return wavetable
    elif wavetable_type == 9:
        print("Creating spectral morphing wavetable...")
        wavetable = generate_spectral_morphing_wavetable()
        save_and_visualize_wavetable(wavetable, "spectral_morphing_wavetable.wav")
        return wavetable
    elif wavetable_type == 10:
        print("Available chaotic equations:")
        print("1. logistic")
        print("2. henon")
        print("3. lorenz")
        
        choice = input("Enter equation type (1-3, default: 1): ")
        equation_types = ['logistic', 'henon', 'lorenz']
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(equation_types):
                eq_type = equation_types[idx]
            else:
                print("Invalid choice! Using default: logistic")
                eq_type = 'logistic'
        except ValueError:
            print("Invalid input! Using default: logistic")
            eq_type = 'logistic'
            
        wavetable = generate_chaotic_wavetable(eq_type)
        save_and_visualize_wavetable(wavetable, "chaotic_wavetable.wav")
        return wavetable
        
    elif wavetable_type == 11:
        print("Available physical models:")
        print("1. string")
        print("2. tube")
        print("3. membrane")
        
        choice = input("Enter model type (1-3, default: 1): ")
        model_types = ['string', 'tube', 'membrane']
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_types):
                model = model_types[idx]
            else:
                print("Invalid choice! Using default: string")
                model = 'string'
        except ValueError:
            print("Invalid input! Using default: string")
            model = 'string'
            
        wavetable = generate_physical_modeling_wavetable(model)
        save_and_visualize_wavetable(wavetable, "physical_modeling_wavetable.wav")
        return wavetable
        
    elif wavetable_type == 12:
        print("Creating vocal wavetable (full vowel morphing)...")
        wavetable = generate_vocal_wavetable()
        save_and_visualize_wavetable(wavetable, "vocal_wavetable.wav")
        return wavetable
    elif wavetable_type == 13:
        print("Creating wave morphing wavetable (sine → square → saw → triangle)...")
        wavetable = generate_wave_morphing_wavetable()
        save_and_visualize_wavetable(wavetable, "wave_morphing_wavetable.wav")
        return wavetable
    elif wavetable_type == 14:
        print("Available fractal types:")
        print("1. mandelbrot")
        print("2. julia") 
        print("3. ifs")
        
        choice = input("Enter fractal type (1-3, default: 1): ")
        fractal_types = ['mandelbrot', 'julia', 'ifs']
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(fractal_types):
                fractal_type = fractal_types[idx]
            else:
                print("Invalid choice! Using default: mandelbrot")
                fractal_type = 'mandelbrot'
        except ValueError:
            print("Invalid input! Using default: mandelbrot")
            fractal_type = 'mandelbrot'
            
        wavetable = generate_fractal_wavetable(fractal_type)
        save_and_visualize_wavetable(wavetable, "fractal_wavetable.wav")
        return wavetable
    elif wavetable_type == 15:
        sample_path = input("Enter path to audio file (leave blank for default): ")
        wavetable = generate_sample_based_wavetable(sample_path if sample_path else None)
        save_and_visualize_wavetable(wavetable, "sample_based_wavetable.wav")
        return wavetable
    elif wavetable_type == 16:
        print("Available waveforms:")
        print("1. sine")
        print("2. square")
        print("3. sawtooth")
        print("4. triangle")
        
        choice = input("Enter base waveform (1-4, default: 1): ")
        waveform_types = ['sine', 'square', 'sawtooth', 'triangle']
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(waveform_types):
                waveform = waveform_types[idx]
            else:
                print("Invalid choice! Using default: sine")
                waveform = 'sine'
        except ValueError:
            print("Invalid input! Using default: sine")
            waveform = 'sine'
            
        wavetable = generate_bitcrushed_wavetable((2, 16), waveform)
        save_and_visualize_wavetable(wavetable, "bitcrushed_wavetable.wav")
        return wavetable
    elif wavetable_type == 17:
        print("Opening drawing interface. Close the window when done.")
        wavetable = generate_user_drawn_wavetable()
        save_and_visualize_wavetable(wavetable, "user_drawn_wavetable.wav")
        return wavetable
    elif wavetable_type == 18:
        print("Creating image-based wavetable...")
        wavetable = create_image_wavetable()
        return wavetable
    else:
        print("Invalid choice! Defaulting to Formant Wavetable")
        return create_formant_wavetable()

def main():
    """Main function to control the program flow based on user selection."""
    while True:
        print("\n===== Advanced Wavetable Creator =====\n")
        print(" 1. Formant")
        print(" 2. Harmonic")
        print(" 3. Additive")
        print(" 4. Subtractive")
        print(" 5. Noise")
        print(" 6. FM")
        print(" 7. Wavefolding")
        print(" 8. Granular")
        print(" 9. Spectral Morphing")
        print("10. Chaotic")
        print("11. Physical Modeling")
        print("12. Vocal")
        print("13. Wave Morphing")
        print("14. Fractal")
        print("15. Sample-Based")
        print("16. Bitcrushed")
        print("17. User-Drawn")
        print("18. Image-Based")
        print("\n")
        
        try:
            choice = input("Enter your choice (1-18, or 'q' to quit): ")
            if choice.lower() == 'q':
                print("Exiting program.")
                break
                
            choice = int(choice)
            if 1 <= choice <= 18:  # Updated range to include new option
                wavetable = create_wavetable(choice)
                # Ask if user wants to create another wavetable
                if input("\nCreate another wavetable? (y/n): ").lower() != 'y':
                    break
            else:
                print("Invalid choice! Defaulting to Formant Wavetable")
                wavetable = create_wavetable(1)
        except ValueError:
            print("Invalid input! Defaulting to Formant Wavetable")
            wavetable = create_wavetable(1)

if __name__ == "__main__":
    main()
