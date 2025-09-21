import numpy as np
from scipy.io.wavfile import write
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

def generate_waveform(wave_type, frequency, num_points, sample_rate=44100):
    t = np.linspace(0, 1, num_points, endpoint=False)
    if wave_type == "sine":
        return np.sin(2 * np.pi * frequency * t)
    elif wave_type == "saw":
        return 2 * (t * frequency % 1) - 1
    elif wave_type == "square":
        return np.sign(np.sin(2 * np.pi * frequency * t))
    else:
        raise ValueError("Unsupported waveform type")

def morph_wavetables(start_wave, end_wave, num_frames):
    return np.linspace(start_wave, end_wave, num_frames)

def save_wavetable(wavetable, filename, sample_rate=44100):
    wavetable = (wavetable * 0.99).astype(np.float32)  # Normalize to avoid clipping
    write(filename, sample_rate, wavetable)

def plot_3d_wavetable(wavetable):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(wavetable.shape[1])
    y = np.arange(wavetable.shape[0])
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, wavetable, cmap='viridis')
    ax.set_title("3D Waterfall Plot of Wavetable")
    ax.set_xlabel("Sample Points")
    ax.set_ylabel("Frames")
    ax.set_zlabel("Amplitude")
    plt.show()

def main():
    num_points = int(input("Enter the number of points per frame (2048 or 1024): "))
    if num_points not in [2048, 1024]:
        print("Invalid number of points. Exiting.")
        return

    num_frames = 64
    sample_rate = 44100

    print("Choose the initial frame waveform:")
    start_wave_type = input("Enter waveform type (sine, saw, square): ").strip().lower()
    start_frequency = float(input("Enter initial frequency (e.g., 1, 3): "))
    start_wave = generate_waveform(start_wave_type, start_frequency, num_points)

    print("Choose the final frame waveform:")
    end_wave_type = input("Enter waveform type (sine, saw, square): ").strip().lower()
    end_frequency = float(input("Enter final frequency (e.g., 1, 3): "))
    end_wave = generate_waveform(end_wave_type, end_frequency, num_points)

    wavetable = morph_wavetables(start_wave, end_wave, num_frames)

    plot_3d_wavetable(wavetable)

    filename = input("Enter the filename to save the wavetable (e.g., wavetable.wav): ").strip()
    save_wavetable(wavetable.flatten(), filename, sample_rate)
    print(f"Wavetable saved as {filename}")

if __name__ == "__main__":
    main()