import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

def fn_create_wavetable(sample_rate=44100, num_waveforms=64, samples_per_waveform=2048):
    # Prompt the user to choose the start and end frequencies
    print("Enter the start frequency (in Hz):")
    start_frequency = float(input("Start frequency: ").strip())
    print("Enter the end frequency (in Hz):")
    end_frequency = float(input("End frequency: ").strip())
    """
    Creates a wavetable with multiple waveforms.

    Args:
        sample_rate (int): The audio sample rate (default is 44100 Hz).
        num_waveforms (int): Number of waveforms in the wavetable (default is 64).
        samples_per_waveform (int): (default 2048).

    Returns:
        np.ndarray: The generated wavetable as a NumPy array.
    """
    # Initialize a 2D array to store all waveforms
    wavetable = np.zeros((num_waveforms, samples_per_waveform), dtype=np.float32)
    
    # Prompt the user to choose the start and end waveforms
    print("Choose the start waveform (sin, squ, saw, tri):")
    start_waveform = input("Start waveform: ").strip().lower()
    print("Choose the end waveform (sin, squ, saw, tri):")
    end_waveform = input("End waveform: ").strip().lower()

    # Define waveform generation functions
    def generate_waveform(waveform_type, t, frequency):
        if waveform_type == "sin":
            return np.sin(2 * np.pi * frequency * t)
        elif waveform_type == "squ":
            return np.sign(np.sin(2 * np.pi * frequency * t))
        elif waveform_type == "saw":
            return 2 * (t * frequency - np.floor(t * frequency + 0.5))
        elif waveform_type == "tri":
            return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1
        else:
            raise ValueError(f"Unsupported waveform type: {waveform_type}")

    for i in range(num_waveforms):
        # Generate time vector
        t = np.linspace(0, 1, samples_per_waveform, endpoint=False)
        
        # Generate start and end waveforms
        start_wave = generate_waveform(start_waveform, t, start_frequency)
        end_wave = generate_waveform(end_waveform, t, end_frequency)

        # Morph between start and end waveforms
        morph_factor = i / (num_waveforms - 1)  # Linearly interpolate between 0 and 1
        waveform = (1 - morph_factor) * start_wave + morph_factor * end_wave

        # Normalize waveform to -1 to 1
        waveform = waveform / np.max(np.abs(waveform))

        # Assign to wavetable row
        wavetable[i] = waveform
    
    # For saving as 32-bit float WAV, flatten to 1D
    wavetable_float = wavetable.flatten().astype(np.float32)
    return wavetable_float

###########################################################################
def fn_save_wavetable(filename, sample_rate, wavetable):
    write(filename, sample_rate, wavetable)
    print(f"Wavetable saved successfully as '{filename}'")

############################################################################
def fn_plot_wavetable(w, num_waveforms=64, samples_per_waveform=2048):
    # Reshape the wavetable into a 2D array (num_waveforms x samples_per_waveform)
    wavetable_2d = w.reshape((num_waveforms, samples_per_waveform))
    
    # 2D visualization
    plt.figure(figsize=(14, 8))
    plt.imshow(wavetable_2d, aspect='auto', cmap='viridis', origin='lower')
    plt.title("Wavetable Visualization (2D)")
    plt.xlabel("Sample Index")
    plt.ylabel("Waveform Index")
    plt.colorbar(label="Amplitude")
    plt.grid(False)
    plt.show()  # Ensure this is called to display the plot
    
############################################################################
def fn_plot_wavetable_3d(w, num_waveforms=64, samples_per_waveform=2048):
    """
    Plots the wavetable as a 3D line plot with a colormap and adjusted aspect ratio.

    Args:
        w (np.ndarray): The wavetable data.
        num_waveforms (int): Number of waveforms in the wavetable.
        samples_per_waveform (int): Number of samples per waveform.
    """
    # Reshape the wavetable into a 2D array (num_waveforms x samples_per_waveform)
    wavetable_2d = w.reshape((num_waveforms, samples_per_waveform))
    
    # Create a 3D figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate the X, Y, Z coordinates for the 3D plot
    x = np.arange(samples_per_waveform)  # Sample indices
    cmap = plt.get_cmap('viridis')  # Use the 'viridis' colormap
    colors = [cmap(i / num_waveforms) for i in range(num_waveforms)]  # Generate colors for each line

    for i in range(num_waveforms):
        y = np.full(samples_per_waveform, i)  # Waveform index
        z = wavetable_2d[i]  # Amplitude values
        ax.plot(x, y, z, color=colors[i], label=f"Waveform {i+1}")  # Plot each waveform in 3D

    ax.set_title("Wavetable Visualization (3D Line Plot with Colormap)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Waveform Index")
    ax.set_zlabel("Amplitude")

    # Adjust the aspect ratio by scaling the Z-axis
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, 1, 0.25, 1]))  # Reduce Z-axis prominence

    plt.show()

############################################################################
def main():
    print("\nCustom Wavetable Creator for Korg Modwave MK2\n")
    
    # Create the wavetable
    sample_rate = 44100
    wavetable = fn_create_wavetable(sample_rate=sample_rate)

    # Plot the wavetable in 2D
    fn_plot_wavetable(wavetable)

    # Plot the wavetable in 3D
    fn_plot_wavetable_3d(wavetable)

    # Save the wavetable to a file
    filename = input("\nEnter a save filename (the .wav suffix will be auto-added): ")
    # Ensure the filename ends with .wav
    if not filename.lower().endswith('.wav'):
        filename += '.wav'
    fn_save_wavetable(filename, sample_rate, wavetable)
    print("Saved wavetable successfully")

if __name__ == "__main__":
    main()
