import os
import glob
import numpy as np
import librosa
from scipy.io import wavfile
from scipy.signal import resample
from sklearn.cluster import KMeans
import concurrent.futures
import logging
import matplotlib.pyplot as plt

# For smoothing:
import scipy.ndimage

# ----- PARAMETERS -----
# This script v27 uses the Bristow-Johnson cycle extraction technique. Sounds really good and makes seamless frames.

input_directory = r"C:\Loops and Samples\Open Source Orchestral\VSCO-2-CE-1.1.0\Strings\Cello Section"
output_directory = r"C:\Loops and Samples\Modwave\Resynth Wavetables\V27 Cello Ens Test 5 harms"

target_frame_length = 2048
num_frames_list = [3, 4, 6, 8, 16, 32, 64]
trim_top_db = 30

alpha = 1.0       # Weight for Euclidean distance
beta = 0.5        # Weight for phase difference cost
num_harmonics = 5 # Number of harmonics for phase diff (values between 3-8 seem pretty good)

# Voiced probability threshold for extraction
voiced_prob_threshold = 0.5  # Lower threshold for noisy signals
skip_increment = 0.02        # 20 ms skip for unvoiced regions
fail_increment = 0.1         # 100 ms skip on extraction failure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# NaN-Filling and Smoothing Functions
# ---------------------------
def fill_nan_f0(f0_array):
    """
    Fill NaNs in f0_array by forward and backward propagation of valid values.
    If the entire array is NaN, returns the array unchanged.
    """
    valid_mask = ~np.isnan(f0_array)
    if not np.any(valid_mask):
        return f0_array  # all NaN, cannot fix
    
    # Forward fill
    for i in range(1, len(f0_array)):
        if np.isnan(f0_array[i]):
            f0_array[i] = f0_array[i - 1]
    
    # Backward fill
    for i in range(len(f0_array) - 2, -1, -1):
        if np.isnan(f0_array[i]):
            f0_array[i] = f0_array[i + 1]
    
    return f0_array

def smooth_f0(f0_array, window_size=5):
    """
    Apply a simple moving average (or median) filter to smooth pitch values.
    Here we use a 1D uniform filter from scipy.ndimage for an average.
    """
    # Make a copy so we don't overwrite original
    f0_smooth = np.copy(f0_array)
    # Apply a moving average filter of size window_size
    f0_smooth = scipy.ndimage.uniform_filter1d(f0_smooth, size=window_size)
    return f0_smooth

# ---------------------------
# New Bristow-Johnson Cycle Extraction Functions
# ---------------------------
def hann_window(beta):
    """
    Complementary Hann window defined for -1 <= beta <= 1.
    Returns a weight between 0 and 1.
    """
    return 0.5 * (1 + np.cos(np.pi * beta)) if abs(beta) < 1 else 0.0

def extract_cycle(audio, sr, f0_times, f0_values, t0_time, output_length=None):
    # 1. Determine local f0 and period at t0.
    f0_t0 = np.interp(t0_time, f0_times, f0_values)
    if np.isnan(f0_t0) or f0_t0 <= 0:
        raise ValueError("Invalid or unvoiced f0 at t0; skipping extraction.")
    tau = 1.0 / f0_t0
    tau_samples = tau * sr
    
    # 2. Integrate pitch from 0 to t0 to get total phase.
    t0_index = int(t0_time * sr)
    times = np.linspace(0, t0_time, t0_index + 1)
    f0_interp = np.interp(times, f0_times, f0_values)
    if np.isnan(f0_interp).any():
        raise ValueError("Encountered NaNs in interpolated f0 values; skipping extraction.")
    phase_curve = 2 * np.pi * np.cumsum(f0_interp / sr)
    total_phase = phase_curve[-1]
    phase_fraction = (total_phase % (2 * np.pi)) / (2 * np.pi)
    
    # 3. Compute start/end times, clamp if needed.
    start_time = t0_time - phase_fraction * tau
    end_time = start_time + tau
    if start_time < 0 or end_time > len(audio) / sr:
        raise ValueError("t0_time too close to audio boundaries for full cycle extraction.")
    
    # 4. Determine output cycle length.
    if output_length is None:
        N = int(round(tau_samples))
    else:
        N = int(output_length)
    if N <= 0:
        raise ValueError("Output length must be positive.")
    
    # 5. Vectorized extraction.
    # Create an array [0, 1, 2, ..., N-1], then fraction = k/N
    k_array = np.arange(N)
    frac = k_array / N
    
    # Beta is the fraction's offset from the local phase.
    raw_beta = frac - phase_fraction
    beta_val = raw_beta % 1.0
    
    # Hann window weights
    w1 = 0.5 * (1.0 + np.cos(np.pi * beta_val))  # same as hann_window(beta_val)
    w2 = 1.0 - w1
    
    # Times for interpolation
    t1 = t0_time + (beta_val * tau)
    t2 = t0_time + ((beta_val - 1) * tau)
    idx1 = t1 * sr
    idx2 = t2 * sr
    
    # Use np.interp on arrays of indices.
    # x-values are np.arange(len(audio)), y-values are audio
    # xp must be increasing, so the default is fine: xp=range(len(audio)).
    val1 = np.interp(idx1, np.arange(len(audio)), audio)
    val2 = np.interp(idx2, np.arange(len(audio)), audio)
    
    # Combine via crossfade
    cycle_waveform = w1 * val1 + w2 * val2
    return cycle_waveform

def extract_bj_cycles(
    y,
    sr,
    f0_times,
    f0_values,
    voiced_prob,
    output_length=2048,
    prob_threshold=0.5,
    skip_inc=0.02,
    fail_inc=0.1,
    skip_unvoiced=False
):
    """
    Walk through the audio signal and extract cycles using the Bristow-Johnson method.
    If skip_unvoiced is True, the function checks the voiced probability and skips
    regions where the probability is below prob_threshold. Otherwise, it will
    attempt to extract a cycle regardless of the voiced probability.

    Parameters:
        y             : Audio signal (1D numpy array).
        sr            : Sample rate.
        f0_times      : Array of time stamps from the pitch track.
        f0_values     : Array of pitch estimates (Hz).
        voiced_prob   : Array of voiced probabilities.
        output_length : Desired length of each extracted cycle.
        prob_threshold: Minimum voiced probability to attempt extraction if skip_unvoiced=True.
        skip_inc      : Time increment (in seconds) to skip if probability is low.
        fail_inc      : Time increment to skip if extraction fails.
        skip_unvoiced : If True, skip extraction when voiced probability < prob_threshold.
                        If False, try extracting anyway.

    Returns:
        cycles        : List of 1D numpy arrays, each containing one cycle.
    """
    cycles = []
    audio_duration = len(y) / sr
    t = 0.0

    while t < audio_duration:
        current_prob = np.interp(t, f0_times, voiced_prob)

        # Only skip unvoiced if skip_unvoiced is True.
        if skip_unvoiced and current_prob < prob_threshold:
            logger.info(
                f"Skipping at t={t:.3f}s because prob={current_prob:.3f} < threshold={prob_threshold}"
            )
            t += skip_inc
            continue

        try:
            cycle = extract_cycle(y, sr, f0_times, f0_values, t, output_length)
        except ValueError as e:
            logger.info(f"Extraction failed at t={t:.3f}s with error: {str(e)}")
            t += fail_inc
            continue

        cycles.append(cycle)

        # Advance time by one period based on the local f0.
        f0_t = np.interp(t, f0_times, f0_values)
        if f0_t > 0:
            tau = 1.0 / f0_t
        else:
            tau = 1.0 / 60  # Fallback period.
        logger.info(f"Extracted cycle at t={t:.3f}s; next step = t + {tau:.3f}s")
        t += tau

    return cycles

# ---------------------------
# Utility Functions
# ---------------------------
def bandlimit_cycle(cycle, max_bin=1024):
    X = np.fft.rfft(cycle)
    if len(X) > max_bin:
        X[max_bin:] = 0
    new_cycle = np.fft.irfft(X, n=len(cycle))
    if np.max(np.abs(new_cycle)) > 0:
        new_cycle /= np.max(np.abs(new_cycle))
    return new_cycle

def trim_end_only(y, top_db=20):
    threshold = np.max(np.abs(y)) * (10 ** (-top_db / 20))
    indices = np.nonzero(np.abs(y) > threshold)[0]
    if len(indices) == 0:
        return y
    last_index = indices[-1] + 1
    return y[:last_index]

# ---------------------------
# Load and Preprocess
# ---------------------------
def load_and_preprocess(input_filename):
    y, sr = librosa.load(input_filename, sr=None, mono=True)
    logger.info(f"Loaded {input_filename} with {len(y)} samples at {sr} Hz.")

    # Trim trailing silence.
    y = trim_end_only(y, top_db=trim_top_db)
    logger.info(f"After trimming trailing silence, {len(y)} samples remain.")

    # Normalize.
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    logger.info("Trimmed audio normalized.")

    # Create pitch track with librosa.pyin, capturing voiced probability.
    f0_array, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C0'),
        fmax=librosa.note_to_hz('C8'),
        sr=sr
    )
    if f0_array is None:
        logger.warning(f"No pitch track found in {input_filename}.")
        return None, None
    f0_array = np.array(f0_array)
    voiced_prob = np.array(voiced_prob)
    times_array = librosa.times_like(f0_array, sr=sr)

    if np.all(np.isnan(f0_array)):
        logger.warning(f"No pitch found in {input_filename}.")
        return None, None

    valid_f0 = f0_array[~np.isnan(f0_array) & (f0_array > 0)]
    if len(valid_f0) == 0:
        logger.warning(f"No valid pitch frames in {input_filename}.")
        return None, None

    global_f0 = np.median(valid_f0)
    global_period = sr / global_f0
    logger.info(f"Global pitch: ~{global_f0:.2f} Hz, global_period ~{global_period:.1f} samples")

    # Fill NaNs and smooth pitch track
    f0_array = fill_nan_f0(f0_array)
    f0_array = smooth_f0(f0_array, window_size=5)

    """
    # Debug: plot or log pitch track
    plt.figure(figsize=(10, 4))
    plt.plot(times_array, f0_array, label='f0 (Hz)', color='C0')
    plt.plot(times_array, voiced_prob, label='Voiced Probability', color='C1')
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.title("Pitch Track (Filled & Smoothed) and Voiced Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()
    """

    # Extract cycles using the new Bristow-Johnson method with voiced probability.
    cycles = extract_bj_cycles(y, sr, times_array, f0_array, voiced_prob, output_length=target_frame_length)
    if len(cycles) == 0:
        logger.warning(f"No cycles extracted from {input_filename}.")
        return None, None
    logger.info(f"Extracted {len(cycles)} cycles using Bristow-Johnson extraction.")

    # Resample cycles if needed
    resampled_cycles = []
    for cycle in cycles:
        cycle_resampled = resample(cycle, target_frame_length)
        if np.max(np.abs(cycle_resampled)) > 0:
            cycle_resampled /= np.max(np.abs(cycle_resampled))
        resampled_cycles.append(cycle_resampled)
    resampled_cycles = np.array(resampled_cycles)
    logger.info(f"Resampled cycles shape: {resampled_cycles.shape}")

    # Filter out cycles containing NaN values and log how many were rejected.
    initial_count = resampled_cycles.shape[0]
    resampled_cycles = np.array([cycle for cycle in resampled_cycles if not np.isnan(cycle).any()])
    rejected_count = initial_count - resampled_cycles.shape[0]
    logger.info(f"Rejected {rejected_count} cycles due to NaN values; {resampled_cycles.shape[0]} cycles remain.")

    return resampled_cycles, sr

# ---------------------------
# Clustering and Wavetable Creation
# ---------------------------
def compute_phase_diff(cycle1, cycle2, num_harmonics=3):
    fft1 = np.fft.rfft(cycle1)
    fft2 = np.fft.rfft(cycle2)
    angles1 = np.angle(fft1[1:num_harmonics+1])
    angles2 = np.angle(fft2[1:num_harmonics+1])
    diffs = np.abs(np.angle(np.exp(1j * (angles1 - angles2))))
    return np.sum(diffs)

def process_clustering_and_save(resampled_cycles, sr, output_filename, num_frames):
    n_clusters = min(num_frames, resampled_cycles.shape[0])
    if n_clusters < num_frames:
        logger.warning(f"Only {resampled_cycles.shape[0]} cycles available, clustering with {n_clusters} clusters (requested {num_frames} frames).")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(resampled_cycles)
    centroids = kmeans.cluster_centers_
    
    clusters_dict = {}
    for clust in range(n_clusters):
        indices = np.where(cluster_labels == clust)[0]
        if len(indices) > 0:
            clusters_dict[clust] = indices
    sorted_cluster_labels = sorted(clusters_dict.keys(), key=lambda cl: np.min(clusters_dict[cl]))
    
    selected_indices = []
    for i, cl in enumerate(sorted_cluster_labels):
        candidate_indices = clusters_dict[cl]
        centroid = centroids[cl]
        distances = np.linalg.norm(resampled_cycles[candidate_indices] - centroid, axis=1)
        if i == 0:
            best_idx = candidate_indices[np.argmin(distances)]
        else:
            prev_cycle = resampled_cycles[selected_indices[-1]]
            costs = []
            for j, idx in enumerate(candidate_indices):
                phase_cost = compute_phase_diff(resampled_cycles[idx], prev_cycle, num_harmonics)
                cost = alpha * distances[j] + beta * phase_cost
                costs.append(cost)
            best_idx = candidate_indices[np.argmin(costs)]
        selected_indices.append(best_idx)
    
    selected_indices = sorted(selected_indices)
    selected_cycles = resampled_cycles[selected_indices]
    
    if selected_cycles.shape[0] < num_frames:
        pad_count = num_frames - selected_cycles.shape[0]
        pad_cycles = np.tile(selected_cycles[-1], (pad_count, 1))
        selected_cycles = np.concatenate((selected_cycles, pad_cycles), axis=0)
    
    logger.info(f"Selected {selected_cycles.shape[0]} cycles for {num_frames} frames.")
    
    bandlimited_cycles = []
    max_bin = 1024
    for cycle in selected_cycles:
        X = np.fft.rfft(cycle)
        if len(X) > max_bin:
            X[max_bin:] = 0
        cycle_bandlimited = np.fft.irfft(X, n=len(cycle))
        if np.max(np.abs(cycle_bandlimited)) > 0:
            cycle_bandlimited /= np.max(np.abs(cycle_bandlimited))
        bandlimited_cycles.append(cycle_bandlimited)
    bandlimited_cycles = np.array(bandlimited_cycles)
    
    wavetable = np.concatenate(bandlimited_cycles[::-1])
    if np.max(np.abs(wavetable)) > 0:
        wavetable /= np.max(np.abs(wavetable))
    
    wavfile.write(output_filename, sr, wavetable.astype(np.float32))
    logger.info(f"Wavetable saved to {output_filename}")

def process_batches_and_save(resampled_cycles, sr, output_filename, num_frames):
    total_cycles = len(resampled_cycles)
    batch_size = total_cycles / num_frames
    selected_indices = []
    
    for i in range(num_frames):
        start = int(round(i * batch_size))
        end = int(round((i + 1) * batch_size))
        batch = resampled_cycles[start:end]
        logger.info(f"Batch {i}: {len(batch)} candidate cycles.")
        
        if len(batch) == 0:
            continue
        
        k = min(3, len(batch))
        kmeans = KMeans(n_clusters=k, random_state=42)
        batch_labels = kmeans.fit_predict(batch)
        centroids = kmeans.cluster_centers_
        
        clusters_dict = {}
        for cl in range(k):
            indices = np.where(batch_labels == cl)[0]
            if len(indices) > 0:
                clusters_dict[cl] = indices
                logger.info(f"  Batch {i}, Cluster {cl}: {len(indices)} candidate cycles.")
        
        best_idx_in_batch = None
        best_cost = float('inf')
        for cl in clusters_dict:
            candidate_indices = clusters_dict[cl]
            centroid = centroids[cl]
            distances = np.linalg.norm(np.array(batch)[candidate_indices] - centroid, axis=1)
            for j, idx in enumerate(candidate_indices):
                candidate = batch[idx]
                cost = distances[j]
                if i > 0 and len(selected_indices) > 0:
                    prev_cycle = resampled_cycles[selected_indices[-1]]
                    phase_cost = compute_phase_diff(candidate, prev_cycle, num_harmonics)
                    cost += beta * phase_cost
                if cost < best_cost:
                    best_cost = cost
                    best_idx_in_batch = start + idx  # Convert to global index.
        if best_idx_in_batch is not None:
            selected_indices.append(best_idx_in_batch)
    
    if len(selected_indices) < num_frames and selected_indices:
        pad_count = num_frames - len(selected_indices)
        selected_indices.extend([selected_indices[-1]] * pad_count)
    
    selected_indices = sorted(selected_indices)
    selected_cycles = resampled_cycles[selected_indices]
    logger.info(f"Selected {selected_cycles.shape[0]} cycles for {num_frames} frames using batch-based clustering.")

    bandlimited_cycles = []
    max_bin = 1024
    for cycle in selected_cycles:
        X = np.fft.rfft(cycle)
        if len(X) > max_bin:
            X[max_bin:] = 0
        cycle_bandlimited = np.fft.irfft(X, n=len(cycle))
        if np.max(np.abs(cycle_bandlimited)) > 0:
            cycle_bandlimited /= np.max(np.abs(cycle_bandlimited))
        bandlimited_cycles.append(cycle_bandlimited)
    bandlimited_cycles = np.array(bandlimited_cycles)
    
    wavetable = np.concatenate(bandlimited_cycles[::-1])
    if np.max(np.abs(wavetable)) > 0:
        wavetable /= np.max(np.abs(wavetable))
    
    wavfile.write(output_filename, sr, wavetable.astype(np.float32))
    logger.info(f"Wavetable saved to {output_filename}")

# ---------------------------
# Per-File Processing
# ---------------------------
def process_file(file_path):
    rel_path = os.path.relpath(file_path, input_directory)
    resampled_cycles, sr = load_and_preprocess(file_path)
    if resampled_cycles is None:
        return
    
    for frames in num_frames_list:
        out_dir = os.path.join(output_directory, "Batches", os.path.dirname(rel_path))
        out_dir_c = os.path.join(output_directory, "Clusters", os.path.dirname(rel_path))
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir_c, exist_ok=True)
        base = os.path.basename(file_path)
        output_path = os.path.join(out_dir, f"v27b_f{frames}_{base}")
        output_path_c = os.path.join(out_dir_c, f"v27c_f{frames}_{base}")
        process_batches_and_save(resampled_cycles, sr, output_path, num_frames=frames)
        process_clustering_and_save(resampled_cycles, sr, output_path_c, num_frames=frames)

if __name__ == "__main__":
    logger.info(f"Scanning for .wav files in {input_directory} ...")
    wav_files = glob.glob(os.path.join(input_directory, "**", "*.wav"), recursive=True)
    logger.info(f"Found {len(wav_files)} .wav files.")
    
    # Uncomment to use parallel processing (will likely overload your machine):
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     executor.map(process_file, wav_files)
    
    # For debugging, run serially:
    for f in wav_files:
        process_file(f)
    logger.info("Processing complete. Exiting.")
