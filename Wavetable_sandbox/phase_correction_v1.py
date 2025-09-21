import numpy as np
import matplotlib.pyplot as plt

def fractional_phase_correction(X, N):
    """
    Given the FFT spectrum X (of length N) of a frame,
    search for a fractional delay (applied as a linear phase shift)
    that minimizes the discontinuity between the first and last samples
    of the reconstructed waveform.
    """
    k = np.arange(N)

    # Define a function that applies a linear phase (fractional delay) of tau (in samples)
    def apply_tau(tau):
        # Multiply each bin k by exp(-i*2π*k*tau/N)
        return X * np.exp(-1j * 2 * np.pi * k * tau / N)
    
    # Define an error measure: difference between first and last sample after IFFT
    def discontinuity(tau):
        X_corr = apply_tau(tau)
        y_corr = np.fft.ifft(X_corr).real
        return np.abs(y_corr[0] - y_corr[-1])
    
    # Brute-force search for the optimal tau over a 1-sample span.
    taus = np.linspace(0, 1, 1000)
    errors = np.array([discontinuity(t) for t in taus])
    best_tau = taus[np.argmin(errors)]
    best_error = np.min(errors)
    # Apply the optimal phase correction.
    X_corr = apply_tau(best_tau)
    return X_corr, best_tau, best_error

# ----- Example: Constructing and Correcting a Non-Perfect Cycle -----

# The frame length for our wavetable cycle
N = 2048

# For demonstration, we create a sine waveform that should be periodic if it had an integer number of cycles.
# We deliberately use a non-integer number of cycles (e.g. 5.2 cycles in N samples) so that the endpoints don't match.
cycles = 5.2
t = np.arange(N) / N  # normalized time index from 0 to just under 1
x = np.sin(2 * np.pi * cycles * t)

# Check the old endpoints
print("Original x[0] =", x[0])
print("Original x[-1] =", x[-1])
print("Difference =", x[0] - x[-1])

# Compute the FFT of the original frame.
X = np.fft.fft(x)

# Apply the spectral (linear phase) correction to minimize the discontinuity.
X_corr, best_tau, best_error = fractional_phase_correction(X, N)
y_corr = np.fft.ifft(X_corr).real

print("\nOptimal fractional delay (tau) =", best_tau)
print("Residual discontinuity after correction =", best_error)
print("Corrected y_corr[0] =", y_corr[0])
print("Corrected y_corr[-1] =", y_corr[-1])

# ----- Visualization -----
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(x, label='Original Waveform')
plt.plot([0, N-1], [x[0], x[-1]], 'ro', label='Endpoints')
plt.title("Original 2048-Sample Frame (Non-perfect Cycle)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(y_corr, label='Phase Corrected Waveform')
plt.plot([0, N-1], [y_corr[0], y_corr[-1]], 'go', label='Endpoints')
plt.title("Phase-Corrected Frame (Optimized for Looping)")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()
