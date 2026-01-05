import numpy as np
import matplotlib.pyplot as plt

def fractional_phase_correction_improved(X, N):
    """
    IMPROVEMENTS over v1:
    1. Search range expanded to -0.5 to 0.5 (instead of 0 to 1)
    2. Much finer resolution (10000 points instead of 1000)
    3. Optional: Remove DC offset before correction
    
    The key insight: fractional delays in the range [-0.5, 0.5] cover
    all possible phase shifts without redundancy.
    """
    k = np.arange(N)

    def apply_tau(tau):
        return X * np.exp(-1j * 2 * np.pi * k * tau / N)
    
    def discontinuity(tau):
        X_corr = apply_tau(tau)
        y_corr = np.fft.ifft(X_corr).real
        return np.abs(y_corr[0] - y_corr[-1])
    
    # IMPROVEMENT 1: Wider search range centered at 0
    # IMPROVEMENT 2: 10x finer resolution
    taus = np.linspace(-0.5, 0.5, 10000)
    errors = np.array([discontinuity(t) for t in taus])
    best_tau = taus[np.argmin(errors)]
    best_error = np.min(errors)
    
    X_corr = apply_tau(best_tau)
    return X_corr, best_tau, best_error


def remove_dc_component(X):
    """
    IMPROVEMENT 3: Remove DC offset (bin 0)
    DC offsets can cause endpoint discontinuities
    """
    X_clean = X.copy()
    X_clean[0] = 0
    return X_clean


def endpoint_crossfade(y, fade_samples=16):
    """
    IMPROVEMENT 4: Time-domain crossfading as a final polish
    Smoothly blend the endpoints to force them to match exactly.
    This introduces minimal distortion if done over a small region.
    """
    y_faded = y.copy()
    N = len(y)
    
    # Target value: average of endpoints
    target = (y[0] + y[-1]) / 2.0
    
    # Cosine fade window (smoother than linear)
    fade_window = (1 - np.cos(np.pi * np.arange(fade_samples) / fade_samples)) / 2
    
    # Fade start toward target
    y_faded[:fade_samples] = y[:fade_samples] * (1 - fade_window) + target * fade_window
    
    # Fade end toward target
    y_faded[-fade_samples:] = y[-fade_samples:] * (1 - fade_window[::-1]) + target * fade_window[::-1]
    
    return y_faded


def multi_iteration_correction(X, N, iterations=2):
    """
    IMPROVEMENT 5: Apply phase correction iteratively
    Sometimes one pass isn't enough - iterate to converge
    """
    X_work = X.copy()
    taus = []
    errors = []
    
    for i in range(iterations):
        X_work, tau, error = fractional_phase_correction_improved(X_work, N)
        taus.append(tau)
        errors.append(error)
        
        if i > 0 and abs(tau) < 0.001:  # Converged
            break
    
    return X_work, taus, errors


# ----- Example & Comparison -----

N = 2048
cycles = 5.2
t = np.arange(N) / N
x = np.sin(2 * np.pi * cycles * t)

print("=" * 80)
print("ORIGINAL WAVEFORM")
print("=" * 80)
print(f"x[0]          = {x[0]:.12f}")
print(f"x[-1]         = {x[-1]:.12f}")
print(f"Discontinuity = {np.abs(x[0] - x[-1]):.12f}\n")

X_original = np.fft.fft(x)

# ========== METHOD 1: Original (from v1) ==========
print("=" * 80)
print("METHOD 1: Original v1 (search 0 to 1, 1000 points)")
print("=" * 80)
k = np.arange(N)
def apply_tau_old(tau):
    return X_original * np.exp(-1j * 2 * np.pi * k * tau / N)

taus_old = np.linspace(0, 1, 1000)
errors_old = np.array([np.abs(np.fft.ifft(apply_tau_old(t)).real[0] - 
                               np.fft.ifft(apply_tau_old(t)).real[-1]) for t in taus_old])
best_tau_old = taus_old[np.argmin(errors_old)]
X_corr_old = apply_tau_old(best_tau_old)
y_corr_old = np.fft.ifft(X_corr_old).real

print(f"Best tau      = {best_tau_old:.12f}")
print(f"y[0]          = {y_corr_old[0]:.12f}")
print(f"y[-1]         = {y_corr_old[-1]:.12f}")
print(f"Discontinuity = {np.abs(y_corr_old[0] - y_corr_old[-1]):.12f}\n")

# ========== METHOD 2: Improved range and resolution ==========
print("=" * 80)
print("METHOD 2: Improved (search -0.5 to 0.5, 10000 points)")
print("=" * 80)
X_corr2, tau2, err2 = fractional_phase_correction_improved(X_original, N)
y_corr2 = np.fft.ifft(X_corr2).real

print(f"Best tau      = {tau2:.12f}")
print(f"y[0]          = {y_corr2[0]:.12f}")
print(f"y[-1]         = {y_corr2[-1]:.12f}")
print(f"Discontinuity = {err2:.12f}\n")

# ========== METHOD 3: DC removal + improved phase correction ==========
print("=" * 80)
print("METHOD 3: DC Removal + Improved Phase Correction")
print("=" * 80)
X_no_dc = remove_dc_component(X_original)
X_corr3, tau3, err3 = fractional_phase_correction_improved(X_no_dc, N)
y_corr3 = np.fft.ifft(X_corr3).real

print(f"Best tau      = {tau3:.12f}")
print(f"y[0]          = {y_corr3[0]:.12f}")
print(f"y[-1]         = {y_corr3[-1]:.12f}")
print(f"Discontinuity = {err3:.12f}\n")

# ========== METHOD 4: Multiple iterations ==========
print("=" * 80)
print("METHOD 4: Multi-iteration Correction (2 passes)")
print("=" * 80)
X_no_dc = remove_dc_component(X_original)
X_corr4, taus4, errs4 = multi_iteration_correction(X_no_dc, N, iterations=2)
y_corr4 = np.fft.ifft(X_corr4).real

print(f"Iteration 1: tau={taus4[0]:.12f}, error={errs4[0]:.12f}")
if len(taus4) > 1:
    print(f"Iteration 2: tau={taus4[1]:.12f}, error={errs4[1]:.12f}")
print(f"Final y[0]    = {y_corr4[0]:.12f}")
print(f"Final y[-1]   = {y_corr4[-1]:.12f}")
print(f"Discontinuity = {np.abs(y_corr4[0] - y_corr4[-1]):.12f}\n")

# ========== METHOD 5: Phase correction + endpoint crossfade ==========
print("=" * 80)
print("METHOD 5: Improved Phase + Endpoint Crossfade (16 samples)")
print("=" * 80)
X_no_dc = remove_dc_component(X_original)
X_corr5, tau5, err5 = fractional_phase_correction_improved(X_no_dc, N)
y_corr5_pre = np.fft.ifft(X_corr5).real
y_corr5 = endpoint_crossfade(y_corr5_pre, fade_samples=16)

print(f"Best tau      = {tau5:.12f}")
print(f"Before fade: y[0]={y_corr5_pre[0]:.12f}, y[-1]={y_corr5_pre[-1]:.12f}, Δ={err5:.12f}")
print(f"After fade:  y[0]={y_corr5[0]:.12f}, y[-1]={y_corr5[-1]:.12f}, Δ={np.abs(y_corr5[0]-y_corr5[-1]):.12f}\n")

# ========== Visualization ==========
fig = plt.figure(figsize=(16, 10))

# Create a 3x3 grid
methods = [
    ("Original", x, None),
    ("Method 1: v1 Original", y_corr_old, np.abs(y_corr_old[0] - y_corr_old[-1])),
    ("Method 2: Improved Range", y_corr2, err2),
    ("Method 3: + DC Removal", y_corr3, err3),
    ("Method 4: Multi-iteration", y_corr4, np.abs(y_corr4[0] - y_corr4[-1])),
    ("Method 5: + Crossfade", y_corr5, np.abs(y_corr5[0] - y_corr5[-1])),
]

for idx, (title, waveform, error) in enumerate(methods):
    # Full waveform
    ax1 = plt.subplot(3, 4, idx*2 + 1)
    ax1.plot(waveform, linewidth=0.8, alpha=0.8)
    ax1.plot([0, N-1], [waveform[0], waveform[-1]], 'ro', markersize=6)
    if error is not None:
        ax1.set_title(f"{title}\nΔ = {error:.8f}", fontsize=9)
    else:
        ax1.set_title(f"{title}\nΔ = {np.abs(waveform[0]-waveform[-1]):.8f}", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, N-1])
    
    # Zoomed view (first 100 samples)
    ax2 = plt.subplot(3, 4, idx*2 + 2)
    ax2.plot(waveform, linewidth=0.8, alpha=0.8)
    ax2.plot([0, N-1], [waveform[0], waveform[-1]], 'ro', markersize=6)
    ax2.set_title(f"{title} (Zoom)", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-20, 100])

plt.tight_layout()
plt.savefig('c:\\MyCoding\\WAVETABLE_SANDBOX\\phase_correction_comparison.png', dpi=150)
print("=" * 80)
print("Visualization saved: phase_correction_comparison.png")
print("=" * 80)
plt.show()
