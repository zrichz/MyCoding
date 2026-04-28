import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def fractional_phase_correction_v2(X, N, method='optimize'):
    """
    Improved phase correction with multiple enhancements:
    1. Uses scipy optimization for better precision
    2. Searches over a wider range (-0.5 to 0.5 samples)
    3. Optional: weighted endpoint matching
    
    Args:
        X: FFT spectrum (length N)
        N: Frame length
        method: 'optimize' (scipy) or 'brute' (original method)
    
    Returns:
        X_corr: Phase-corrected spectrum
        best_tau: Optimal fractional delay
        best_error: Residual discontinuity
    """
    k = np.arange(N)

    def apply_tau(tau):
        """Apply linear phase shift (fractional delay)"""
        return X * np.exp(-1j * 2 * np.pi * k * tau / N)
    
    def discontinuity(tau):
        """Error measure: difference between first and last sample"""
        X_corr = apply_tau(tau)
        y_corr = np.fft.ifft(X_corr).real
        return np.abs(y_corr[0] - y_corr[-1])
    
    if method == 'optimize':
        # Use scipy's optimization for higher precision
        # Search from -0.5 to 0.5 (full sample shift range)
        result = minimize_scalar(discontinuity, bounds=(-0.5, 0.5), method='bounded')
        best_tau = result.x
        best_error = result.fun
    else:
        # Original brute-force method but with finer resolution
        taus = np.linspace(-0.5, 0.5, 5000)
        errors = np.array([discontinuity(t) for t in taus])
        best_tau = taus[np.argmin(errors)]
        best_error = np.min(errors)
    
    X_corr = apply_tau(best_tau)
    return X_corr, best_tau, best_error


def endpoint_blend_correction(x, blend_samples=32):
    """
    Alternative approach: Smoothly blend the endpoints using crossfading.
    This doesn't change the spectrum much but ensures perfect looping.
    
    Args:
        x: Input waveform
        blend_samples: Number of samples to blend at each end
    
    Returns:
        x_blended: Waveform with blended endpoints
    """
    x_blended = x.copy()
    N = len(x)
    
    if blend_samples > N // 4:
        blend_samples = N // 4
    
    # Create crossfade window (linear for simplicity, could use cosine)
    fade = np.linspace(0, 1, blend_samples)
    
    # Average the start and end values to meet in the middle
    target_value = (x[0] + x[-1]) / 2.0
    
    # Blend the beginning
    x_blended[:blend_samples] = x[:blend_samples] * (1 - fade) + target_value * fade
    
    # Blend the end
    x_blended[-blend_samples:] = x[-blend_samples:] * (1 - fade[::-1]) + target_value * fade[::-1]
    
    return x_blended


def dc_removal(X):
    """
    Remove DC offset (bin 0) which can cause discontinuities.
    """
    X_clean = X.copy()
    X_clean[0] = 0
    return X_clean


def nyquist_correction(X):
    """
    For real signals, ensure Nyquist bin (if N is even) is real.
    This can help reduce artifacts.
    """
    X_clean = X.copy()
    N = len(X)
    if N % 2 == 0:
        X_clean[N//2] = np.real(X_clean[N//2])
    return X_clean


def combined_phase_correction(X, N, remove_dc=True, fix_nyquist=True, blend_after=False, blend_samples=32):
    """
    Combined approach using multiple techniques:
    1. Remove DC offset
    2. Fix Nyquist bin
    3. Apply phase correction
    4. Optional: endpoint blending in time domain
    
    Args:
        X: FFT spectrum
        N: Frame length
        remove_dc: Remove DC component
        fix_nyquist: Ensure Nyquist bin is real
        blend_after: Apply time-domain blending after phase correction
        blend_samples: Samples to blend if blend_after=True
    
    Returns:
        y_corr: Corrected waveform
        stats: Dictionary with correction statistics
    """
    X_work = X.copy()
    
    # Step 1: Clean up spectrum
    if remove_dc:
        X_work = dc_removal(X_work)
    
    if fix_nyquist:
        X_work = nyquist_correction(X_work)
    
    # Step 2: Phase correction
    X_corr, best_tau, error_after_phase = fractional_phase_correction_v2(X_work, N, method='optimize')
    y_corr = np.fft.ifft(X_corr).real
    
    # Step 3: Optional time-domain blending
    if blend_after:
        y_corr = endpoint_blend_correction(y_corr, blend_samples)
    
    # Calculate final discontinuity
    final_error = np.abs(y_corr[0] - y_corr[-1])
    
    stats = {
        'tau': best_tau,
        'error_after_phase': error_after_phase,
        'final_error': final_error,
        'dc_removed': remove_dc,
        'nyquist_fixed': fix_nyquist,
        'blend_applied': blend_after
    }
    
    return y_corr, stats


# ----- Example: Test all methods -----

N = 2048
cycles = 5.2
t = np.arange(N) / N
x = np.sin(2 * np.pi * cycles * t)

print("=" * 70)
print("ORIGINAL WAVEFORM")
print("=" * 70)
print(f"x[0] = {x[0]:.10f}")
print(f"x[-1] = {x[-1]:.10f}")
print(f"Discontinuity = {np.abs(x[0] - x[-1]):.10f}")

X = np.fft.fft(x)

# Method 1: Original brute-force
print("\n" + "=" * 70)
print("METHOD 1: Original Brute-force (0 to 1 range)")
print("=" * 70)
X_corr1, tau1, err1 = fractional_phase_correction_v2(X, N, method='brute')
y_corr1 = np.fft.ifft(X_corr1).real
print(f"Optimal tau = {tau1:.10f}")
print(f"y_corr[0] = {y_corr1[0]:.10f}")
print(f"y_corr[-1] = {y_corr1[-1]:.10f}")
print(f"Discontinuity = {err1:.10f}")

# Method 2: Scipy optimization
print("\n" + "=" * 70)
print("METHOD 2: Scipy Optimization (-0.5 to 0.5 range)")
print("=" * 70)
X_corr2, tau2, err2 = fractional_phase_correction_v2(X, N, method='optimize')
y_corr2 = np.fft.ifft(X_corr2).real
print(f"Optimal tau = {tau2:.10f}")
print(f"y_corr[0] = {y_corr2[0]:.10f}")
print(f"y_corr[-1] = {y_corr2[-1]:.10f}")
print(f"Discontinuity = {err2:.10f}")

# Method 3: Combined (DC removal + Nyquist fix + Phase correction)
print("\n" + "=" * 70)
print("METHOD 3: Combined (DC removal + Nyquist fix + Phase correction)")
print("=" * 70)
y_corr3, stats3 = combined_phase_correction(X, N, remove_dc=True, fix_nyquist=True, blend_after=False)
print(f"Optimal tau = {stats3['tau']:.10f}")
print(f"y_corr[0] = {y_corr3[0]:.10f}")
print(f"y_corr[-1] = {y_corr3[-1]:.10f}")
print(f"Discontinuity = {stats3['final_error']:.10f}")

# Method 4: Combined with endpoint blending
print("\n" + "=" * 70)
print("METHOD 4: Combined + Endpoint Blending (32 samples)")
print("=" * 70)
y_corr4, stats4 = combined_phase_correction(X, N, remove_dc=True, fix_nyquist=True, 
                                             blend_after=True, blend_samples=32)
print(f"Optimal tau = {stats4['tau']:.10f}")
print(f"y_corr[0] = {y_corr4[0]:.10f}")
print(f"y_corr[-1] = {y_corr4[-1]:.10f}")
print(f"Discontinuity = {stats4['final_error']:.10f}")

# ----- Visualization -----
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(x, linewidth=1)
axes[0, 0].plot([0, N-1], [x[0], x[-1]], 'ro', markersize=8, label=f'Endpoints (Δ={np.abs(x[0]-x[-1]):.4f})')
axes[0, 0].set_title("Original (Non-perfect Cycle)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Zoom on original
axes[0, 1].plot(x, linewidth=1)
axes[0, 1].plot([0, N-1], [x[0], x[-1]], 'ro', markersize=8)
axes[0, 1].set_xlim([-50, 50])
axes[0, 1].set_title("Original (Zoomed Start)")
axes[0, 1].grid(True, alpha=0.3)

# Method 2: Scipy optimization
axes[1, 0].plot(y_corr2, linewidth=1)
axes[1, 0].plot([0, N-1], [y_corr2[0], y_corr2[-1]], 'go', markersize=8, 
                label=f'Endpoints (Δ={err2:.6f})')
axes[1, 0].set_title(f"Method 2: Scipy Optimization (tau={tau2:.4f})")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Method 2 zoom
axes[1, 1].plot(y_corr2, linewidth=1)
axes[1, 1].plot([0, N-1], [y_corr2[0], y_corr2[-1]], 'go', markersize=8)
axes[1, 1].set_xlim([-50, 50])
axes[1, 1].set_title("Method 2 (Zoomed Start)")
axes[1, 1].grid(True, alpha=0.3)

# Method 4: Combined with blending
axes[2, 0].plot(y_corr4, linewidth=1)
axes[2, 0].plot([0, N-1], [y_corr4[0], y_corr4[-1]], 'mo', markersize=8, 
                label=f'Endpoints (Δ={stats4["final_error"]:.6f})')
axes[2, 0].set_title(f"Method 4: Combined + Blending (tau={stats4['tau']:.4f})")
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# Method 4 zoom
axes[2, 1].plot(y_corr4, linewidth=1)
axes[2, 1].plot([0, N-1], [y_corr4[0], y_corr4[-1]], 'mo', markersize=8)
axes[2, 1].set_xlim([-50, 50])
axes[2, 1].set_title("Method 4 (Zoomed Start)")
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase_correction_comparison.png', dpi=150)
print("\n" + "=" * 70)
print("Plot saved as 'phase_correction_comparison.png'")
print("=" * 70)
plt.show()
