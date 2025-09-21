"""
RNG Comparison: Bad LCG vs Excellent Xorshift

This script compares two random number generators side-by-side:
🔴 BAD:  x[i + 1] = (5 * x[i] + 1) mod 256  (Linear Congruential Generator)
🔵 GOOD: x ^= x<<13 ^ x>>17 ^ x<<5          (Xorshift)

The visualization uses 3D polar plots to reveal the dramatic quality difference.

Key Features:
- Implements both Bad LCG and Xorshift algorithms
- Converts RNG output to 3D polar coordinates (r, theta, phi)
- Side-by-side 3D scatter plots showing distribution differences
- Side-by-side 2D XY projections revealing patterns vs uniformity
- Color-coded sequence visualization (Red=Bad, Blue=Good)
- Runs 5000 iterations each for comprehensive comparison

The comparison will clearly show:
- Bad RNG: Clustering, geometric patterns, poor space-filling
- Xorshift: Uniform distribution, no visible patterns, excellent quality
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Check if all required libraries are available
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for better Windows compatibility
except ImportError:
    print("ERROR: matplotlib is required but not installed.")
    print("Please install it using: pip install matplotlib numpy scipy")
    exit(1)

class BadRNG:
    """Simple Linear Congruential Generator with poor parameters"""
    
    def __init__(self, seed=1):
        self.x = seed % 256
        self.sequence = [self.x]
    
    def next(self):
        """Generate next value using: x[i + 1] = (5 * x[i] + 1) mod 256"""
        self.x = (5 * self.x + 1) % 256
        self.sequence.append(self.x)
        return self.x
    
    def generate_sequence(self, n):
        """Generate n random numbers"""
        values = []
        for _ in range(n):
            values.append(self.next())
        return values
    
    def reset(self, seed=1):
        """Reset the generator with a new seed"""
        self.x = seed % 256
        self.sequence = [self.x]

class XorshiftRNG:
    """Xorshift RNG - Simple but excellent quality"""
    
    def __init__(self, seed=12345):
        self.x = seed if seed != 0 else 12345  # Xorshift can't start with 0
        self.sequence = []
    
    def next(self):
        """Generate next value using Xorshift algorithm"""
        self.x ^= self.x << 13
        self.x ^= self.x >> 17
        self.x ^= self.x << 5
        self.x &= 0xFFFFFFFF  # Keep 32-bit
        value = self.x % 256  # Scale to same range as BadRNG
        self.sequence.append(value)
        return value
    
    def generate_sequence(self, n):
        """Generate n random numbers"""
        values = []
        for _ in range(n):
            values.append(self.next())
        return values
    
    def reset(self, seed=12345):
        """Reset the generator with a new seed"""
        self.x = seed if seed != 0 else 12345
        self.sequence = []

def convert_to_polar_3d(values):
    """
    Convert sequence of values to 3D polar coordinates
    Uses three consecutive values to generate (r, theta, phi)
    """
    r_values = []
    theta_values = []
    phi_values = []
    
    for i in range(0, len(values) - 2, 3):
        # Normalize values to [0, 1]
        v1, v2, v3 = values[i] / 255.0, values[i+1] / 255.0, values[i+2] / 255.0
        
        # Convert to polar coordinates
        r = v1  # Radial distance [0, 1]
        theta = v2 * 2 * np.pi  # Azimuthal angle [0, 2π]
        phi = v3 * np.pi  # Polar angle [0, π]
        
        r_values.append(r)
        theta_values.append(theta)
        phi_values.append(phi)
    
    return np.array(r_values), np.array(theta_values), np.array(phi_values)

def polar_to_cartesian(r, theta, phi):
    """Convert 3D polar to Cartesian coordinates"""
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return x, y, z



def create_rng_comparison_visualization():
    """Create side-by-side comparison of Bad RNG vs Xorshift RNG"""
    
    # Generate data from both RNGs - 5000 iterations each
    print("Generating 5000 values from Bad RNG: x[i + 1] = (5 * x[i] + 1) mod 256")
    bad_rng = BadRNG(seed=1)
    bad_values = bad_rng.generate_sequence(5000)
    
    print("Generating 5000 values from Xorshift RNG: x ^= x<<13 ^ x>>17 ^ x<<5")
    xor_rng = XorshiftRNG(seed=12345)
    xor_values = xor_rng.generate_sequence(5000)
    
    print(f"Bad RNG - Range: {min(bad_values)} to {max(bad_values)}, Mean: {np.mean(bad_values):.2f}")
    print(f"Xorshift - Range: {min(xor_values)} to {max(xor_values)}, Mean: {np.mean(xor_values):.2f}")
    
    # Convert both to 3D polar coordinates
    r_bad, theta_bad, phi_bad = convert_to_polar_3d(bad_values)
    x_bad, y_bad, z_bad = polar_to_cartesian(r_bad, theta_bad, phi_bad)
    
    r_xor, theta_xor, phi_xor = convert_to_polar_3d(xor_values)
    x_xor, y_xor, z_xor = polar_to_cartesian(r_xor, theta_xor, phi_xor)
    
    print(f"Created {len(x_bad)} Bad RNG points and {len(x_xor)} Xorshift points")
    
    # Create the visualization with 4 plots (2x2 grid)
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Bad RNG 3D scatter
    ax1 = fig.add_subplot(221, projection='3d')
    colors_bad = np.arange(len(x_bad))
    scatter1 = ax1.scatter(x_bad, y_bad, z_bad, c=colors_bad, cmap='Reds', 
                          alpha=0.7, s=20)
    ax1.set_title('BAD RNG: 3D Polar Distribution\nx[i+1] = (5*x[i] + 1) mod 256\n(Notice the patterns!)', 
                  fontsize=14, pad=20)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Xorshift 3D scatter
    ax2 = fig.add_subplot(222, projection='3d')
    colors_xor = np.arange(len(x_xor))
    scatter2 = ax2.scatter(x_xor, y_xor, z_xor, c=colors_xor, cmap='Blues', 
                          alpha=0.7, s=20)
    ax2.set_title('XORSHIFT RNG: 3D Polar Distribution\nx ^= x<<13 ^ x>>17 ^ x<<5\n(Uniform distribution!)', 
                  fontsize=14, pad=20)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Plot 3: Bad RNG 2D XY projection
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(x_bad, y_bad, c=colors_bad, cmap='Reds', alpha=0.7, s=15)
    ax3.set_title('BAD RNG: X-Y Projection\n(Geometric patterns visible)', 
                  fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Xorshift 2D XY projection
    ax4 = fig.add_subplot(224)
    scatter4 = ax4.scatter(x_xor, y_xor, c=colors_xor, cmap='Blues', alpha=0.7, s=15)
    ax4.set_title('XORSHIFT RNG: X-Y Projection\n(No visible patterns)', 
                  fontsize=14)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.grid(True, alpha=0.3)
    ax4.set_aspect('equal')
    
    # Add colorbars
    cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=15)
    cbar1.set_label('Sequence', fontsize=8)
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.5, aspect=15)
    cbar2.set_label('Sequence', fontsize=8)
    cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    cbar3.set_label('Sequence', fontsize=8)
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.8)
    cbar4.set_label('Sequence', fontsize=8)
    
    plt.tight_layout()
    
    # Add main title
    fig.suptitle('RNG Comparison: Bad LCG vs Excellent Xorshift (5000 iterations each)', 
                 fontsize=18, y=0.98)
    
    print("\nVisualization created! Compare:")
    print("🔴 BAD RNG (Red): Patterns, clustering, geometric structures")
    print("🔵 XORSHIFT (Blue): Uniform distribution, no visible patterns")
    print("- Notice how Xorshift fills space uniformly")
    print("- Bad RNG creates obvious geometric patterns")
    print("- This shows why algorithm choice matters!")
    
    return fig



if __name__ == "__main__":
    print("RNG Comparison: Bad LCG vs Excellent Xorshift")
    print("=" * 60)
    print("Comparing:")
    print("🔴 BAD:  x[i + 1] = (5 * x[i] + 1) mod 256")
    print("🔵 GOOD: x ^= x<<13 ^ x>>17 ^ x<<5")
    print("Running 5000 iterations each...")
    print()
    
    # Create comparison visualization
    main_fig = create_rng_comparison_visualization()
    
    # Show the plots
    plt.show()
    
    print("\nDramatic difference revealed!")
    print("🔴 Bad RNG: Patterns, clustering, predictable structures")
    print("🔵 Xorshift: Uniform distribution, no visible patterns")
    print()
    print("Key takeaways:")
    print("1. Algorithm choice makes a HUGE difference")
    print("2. Simple doesn't mean bad (Xorshift is very simple)")
    print("3. Visual tests can reveal RNG quality instantly")
    print("4. Bad RNGs create exploitable patterns")
    print("5. Good RNGs fill space uniformly")
    print()
    print("This shows why proper RNG design is crucial!")