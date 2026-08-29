#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Lenia Simulation with Taichi and OpenCV

A continuous cellular automaton that creates life-like patterns.
Features real-time parameter control via OpenCV trackbars.

Controls:
- R: Restart with new random seed
- SPACE: Pause/Resume
- ESC or Q: Quit
- Use trackbars to adjust parameters in real-time

Requirements: pip install taichi opencv-python
"""

import taichi as ti
import cv2
import numpy as np
import math

# Start Taichi and tell it to use the CPU
ti.init(arch=ti.cpu)

# Size of the grid
N = 256

# The main Lenia field (values between 0 and 1)
field = ti.field(dtype=ti.f32, shape=(N, N))

# A temporary field used for updates
next_field = ti.field(dtype=ti.f32, shape=(N, N))

# A colour image (RGB) for display
image = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

# Parameters (will be controlled by trackbars)
params = {
    'dt': 10,           # Growth rate * 100 (0.00 to 0.50)
    'mu': 15,           # Target density * 100 (0.00 to 1.00)
    'sigma': 15,        # Growth window * 1000 (0.001 to 0.100)
    'kernel_radius': 16, # Neighborhood radius (4 to 32)
    'color_scheme': 0    # Color scheme selector (0-4)
}

# Color scheme names
COLOR_SCHEMES = [
    "Rainbow",
    "Fire",
    "Ocean",
    "Monochrome",
    "Plasma"
]

@ti.func
def wrap(i, j):
    """Wrap-around so edges connect (toroidal world)"""
    return i % N, j % N

@ti.func
def kernel(r, m: ti.f32, s: ti.f32):
    """Smooth Gaussian kernel for neighbor weighting"""
    return ti.exp(-((r - m) ** 2) / (2.0 * s * s))

@ti.kernel
def init():
    """Initialize with a centered blob plus random noise"""
    for i, j in field:
        # Distance from center
        dx = (i - N // 2) / float(N)
        dy = (j - N // 2) / float(N)
        dist = ti.sqrt(dx * dx + dy * dy)
        
        # Centered blob with some random noise
        blob = ti.exp(-dist * dist * 20.0)
        noise = ti.random() * 0.1
        field[i, j] = blob + noise

@ti.kernel
def step(dt: ti.f32, mu: ti.f32, sigma: ti.f32, radius: ti.i32):
    """Update every cell in the field based on neighbors"""
    for i, j in field:
        acc = 0.0
        norm = 0.0

        # Look at neighbours in a square around the cell
        for di, dj in ti.ndrange((-radius, radius + 1), (-radius, radius + 1)):
            ni, nj = wrap(i + di, j + dj)  # wrapped neighbour position

            # Distance from centre, scaled to 0..1
            r = ti.sqrt(float(di * di + dj * dj)) / float(radius)

            # Weight from the kernel (bell-shaped)
            w = kernel(r, 0.5, 0.15)

            # Add weighted neighbour value and accumulate normalization
            acc += field[ni, nj] * w
            norm += w

        # Normalize the accumulator
        if norm > 0.0:
            acc = acc / norm

        # Growth function: cells thrive when neighbors are in sweet spot
        growth = 2.0 * ti.exp(-((acc - mu) ** 2) / (2.0 * sigma * sigma)) - 1.0

        # Apply growth to the cell with time step
        val = field[i, j] + dt * growth * field[i, j]

        # Clamp between 0 and 1
        next_field[i, j] = ti.min(ti.max(val, 0.0), 1.0)

    # Copy updated values back into the main field
    for i, j in field:
        field[i, j] = next_field[i, j]

@ti.kernel
def make_color(scheme: ti.i32):
    """Convert the field values into RGB colours based on selected scheme"""
    for i, j in field:
        v = field[i, j]
        r = 0.0
        g = 0.0
        b = 0.0
        
        # Scheme 0: Rainbow (black -> blue -> cyan -> green -> yellow -> white)
        if scheme == 0:
            if v < 0.2:
                b = v * 5.0
            elif v < 0.4:
                b = 1.0
                g = (v - 0.2) * 5.0
            elif v < 0.6:
                b = 1.0 - (v - 0.4) * 5.0
                g = 1.0
            elif v < 0.8:
                g = 1.0
                r = (v - 0.6) * 5.0
            else:
                r = 1.0
                g = 1.0
                b = (v - 0.8) * 5.0
        
        # Scheme 1: Fire (black -> red -> orange -> yellow -> white)
        elif scheme == 1:
            if v < 0.33:
                r = v * 3.0
            elif v < 0.66:
                r = 1.0
                g = (v - 0.33) * 3.0
            else:
                r = 1.0
                g = 1.0
                b = (v - 0.66) * 3.0
        
        # Scheme 2: Ocean (black -> dark blue -> cyan -> white)
        elif scheme == 2:
            if v < 0.5:
                b = v * 2.0
            else:
                b = 1.0
                r = (v - 0.5) * 2.0
                g = (v - 0.5) * 2.0
        
        # Scheme 3: Monochrome (black -> white)
        elif scheme == 3:
            r = v
            g = v
            b = v
        
        # Scheme 4: Plasma (purple -> magenta -> orange -> yellow)
        else:
            if v < 0.33:
                r = v * 1.5
                b = 0.5 + v * 1.5
            elif v < 0.66:
                r = 0.5 + (v - 0.33) * 1.5
                g = (v - 0.33) * 1.5
                b = 1.0 - (v - 0.33) * 1.5
            else:
                r = 1.0
                g = 0.5 + (v - 0.66) * 1.5
                b = 0.0

        # Clamp colours
        r = ti.min(ti.max(r, 0.0), 1.0)
        g = ti.min(ti.max(g, 0.0), 1.0)
        b = ti.min(ti.max(b, 0.0), 1.0)

        image[i, j] = ti.Vector([r, g, b])

def dummy_callback(x):
    """Dummy callback for trackbars"""
    pass

def main():
    """Main simulation loop with OpenCV interface"""
    # Initialize simulation
    init()
    
    # Create separate windows for display and controls
    display_window = "Lenia Simulation"
    control_window = "Controls"
    
    cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window, 800, 800)
    
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 500, 300)
    
    # Create trackbars in the control window
    cv2.createTrackbar("Growth Rate x100", control_window, params['dt'], 50, dummy_callback)
    cv2.createTrackbar("Target Density x100", control_window, params['mu'], 100, dummy_callback)
    cv2.createTrackbar("Growth Window x1000", control_window, params['sigma'], 100, dummy_callback)
    cv2.createTrackbar("Kernel Radius", control_window, params['kernel_radius'], 32, dummy_callback)
    cv2.createTrackbar("Color Scheme", control_window, params['color_scheme'], len(COLOR_SCHEMES) - 1, dummy_callback)
    
    paused = False
    frame_count = 0
    
    print("Lenia Simulation Started")
    print("Controls:")
    print("  R - Restart with new random seed")
    print("  SPACE - Pause/Resume")
    print("  ESC or Q - Quit")
    print("\nColor Schemes:")
    for i, name in enumerate(COLOR_SCHEMES):
        print(f"  {i}: {name}")
    print("\nAdjust parameters using trackbars for real-time control")
    print()

    # Main loop
    while True:
        # Check if windows are still open
        if cv2.getWindowProperty(display_window, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(control_window, cv2.WND_PROP_VISIBLE) < 1:
            break
        
        # Read trackbar values
        dt_val = cv2.getTrackbarPos("Growth Rate x100", control_window) / 100.0
        mu_val = cv2.getTrackbarPos("Target Density x100", control_window) / 100.0
        sigma_val = cv2.getTrackbarPos("Growth Window x1000", control_window) / 1000.0
        radius_val = max(4, cv2.getTrackbarPos("Kernel Radius", control_window))  # Min radius of 4
        scheme_val = cv2.getTrackbarPos("Color Scheme", control_window)
        
        # Ensure sigma is not zero
        if sigma_val < 0.001:
            sigma_val = 0.001
        
        # Update simulation if not paused
        if not paused:
            step(dt_val, mu_val, sigma_val, radius_val)
            frame_count += 1
        
        # Render with selected color scheme
        make_color(scheme_val)
        
        # Convert Taichi field to numpy array for OpenCV
        img_np = image.to_numpy()
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Add text overlay with current parameters
        info_text = [
            f"Frame: {frame_count}",
            f"Status: {'PAUSED' if paused else 'Running'}",
            f"Growth: {dt_val:.3f}",
            f"Density: {mu_val:.3f}",
            f"Window: {sigma_val:.4f}",
            f"Radius: {radius_val}",
            f"Scheme: {COLOR_SCHEMES[scheme_val]}"
        ]
        
        y_offset = 20
        for text in info_text:
            cv2.putText(img_bgr, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Display simulation
        cv2.imshow(display_window, img_bgr)
        
        # Create control panel info display
        control_panel = np.zeros((300, 500, 3), dtype=np.uint8)
        control_info = [
            "LENIA SIMULATION CONTROLS",
            "",
            "Keyboard Controls:",
            "  R - Restart with new random seed",
            "  SPACE - Pause/Resume",
            "  Q or ESC - Quit",
            "",
            "Use trackbars below to adjust:",
            f"  Growth Rate: {dt_val:.3f}",
            f"  Target Density: {mu_val:.3f}",
            f"  Growth Window: {sigma_val:.4f}",
            f"  Kernel Radius: {radius_val}",
            f"  Color Scheme: {COLOR_SCHEMES[scheme_val]}"
        ]
        
        y_pos = 20
        for line in control_info:
            cv2.putText(control_panel, line, (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 20
        
        cv2.imshow(control_window, control_panel)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('r'):
            print(f"Restarting simulation (Frame: {frame_count})")
            init()
            frame_count = 0
        elif key == ord(' '):
            paused = not paused
            print(f"Simulation {'paused' if paused else 'resumed'}")
    
    cv2.destroyAllWindows()
    print(f"Simulation ended after {frame_count} frames")

if __name__ == "__main__":
    main()
