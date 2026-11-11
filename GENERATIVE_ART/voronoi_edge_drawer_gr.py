#!/home/rich/MyCoding/venvMyCoding/bin/python
"""
Voronoi with Taichi and OpenCV

R: new rnd distribution of points
SPACE: Pause/Resume
C: colour scheme
M: dist metric
S: toggle pts visibility
ESC or Q to quit
opencv sliders adjust params
"""

import taichi as ti
import cv2
import numpy as np
import math

# Initialize Taichi
ti.init(arch=ti.cpu)

# grid size and num pts
N = 512
MAX_POINTS = 300

# use Taichi fields
points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_POINTS)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=MAX_POINTS)
point_colors = ti.Vector.field(3, dtype=ti.f32, shape=MAX_POINTS)
image = ti.Vector.field(3, dtype=ti.f32, shape=(N, N))

# Color scheme definitions
COLOR_SCHEMES = [
    "Colored Cells (Dark Edges)",
    "Glowing Neon Edges",
    "Monochrome Blueprint",
    "Distance Field Gradient",
    "Cyberpunk Palette",
    "Spectral Rainbow"
]

# Distance metric definitions
METRICS = [
    "Euclidean (L2)",
    "Manhattan (L1)",
    "Chebyshev (L-inf)",
    "Minkowski (L0.5)"
]

# Initial parameters
params = {
    'num_points': 35,
    'edge_thickness': 8,
    'color_scheme': 0,
    'metric': 0,
    'drift_speed': 10,
    'show_sites': 1,
    'site_size': 4,
    'distance_shading': 25
}


@ti.kernel
def init_points(num_pts: ti.i32):
    """Initialize positions, velocities, and colors"""
    for i in range(num_pts):
        points[i] = ti.Vector([ti.random(ti.f32) * 0.9 + 0.05,
                               ti.random(ti.f32) * 0.9 + 0.05])
        
        angle = ti.random(ti.f32) * 6.28318530718
        speed = 0.05 + ti.random(ti.f32) * 0.08
        velocities[i] = ti.Vector([ti.cos(angle) * speed, ti.sin(angle) * speed])
        
        # Generate vibrant cell colors using golden ratio hue distribution
        hue = (float(i) * 0.618033988749895) % 1.0
        h6 = hue * 6.0
        c = 0.85
        x = c * (1.0 - ti.abs((h6 % 2.0) - 1.0))
        
        r, g, b = 0.0, 0.0, 0.0
        if h6 < 1.0:
            r, g, b = c, x, 0.0
        elif h6 < 2.0:
            r, g, b = x, c, 0.0
        elif h6 < 3.0:
            r, g, b = 0.0, c, x
        elif h6 < 4.0:
            r, g, b = 0.0, x, c
        elif h6 < 5.0:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x
            
        point_colors[i] = ti.Vector([r + 0.15, g + 0.15, b + 0.15])


@ti.kernel
def step_points(num_pts: ti.i32, dt: ti.f32):
    """Update point positions with velocity in a toroidal world."""
    for i in range(num_pts):
        p = points[i]
        p = p + velocities[i] * dt
        points[i] = p - ti.floor(p)


@ti.func
def compute_distance(uv: ti.template(), pt: ti.template(), metric: ti.i32) -> ti.f32:
    """Compute distance between uv and pt using the chosen metric"""
    dx = ti.abs(uv[0] - pt[0])
    dy = ti.abs(uv[1] - pt[1])
    dx = ti.min(dx, 1.0 - dx)
    dy = ti.min(dy, 1.0 - dy)
    d = 0.0
    
    if metric == 0:
        # Euclidean L2
        d = ti.sqrt(dx * dx + dy * dy)
    elif metric == 1:
        # Manhattan L1
        d = dx + dy
    elif metric == 2:
        # Chebyshev L-infinity
        d = ti.max(dx, dy)
    else:
        # Minkowski L0.5
        s = ti.sqrt(dx) + ti.sqrt(dy)
        d = s * s
        
    return d


@ti.kernel
def render_voronoi(num_pts: ti.i32, edge_width: ti.f32, metric: ti.i32,
                   scheme: ti.i32, show_sites: ti.i32, site_size: ti.f32,
                   shading_intensity: ti.f32):
    """Compute closest two Voronoi sites per pixel and render edges and regions"""
    for i, j in image:
        # Normalized UV coordinates (matching row/col orientation)
        uv = ti.Vector([float(i) / float(N), float(j) / float(N)])
        
        # Track 1st and 2nd closest distances
        min_d1 = 1e9
        min_d2 = 1e9
        closest_idx = 0
        
        for k in range(num_pts):
            pt = points[k]
            d = compute_distance(uv, pt, metric)
            
            if d < min_d1:
                min_d2 = min_d1
                min_d1 = d
                closest_idx = k
            elif d < min_d2:
                min_d2 = d
                
        # Edge metric: difference between 2nd closest and 1st closest site distance
        diff = min_d2 - min_d1
        
        # Calculate edge factor (1.0 on the edge boundary, decaying to 0.0 inside cell)
        scaled_diff = diff * float(N) / ti.max(edge_width, 0.5)
        edge_factor = ti.exp(-scaled_diff * scaled_diff * 0.5)
        
        # Distance shading inside the Voronoi cell
        dist_shade = ti.min(min_d1 * shading_intensity * 3.0, 0.7)
        base_color = point_colors[closest_idx]
        
        r, g, b = 0.0, 0.0, 0.0
        
        if scheme == 0:
            # Colored cells with dark edge borders
            cell_col = base_color * (1.0 - dist_shade)
            edge_col = ti.Vector([0.05, 0.05, 0.08])
            final_c = cell_col * (1.0 - edge_factor) + edge_col * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        elif scheme == 1:
            # Glowing neon edges on dark background
            dark_bg = base_color * 0.08
            neon_edge = base_color * 1.8 + ti.Vector([0.3, 0.3, 0.3])
            final_c = dark_bg * (1.0 - edge_factor) + neon_edge * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        elif scheme == 2:
            # Monochrome Blueprint wireframe
            bg_col = ti.Vector([0.05, 0.12, 0.22])
            edge_col = ti.Vector([0.9, 0.95, 1.0])
            grid_shade = (1.0 - min_d1 * 2.0) * 0.15
            cell_bg = bg_col + ti.Vector([grid_shade, grid_shade, grid_shade])
            final_c = cell_bg * (1.0 - edge_factor) + edge_col * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        elif scheme == 3:
            # Worley distance field gradient
            v = ti.sin(min_d1 * 40.0) * 0.5 + 0.5
            grad_c = ti.Vector([v * 0.8, v * 0.5, 1.0 - v * 0.5])
            edge_col = ti.Vector([1.0, 1.0, 0.2])
            final_c = grad_c * (1.0 - edge_factor) + edge_col * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        elif scheme == 4:
            # Cyberpunk Palette
            cyan = ti.Vector([0.0, 0.9, 1.0])
            magenta = ti.Vector([1.0, 0.05, 0.6])
            dark = ti.Vector([0.04, 0.02, 0.1])
            edge_col = cyan if (closest_idx % 2 == 0) else magenta
            cell_col = dark + base_color * 0.12
            final_c = cell_col * (1.0 - edge_factor) + edge_col * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        else:
            # Spectral Rainbow
            hue_angle = float(closest_idx) / float(num_pts) * 6.2831853 + min_d1 * 5.0
            r_val = ti.sin(hue_angle) * 0.5 + 0.5
            g_val = ti.sin(hue_angle + 2.094) * 0.5 + 0.5
            b_val = ti.sin(hue_angle + 4.188) * 0.5 + 0.5
            spectral_col = ti.Vector([r_val, g_val, b_val])
            edge_col = ti.Vector([0.0, 0.0, 0.0])
            final_c = spectral_col * (1.0 - edge_factor) + edge_col * edge_factor
            r, g, b = final_c[0], final_c[1], final_c[2]
            
        # Draw seed site points if enabled
        if show_sites == 1:
            site_rad = (site_size / float(N))
            if min_d1 < site_rad:
                # White center dot with dark outline
                if min_d1 < site_rad * 0.5:
                    r, g, b = 1.0, 1.0, 1.0
                else:
                    r, g, b = 0.1, 0.1, 0.1
                    
        # Clamp colors to [0, 1]
        r = ti.min(ti.max(r, 0.0), 1.0)
        g = ti.min(ti.max(g, 0.0), 1.0)
        b = ti.min(ti.max(b, 0.0), 1.0)
        
        image[i, j] = ti.Vector([r, g, b])


def dummy_callback(x):
    """Dummy callback for OpenCV trackbars"""
    pass


def main():
    """Main interactive loop with OpenCV interface and Taichi acceleration"""
    init_points(params['num_points'])
    
    display_window = "Voronoi Edge Drawer"
    control_window = "Controls"
    
    cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(display_window, 800, 800)
    
    cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(control_window, 540, 360)
    
    # Create trackbars
    cv2.createTrackbar("Point Count", control_window, params['num_points'], MAX_POINTS, dummy_callback)
    cv2.setTrackbarMin("Point Count", control_window, 3)
    cv2.createTrackbar("Edge Thickness", control_window, params['edge_thickness'], 30, dummy_callback)
    cv2.setTrackbarMin("Edge Thickness", control_window, 1)
    cv2.createTrackbar("Color Scheme", control_window, params['color_scheme'], len(COLOR_SCHEMES) - 1, dummy_callback)
    cv2.createTrackbar("Distance Metric", control_window, params['metric'], len(METRICS) - 1, dummy_callback)
    cv2.createTrackbar("Drift Speed", control_window, params['drift_speed'], 50, dummy_callback)
    cv2.createTrackbar("Show Sites", control_window, params['show_sites'], 1, dummy_callback)
    cv2.createTrackbar("Site Size", control_window, params['site_size'], 15, dummy_callback)
    cv2.createTrackbar("Distance Shading", control_window, params['distance_shading'], 100, dummy_callback)
    
    paused = False
    frame_count = 0
    prev_num_pts = params['num_points']
    
    print("Voronoi Edge Drawer Started")
    print("Controls:")
    print("  R - Regenerate points with new random seed")
    print("  SPACE - Pause / Resume point drift animation")
    print("  C - Cycle color scheme")
    print("  M - Cycle distance metric")
    print("  S - Toggle seed sites visibility")
    print("  ESC or Q - Quit")
    print("\nColor Schemes:")
    for i, name in enumerate(COLOR_SCHEMES):
        print(f"  {i}: {name}")
    print("\nDistance Metrics:")
    for i, name in enumerate(METRICS):
        print(f"  {i}: {name}")
    print("\nUse trackbars for real-time control")
    print()
    
    while True:
        # Check window visibility
        if cv2.getWindowProperty(display_window, cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty(control_window, cv2.WND_PROP_VISIBLE) < 1:
            break
            
        # Read trackbar values
        num_pts = max(3, min(MAX_POINTS, cv2.getTrackbarPos("Point Count", control_window)))
        edge_thickness = float(max(1, cv2.getTrackbarPos("Edge Thickness", control_window)))
        scheme_val = cv2.getTrackbarPos("Color Scheme", control_window) % len(COLOR_SCHEMES)
        metric_val = cv2.getTrackbarPos("Distance Metric", control_window) % len(METRICS)
        drift_speed_val = float(cv2.getTrackbarPos("Drift Speed", control_window)) / 50.0
        show_sites_val = cv2.getTrackbarPos("Show Sites", control_window)
        site_size_val = float(max(1, cv2.getTrackbarPos("Site Size", control_window)))
        shading_val = float(cv2.getTrackbarPos("Distance Shading", control_window)) / 100.0
        
        # If point count increased, re-initialize new points
        if num_pts != prev_num_pts:
            init_points(num_pts)
            prev_num_pts = num_pts
            
        # Update point movement when unpaused
        if not paused and drift_speed_val > 0.0:
            dt = 0.016 * drift_speed_val
            step_points(num_pts, dt)
            frame_count += 1
            
        # Render Voronoi diagram and edges with Taichi
        render_voronoi(num_pts, edge_thickness, metric_val, scheme_val,
                       show_sites_val, site_size_val, shading_val)
        
        # Convert Taichi field to OpenCV BGR image
        img_np = image.to_numpy()
        # Transpose or orient properly: image is (N, N, 3) in RGB
        img_bgr = cv2.cvtColor((img_np * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        # Information overlay on display window
        info_lines = [
            f"Frame: {frame_count}",
            f"Status: {'PAUSED' if paused else 'Drifting'}",
            f"Points: {num_pts}",
            f"Metric: {METRICS[metric_val]}",
            f"Scheme: {COLOR_SCHEMES[scheme_val]}",
            f"Edge Width: {int(edge_thickness)}",
            f"Sites: {'Visible' if show_sites_val == 1 else 'Hidden'}"
        ]
        
        y_offset = 24
        for text in info_lines:
            cv2.putText(img_bgr, text, (12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_bgr, text, (12, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
            
        cv2.imshow(display_window, img_bgr)
        
        # Render control panel guide
        control_panel = np.zeros((360, 540, 3), dtype=np.uint8)
        control_info = [
            "VORONOI EDGE DRAWER CONTROLS",
            "",
            "Keyboard Shortcuts:",
            "  R     - Regenerate points",
            "  SPACE - Pause / Resume drift",
            "  C     - Cycle color scheme",
            "  M     - Cycle distance metric",
            "  S     - Toggle sites visibility",
            "  Q/ESC - Quit",
            "",
            "Current Settings:",
            f"  Points:          {num_pts}",
            f"  Metric:          {METRICS[metric_val]}",
            f"  Color Scheme:    {COLOR_SCHEMES[scheme_val]}",
            f"  Edge Thickness:  {int(edge_thickness)}",
            f"  Drift Speed:     {drift_speed_val:.2f}",
            f"  Sites Display:   {'Enabled' if show_sites_val == 1 else 'Disabled'}"
        ]
        
        y_pos = 24
        for line in control_info:
            color = (0, 220, 255) if line.startswith("VORONOI") else (200, 200, 200)
            cv2.putText(control_panel, line, (14, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            y_pos += 20
            
        cv2.imshow(control_window, control_panel)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            print(f"Regenerating points (Frame: {frame_count})")
            init_points(num_pts)
            frame_count = 0
        elif key == ord(' '):
            paused = not paused
            print(f"Simulation {'paused' if paused else 'resumed'}")
        elif key == ord('c'):
            new_scheme = (scheme_val + 1) % len(COLOR_SCHEMES)
            cv2.setTrackbarPos("Color Scheme", control_window, new_scheme)
        elif key == ord('m'):
            new_metric = (metric_val + 1) % len(METRICS)
            cv2.setTrackbarPos("Distance Metric", control_window, new_metric)
        elif key == ord('s'):
            new_sites = 0 if show_sites_val == 1 else 1
            cv2.setTrackbarPos("Show Sites", control_window, new_sites)
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
