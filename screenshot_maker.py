#!/home/rich/MyCoding/venvmycoding313/bin/python
"""
Screenshot Maker - Dot Field Generator and Analyzer
Creates Poisson-distributed dot patterns for camera testing and analyzes photographed results
for focus quality, chromatic aberration, moire patterns, and color banding.
"""

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.interpolate import griddata
import io
import os
from datetime import datetime


class PoissonDiskSampler:
    """Generate evenly-distributed random points using Poisson disk sampling."""
    
    def __init__(self, width, height, min_distance, max_attempts=30):
        self.width = width
        self.height = height
        self.min_distance = min_distance
        self.max_attempts = max_attempts
        self.cell_size = min_distance / np.sqrt(2)
        self.grid_width = int(np.ceil(width / self.cell_size))
        self.grid_height = int(np.ceil(height / self.cell_size))
        self.grid = np.full((self.grid_height, self.grid_width), -1, dtype=int)
        self.points = []
        
    def generate(self):
        """Generate Poisson disk samples."""
        # Start with random point
        first_point = np.array([
            np.random.uniform(0, self.width),
            np.random.uniform(0, self.height)
        ])
        self.points.append(first_point)
        active_list = [0]
        
        gx = int(first_point[0] / self.cell_size)
        gy = int(first_point[1] / self.cell_size)
        self.grid[gy, gx] = 0
        
        while active_list:
            idx = np.random.choice(active_list)
            point = self.points[idx]
            found = False
            
            for _ in range(self.max_attempts):
                # Generate random point in annulus
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(self.min_distance, 2 * self.min_distance)
                new_point = point + radius * np.array([np.cos(angle), np.sin(angle)])
                
                if not (0 <= new_point[0] < self.width and 0 <= new_point[1] < self.height):
                    continue
                
                gx = int(new_point[0] / self.cell_size)
                gy = int(new_point[1] / self.cell_size)
                
                # Check neighborhood for conflicts
                if self._is_valid(new_point, gx, gy):
                    self.points.append(new_point)
                    self.grid[gy, gx] = len(self.points) - 1
                    active_list.append(len(self.points) - 1)
                    found = True
                    break
            
            if not found:
                active_list.remove(idx)
        
        return np.array(self.points)
    
    def _is_valid(self, point, gx, gy):
        """Check if point is valid (no neighbors within min_distance)."""
        search_radius = 2
        x_min = max(0, gx - search_radius)
        x_max = min(self.grid_width, gx + search_radius + 1)
        y_min = max(0, gy - search_radius)
        y_max = min(self.grid_height, gy + search_radius + 1)
        
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                neighbor_idx = self.grid[y, x]
                if neighbor_idx != -1:
                    neighbor = self.points[neighbor_idx]
                    if np.linalg.norm(point - neighbor) < self.min_distance:
                        return False
        return True


def generate_dot_image(density, dot_size, progress=gr.Progress()):
    """Generate 2200x1200 white image with Poisson-distributed black dots."""
    width, height = 2200, 1200
    
    # Map density slider (1-100) to min_distance
    # Higher density = smaller min_distance = more dots
    min_distance = 200 / density  # Range from 200 (sparse) to 2 (dense)
    
    progress(0.1, desc="Generating Poisson disk samples...")
    sampler = PoissonDiskSampler(width, height, min_distance)
    points = sampler.generate()
    
    progress(0.5, desc=f"Generated {len(points)} points, drawing dots...")
    
    # Create white image
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw black dots
    radius = dot_size
    for i, point in enumerate(points):
        x, y = point
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill='black')
        
        if i % 100 == 0:
            progress(0.5 + 0.4 * (i / len(points)), desc=f"Drawing dot {i}/{len(points)}")
    
    progress(1.0, desc="Complete")
    
    stats = f"Generated {len(points)} dots\nDensity setting: {density}\nMin distance: {min_distance:.1f}px\nDot radius: {dot_size}px"
    
    return img, stats


def detect_dots(image):
    """Detect dark dots in the image and return their positions."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Invert (dots should be bright on dark background for blob detection)
    inverted = 255 - gray
    
    # Threshold to isolate dots
    _, binary = cv2.threshold(inverted, 100, 255, cv2.THRESH_BINARY)
    
    # Set up blob detector for circular dots
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.3
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary)
    
    return keypoints, gray


def analyze_focus_quality(image, progress=gr.Progress()):
    """Analyze focus quality across the 2D image field using edge sharpness."""
    if image is None:
        return None, "No image uploaded"
    
    progress(0.1, desc="Converting image...")
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    progress(0.2, desc="Detecting dots...")
    keypoints, gray = detect_dots(image)
    
    if len(keypoints) == 0:
        return None, "No dots detected in image"
    
    progress(0.4, desc=f"Analyzing {len(keypoints)} dots for focus quality...")
    
    height, width = gray.shape
    
    # Create focus map using local edge strength
    grid_size = 50
    focus_map = np.zeros((height // grid_size + 1, width // grid_size + 1))
    count_map = np.zeros_like(focus_map)
    
    # Calculate Laplacian variance (focus measure) around each dot
    focus_scores = []
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Extract region around dot
        margin = 20
        x1 = max(0, x - margin)
        x2 = min(width, x + margin)
        y1 = max(0, y - margin)
        y2 = min(height, y + margin)
        
        region = gray[y1:y2, x1:x2]
        
        if region.size > 0:
            # Laplacian variance as focus measure
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            focus_score = laplacian.var()
            focus_scores.append(focus_score)
            
            # Add to grid
            grid_x = x // grid_size
            grid_y = y // grid_size
            focus_map[grid_y, grid_x] += focus_score
            count_map[grid_y, grid_x] += 1
        
        if i % 50 == 0:
            progress(0.4 + 0.3 * (i / len(keypoints)), desc=f"Analyzing dot {i}/{len(keypoints)}")
    
    progress(0.8, desc="Creating focus heatmap...")
    
    # Average focus scores per grid cell
    with np.errstate(divide='ignore', invalid='ignore'):
        focus_map = np.where(count_map > 0, focus_map / count_map, 0)
    
    # Upsample and smooth for visualization
    focus_map_upsampled = cv2.resize(focus_map, (width, height), interpolation=cv2.INTER_LINEAR)
    focus_map_smooth = gaussian_filter(focus_map_upsampled, sigma=30)
    
    # Normalize to 0-1 range
    if focus_map_smooth.max() > 0:
        focus_map_normalized = (focus_map_smooth - focus_map_smooth.min()) / (focus_map_smooth.max() - focus_map_smooth.min())
    else:
        focus_map_normalized = focus_map_smooth
    
    # Create colored heatmap overlay
    heatmap_colored = cv2.applyColorMap((focus_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original
    overlay = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
    
    progress(1.0, desc="Complete")
    
    # Save focus map for later use
    output_dir = "screenshot_maker_outputs"
    os.makedirs(output_dir, exist_ok=True)
    focus_map_file = os.path.join(output_dir, "focus_map.npy")
    np.save(focus_map_file, focus_map_smooth)
    
    # Statistics
    focus_scores = np.array(focus_scores)
    stats = (f"Detected {len(keypoints)} dots\n"
             f"Focus score range: {focus_scores.min():.1f} to {focus_scores.max():.1f}\n"
             f"Mean focus: {focus_scores.mean():.1f}\n"
             f"Std deviation: {focus_scores.std():.1f}\n\n"
             f"Heatmap: Red = Best focus, Blue = Poor focus\n"
             f"Focus map saved to {focus_map_file}")
    
    return Image.fromarray(overlay), stats


def analyze_chromatic_aberration(image, progress=gr.Progress()):
    """Analyze chromatic aberration by measuring RGB centroid shifts of dots."""
    if image is None:
        return None, "No image uploaded"
    
    progress(0.1, desc="Converting image...")
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    progress(0.2, desc="Detecting dots...")
    keypoints, gray = detect_dots(image)
    
    if len(keypoints) == 0:
        return None, "No dots detected in image"
    
    progress(0.4, desc=f"Analyzing {len(keypoints)} dots for chromatic aberration...")
    
    height, width = image.shape[:2]
    
    # Extract RGB channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    ca_vectors = []
    positions = []
    
    # Calculate RGB centroid shifts for each dot
    for i, kp in enumerate(keypoints):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        
        # Extract region around dot
        margin = 15
        x1 = max(0, x - margin)
        x2 = min(width, x + margin)
        y1 = max(0, y - margin)
        y2 = min(height, y + margin)
        
        r_region = r_channel[y1:y2, x1:x2]
        g_region = g_channel[y1:y2, x1:x2]
        b_region = b_channel[y1:y2, x1:x2]
        
        if r_region.size > 0:
            # Invert regions (dots are dark)
            r_inv = 255 - r_region
            g_inv = 255 - g_region
            b_inv = 255 - b_region
            
            # Calculate centroids
            def centroid(region):
                total = region.sum()
                if total == 0:
                    return 0, 0
                y_coords, x_coords = np.indices(region.shape)
                cx = (x_coords * region).sum() / total
                cy = (y_coords * region).sum() / total
                return cx, cy
            
            r_cx, r_cy = centroid(r_inv)
            g_cx, g_cy = centroid(g_inv)
            b_cx, b_cy = centroid(b_inv)
            
            # Calculate centroid shifts relative to green (reference)
            r_shift = np.array([r_cx - g_cx, r_cy - g_cy])
            b_shift = np.array([b_cx - g_cx, b_cy - g_cy])
            
            # Total CA magnitude
            ca_magnitude = np.linalg.norm(r_shift) + np.linalg.norm(b_shift)
            
            ca_vectors.append({
                'position': (x, y),
                'r_shift': r_shift,
                'b_shift': b_shift,
                'magnitude': ca_magnitude
            })
            positions.append([x, y])
        
        if i % 50 == 0:
            progress(0.4 + 0.4 * (i / len(keypoints)), desc=f"Analyzing dot {i}/{len(keypoints)}")
    
    progress(0.9, desc="Creating visualization...")
    
    # Create visualization
    vis = image.copy()
    
    # Draw CA vectors (exaggerated for visibility)
    scale = 10
    for ca in ca_vectors:
        x, y = ca['position']
        
        # Red channel shift
        r_end_x = int(x + ca['r_shift'][0] * scale)
        r_end_y = int(y + ca['r_shift'][1] * scale)
        cv2.arrowedLine(vis, (x, y), (r_end_x, r_end_y), (255, 0, 0), 1, tipLength=0.3)
        
        # Blue channel shift
        b_end_x = int(x + ca['b_shift'][0] * scale)
        b_end_y = int(y + ca['b_shift'][1] * scale)
        cv2.arrowedLine(vis, (x, y), (b_end_x, b_end_y), (0, 0, 255), 1, tipLength=0.3)
    
    progress(1.0, desc="Complete")
    
    # Save CA data for later use
    output_dir = "screenshot_maker_outputs"
    os.makedirs(output_dir, exist_ok=True)
    ca_data_file = os.path.join(output_dir, "ca_vectors.npy")
    np.save(ca_data_file, ca_vectors, allow_pickle=True)
    
    # Statistics
    magnitudes = [ca['magnitude'] for ca in ca_vectors]
    if magnitudes:
        stats = (f"Analyzed {len(ca_vectors)} dots\n"
                 f"CA magnitude range: {min(magnitudes):.2f} to {max(magnitudes):.2f} pixels\n"
                 f"Mean CA: {np.mean(magnitudes):.2f} pixels\n"
                 f"Std deviation: {np.std(magnitudes):.2f} pixels\n\n"
                 f"Red arrows: Red channel centroid shift\n"
                 f"Blue arrows: Blue channel centroid shift\n"
                 f"(Arrows scaled 10x for visibility)\n"
                 f"CA data saved to {ca_data_file}")
    else:
        stats = "No chromatic aberration data available"
    
    return Image.fromarray(vis), stats


def extract_moire_and_banding(image, moire_threshold, banding_sigma, progress=gr.Progress()):
    """Extract moire patterns and color banding from a photo of a white screen."""
    if image is None:
        return None, None, None, "No image uploaded"
    
    progress(0.1, desc="Converting image...")
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    height, width = image.shape[:2]
    
    progress(0.2, desc="Extracting color channels...")
    
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Calculate expected uniform white (mean of entire image)
    mean_color = img_float.mean(axis=(0, 1))
    
    progress(0.3, desc="Separating frequency components...")
    
    # Extract low-frequency component (banding) - smooth color variations
    banding = np.zeros_like(img_float)
    for c in range(3):
        banding[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=banding_sigma)
    
    # Calculate deviation from expected uniform color
    banding_deviation = banding - mean_color
    
    progress(0.5, desc="Extracting high-frequency moire patterns...")
    
    # Extract high-frequency component (moire) - remove the low-frequency banding
    high_freq = img_float - banding
    
    # Threshold to isolate significant moire patterns
    moire_patterns = np.where(np.abs(high_freq) > moire_threshold, high_freq, 0)
    
    progress(0.7, desc="Creating visualizations...")
    
    # Create visualization images
    # 1. Banding visualization (exaggerated)
    banding_vis = banding_deviation * 5  # Amplify for visibility
    banding_vis = np.clip(banding_vis + 127, 0, 255).astype(np.uint8)
    
    # 2. Moire visualization (exaggerated)
    moire_vis = moire_patterns * 10  # Amplify for visibility
    moire_vis = np.clip(moire_vis + 127, 0, 255).astype(np.uint8)
    
    # 3. Combined visualization
    combined = banding_deviation + moire_patterns
    combined_vis = combined * 5
    combined_vis = np.clip(combined_vis + 127, 0, 255).astype(np.uint8)
    
    progress(0.9, desc="Calculating statistics...")
    
    # Statistics
    banding_magnitude = np.linalg.norm(banding_deviation, axis=2)
    moire_magnitude = np.linalg.norm(moire_patterns, axis=2)
    
    stats = (f"Image dimensions: {width}x{height}\n"
             f"Mean screen color: R={mean_color[0]:.1f}, G={mean_color[1]:.1f}, B={mean_color[2]:.1f}\n\n"
             f"BANDING (low frequency):\n"
             f"  Max deviation: {banding_magnitude.max():.2f}\n"
             f"  Mean deviation: {banding_magnitude.mean():.2f}\n"
             f"  Std deviation: {banding_magnitude.std():.2f}\n\n"
             f"MOIRE (high frequency):\n"
             f"  Max magnitude: {moire_magnitude.max():.2f}\n"
             f"  Mean magnitude: {moire_magnitude[moire_magnitude > 0].mean() if (moire_magnitude > 0).any() else 0:.2f}\n"
             f"  Affected pixels: {(moire_magnitude > 0).sum()} ({100 * (moire_magnitude > 0).sum() / (width * height):.2f}%)\n\n"
             f"These patterns can be saved and applied to other images as overlay masks.")
    
    progress(1.0, desc="Complete")
    
    return (Image.fromarray(banding_vis), 
            Image.fromarray(moire_vis), 
            Image.fromarray(combined_vis), 
            stats)


def save_moire_banding_data(image, moire_threshold, banding_sigma):
    """Save the extracted moire and banding data as numpy arrays for later use."""
    if image is None:
        return "No image to process"
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to float for processing
    img_float = image.astype(np.float32)
    
    # Calculate expected uniform white
    mean_color = img_float.mean(axis=(0, 1))
    
    # Extract banding
    banding = np.zeros_like(img_float)
    for c in range(3):
        banding[:, :, c] = gaussian_filter(img_float[:, :, c], sigma=banding_sigma)
    
    banding_deviation = banding - mean_color
    
    # Extract moire
    high_freq = img_float - banding
    moire_patterns = np.where(np.abs(high_freq) > moire_threshold, high_freq, 0)
    
    # Save to files
    output_dir = "screenshot_maker_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    banding_file = os.path.join(output_dir, "banding_map.npy")
    moire_file = os.path.join(output_dir, "moire_map.npy")
    combined_file = os.path.join(output_dir, "combined_artifacts.npy")
    
    np.save(banding_file, banding_deviation)
    np.save(moire_file, moire_patterns)
    np.save(combined_file, banding_deviation + moire_patterns)
    
    return (f"Saved artifact maps to {output_dir}/\n"
            f"  - banding_map.npy ({banding_deviation.shape})\n"
            f"  - moire_map.npy ({moire_patterns.shape})\n"
            f"  - combined_artifacts.npy\n\n"
            f"These can be loaded and applied to other images as:\n"
            f"  artifacts = np.load('banding_map.npy')\n"
            f"  processed = original_image + artifacts * strength")


def load_artifact_data():
    """Load saved artifact data if available."""
    output_dir = "screenshot_maker_outputs"
    
    banding_file = os.path.join(output_dir, "banding_map.npy")
    moire_file = os.path.join(output_dir, "moire_map.npy")
    focus_file = os.path.join(output_dir, "focus_map.npy")
    ca_file = os.path.join(output_dir, "ca_vectors.npy")
    
    data = {}
    if os.path.exists(banding_file):
        data['banding'] = np.load(banding_file)
    if os.path.exists(moire_file):
        data['moire'] = np.load(moire_file)
    if os.path.exists(focus_file):
        data['focus_map'] = np.load(focus_file)
    if os.path.exists(ca_file):
        data['ca_vectors'] = np.load(ca_file, allow_pickle=True)
    
    return data


def resize_map_to_image(artifact_map, target_shape):
    """Resize artifact map to match target image dimensions using interpolation."""
    src_height, src_width = artifact_map.shape[:2]
    tgt_height, tgt_width = target_shape[:2]
    
    # Create interpolators for each channel
    if len(artifact_map.shape) == 3:
        resized = np.zeros((tgt_height, tgt_width, artifact_map.shape[2]), dtype=artifact_map.dtype)
        
        for c in range(artifact_map.shape[2]):
            resized[:, :, c] = cv2.resize(artifact_map[:, :, c], (tgt_width, tgt_height), 
                                          interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(artifact_map, (tgt_width, tgt_height), 
                            interpolation=cv2.INTER_LINEAR)
    
    return resized


def apply_focus_blur(image, focus_map, blur_strength):
    """Apply spatially-varying blur based on focus map."""
    if blur_strength == 0:
        return image
    
    # Normalize focus map to 0-1 (inverse: low focus = more blur)
    focus_normalized = focus_map.copy()
    if focus_normalized.max() > 0:
        focus_normalized = 1.0 - (focus_normalized - focus_normalized.min()) / (focus_normalized.max() - focus_normalized.min())
    
    # Create multiple blur levels
    blur_levels = []
    for kernel_size in [3, 7, 15, 31]:
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        blur_levels.append(blurred)
    
    # Blend based on focus map
    result = image.copy().astype(np.float32)
    
    for i, blurred in enumerate(blur_levels):
        # Create weight map for this blur level
        blur_weight = np.clip((focus_normalized - i * 0.25) * 4, 0, 1)
        blur_weight = np.expand_dims(blur_weight, axis=2)
        
        # Blend
        weight = blur_weight * blur_strength * (i + 1) * 0.25
        result = result * (1 - weight) + blurred.astype(np.float32) * weight
    
    return result.astype(np.uint8)


def apply_chromatic_aberration(image, ca_vectors, ca_strength):
    """Apply chromatic aberration using analyzed RGB channel shifts from dot field."""
    if ca_strength == 0:
        return image
    
    height, width = image.shape[:2]
    
    # If we have actual CA vector data from analysis, use it
    if ca_vectors is not None and len(ca_vectors) > 0:
        # Create interpolated shift maps from analyzed dot positions
        from scipy.interpolate import griddata
        
        # Extract positions and shifts
        positions = np.array([ca['position'] for ca in ca_vectors])
        r_shifts_x = np.array([ca['r_shift'][0] for ca in ca_vectors])
        r_shifts_y = np.array([ca['r_shift'][1] for ca in ca_vectors])
        b_shifts_x = np.array([ca['b_shift'][0] for ca in ca_vectors])
        b_shifts_y = np.array([ca['b_shift'][1] for ca in ca_vectors])
        
        # Create grid for interpolation
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        grid_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        
        # Interpolate shift values across entire image
        r_shift_x_map = griddata(positions, r_shifts_x, (x_grid, y_grid), method='linear', fill_value=0)
        r_shift_y_map = griddata(positions, r_shifts_y, (x_grid, y_grid), method='linear', fill_value=0)
        b_shift_x_map = griddata(positions, b_shifts_x, (x_grid, y_grid), method='linear', fill_value=0)
        b_shift_y_map = griddata(positions, b_shifts_y, (x_grid, y_grid), method='linear', fill_value=0)
        
        # Apply strength multiplier
        r_shift_x_map *= ca_strength
        r_shift_y_map *= ca_strength
        b_shift_x_map *= ca_strength
        b_shift_y_map *= ca_strength
        
    else:
        # Fallback to radial CA if no analyzed data
        center_x, center_y = width / 2, height / 2
        y_grid, x_grid = np.mgrid[0:height, 0:width]
        dx = (x_grid - center_x) / center_x
        dy = (y_grid - center_y) / center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        r_shift_x_map = dx * distance * ca_strength * 2
        r_shift_y_map = dy * distance * ca_strength * 2
        b_shift_x_map = -dx * distance * ca_strength * 2
        b_shift_y_map = -dy * distance * ca_strength * 2
    
    # Shift channels
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Create shifted coordinates
    r_x = np.clip(x_coords + r_shift_x_map, 0, width - 1).astype(np.float32)
    r_y = np.clip(y_coords + r_shift_y_map, 0, height - 1).astype(np.float32)
    
    b_x = np.clip(x_coords + b_shift_x_map, 0, width - 1).astype(np.float32)
    b_y = np.clip(y_coords + b_shift_y_map, 0, height - 1).astype(np.float32)
    
    # Remap channels
    r_shifted = cv2.remap(r_channel, r_x, r_y, cv2.INTER_LINEAR)
    b_shifted = cv2.remap(b_channel, b_x, b_y, cv2.INTER_LINEAR)
    
    result = np.stack([r_shifted, g_channel, b_shifted], axis=2)
    
    return result


def apply_screen_effects(image, blur_strength, ca_strength, banding_strength, moire_strength, 
                        progress=gr.Progress()):
    """Apply all captured screen effects to an image."""
    if image is None:
        return None, "No image uploaded"
    
    progress(0.05, desc="Loading image...")
    
    # Convert PIL to numpy if needed
    if isinstance(image, Image.Image):
        original_image = np.array(image)
    else:
        original_image = image.copy()
    
    height, width = original_image.shape[:2]
    result = original_image.astype(np.float32)
    
    # Load artifact data
    progress(0.1, desc="Loading artifact data...")
    artifact_data = load_artifact_data()
    
    effects_applied = []
    
    # Apply focus blur using analyzed data
    if blur_strength > 0:
        progress(0.2, desc="Applying focus blur...")
        if 'focus_map' in artifact_data:
            # Use analyzed focus map, resize to target image
            focus_map = resize_map_to_image(artifact_data['focus_map'], (height, width))
            effects_applied.append(f"Focus blur (analyzed): {blur_strength:.2f}")
        else:
            # Fallback to synthetic radial focus map
            center_x, center_y = width / 2, height / 2
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            dx = (x_coords - center_x) / center_x
            dy = (y_coords - center_y) / center_y
            distance = np.sqrt(dx**2 + dy**2)
            focus_map = distance
            effects_applied.append(f"Focus blur (synthetic): {blur_strength:.2f}")
        
        result = apply_focus_blur(result.astype(np.uint8), focus_map, blur_strength)
        result = result.astype(np.float32)
    
    # Apply chromatic aberration using analyzed data
    if ca_strength > 0:
        progress(0.4, desc="Applying chromatic aberration...")
        ca_vectors = artifact_data.get('ca_vectors', None)
        result = apply_chromatic_aberration(result.astype(np.uint8), ca_vectors, ca_strength)
        result = result.astype(np.float32)
        if ca_vectors is not None:
            effects_applied.append(f"Chromatic aberration (analyzed): {ca_strength:.2f}")
        else:
            effects_applied.append(f"Chromatic aberration (synthetic): {ca_strength:.2f}")
    
    # Apply banding
    if banding_strength > 0 and 'banding' in artifact_data:
        progress(0.6, desc="Applying color banding...")
        banding_map = artifact_data['banding']
        
        # Resize to match image
        banding_resized = resize_map_to_image(banding_map, (height, width))
        
        # Apply as additive
        result = result + banding_resized * banding_strength
        effects_applied.append(f"Color banding: {banding_strength:.2f}")
    elif banding_strength > 0:
        effects_applied.append("Banding: No data (run tab 3 first)")
    
    # Apply moire
    if moire_strength > 0 and 'moire' in artifact_data:
        progress(0.8, desc="Applying moire patterns...")
        moire_map = artifact_data['moire']
        
        # Resize to match image
        moire_resized = resize_map_to_image(moire_map, (height, width))
        
        # Apply as additive
        result = result + moire_resized * moire_strength
        effects_applied.append(f"Moire patterns: {moire_strength:.2f}")
    elif moire_strength > 0:
        effects_applied.append("Moire: No data (run tab 3 first)")
    
    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    progress(0.95, desc="Creating output...")
    
    # Statistics
    stats = (f"Original image: {width}x{height}\n"
             f"Effects applied:\n" + "\n".join(f"  - {e}" for e in effects_applied) + "\n\n"
             f"Output format: PNG\n"
             f"Ready to save")
    
    progress(1.0, desc="Complete")
    
    return Image.fromarray(result), stats


def save_processed_image(image):
    """Save the processed image as PNG."""
    if image is None:
        return "No image to save"
    
    output_dir = "screenshot_maker_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"processed_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure we save as PNG
    if isinstance(image, Image.Image):
        image.save(filepath, "PNG")
    else:
        Image.fromarray(image).save(filepath, "PNG")
    
    return f"Saved to {filepath}"


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Screenshot Maker - Dot Field Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Screenshot Maker - Dot Field Generator and Analyzer")
        gr.Markdown("Create calibration dot patterns and analyze photographed images for focus, chromatic aberration, moire, and banding")
        
        with gr.Tab("1. dot_image_creator"):
            gr.Markdown("### Generate Poisson-Distributed Dot Pattern")
            gr.Markdown("Creates a 2200x1200 white image with evenly-distributed black dots")
            
            with gr.Row():
                with gr.Column(scale=1):
                    density_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                        label="Dot Density",
                        info="Higher values = more dots (1=sparse, 100=dense)"
                    )
                    
                    dot_size_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Dot Size (radius in pixels)",
                        info="Size of each black dot"
                    )
                    
                    generate_btn = gr.Button("Generate Dot Pattern", variant="primary", size="lg")
                    
                    stats_output = gr.Textbox(
                        label="Generation Statistics",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    dot_image_output = gr.Image(
                        label="Generated Dot Pattern (2200x1200)",
                        type="pil",
                        height=600
                    )
        
        with gr.Tab("2. analysis"):
            gr.Markdown("### Analyze Photographed Dot Field")
            gr.Markdown("Upload a photo of the dot pattern to analyze focus quality and chromatic aberration. **Analysis data is automatically saved for use in Tab 4.**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    upload_image = gr.Image(
                        label="Upload Photo of Dot Field",
                        type="pil",
                        height=400
                    )
                    
                    with gr.Row():
                        analyze_focus_btn = gr.Button("Analyze Focus", variant="primary")
                        analyze_ca_btn = gr.Button("Analyze Chromatic Aberration", variant="primary")
                
                with gr.Column(scale=2):
                    with gr.Tab("Focus Analysis"):
                        focus_output = gr.Image(
                            label="Focus Quality Heatmap",
                            type="pil",
                            height=500
                        )
                        focus_stats = gr.Textbox(
                            label="Focus Statistics",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Tab("Chromatic Aberration"):
                        ca_output = gr.Image(
                            label="Chromatic Aberration Visualization",
                            type="pil",
                            height=500
                        )
                        ca_stats = gr.Textbox(
                            label="CA Statistics",
                            lines=10,
                            interactive=False
                        )
        
        with gr.Tab("3. moire_banding_capture"):
            gr.Markdown("### Capture Moire Patterns and Color Banding")
            gr.Markdown("Upload a photo of a pure white screen to extract actual sensor artifacts for later use as overlays")
            
            with gr.Row():
                with gr.Column(scale=1):
                    white_screen_image = gr.Image(
                        label="Upload Photo of White Screen",
                        type="pil",
                        height=400
                    )
                    
                    gr.Markdown("#### Extraction Parameters")
                    
                    moire_threshold_slider = gr.Slider(
                        minimum=0.5,
                        maximum=20,
                        value=5,
                        step=0.5,
                        label="Moire Threshold",
                        info="Sensitivity for detecting high-frequency patterns (lower = more sensitive)"
                    )
                    
                    banding_sigma_slider = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Banding Smoothing",
                        info="Size of low-frequency variations to capture (larger = broader gradients)"
                    )
                    
                    extract_btn = gr.Button("Extract Artifacts", variant="primary", size="lg")
                    save_data_btn = gr.Button("Save Data for Processing", variant="secondary")
                    
                    save_status = gr.Textbox(
                        label="Save Status",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    artifact_stats = gr.Textbox(
                        label="Artifact Statistics",
                        lines=10,
                        interactive=False
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            banding_output = gr.Image(
                                label="Banding (Low Frequency)",
                                type="pil",
                                height=300
                            )
                        
                        with gr.Column():
                            moire_output = gr.Image(
                                label="Moire Patterns (High Frequency)",
                                type="pil",
                                height=300
                            )
                    
                    combined_output = gr.Image(
                        label="Combined Artifacts (amplified 5x for visibility)",
                        type="pil",
                        height=400
                    )
                    
                    gr.Markdown("""
                    **Usage:** After extraction, click 'Save Data for Processing' to save the artifact maps as .npy files.
                    These can be loaded and applied to other images as multiplicative or additive overlays to simulate
                    the camera's actual optical and sensor characteristics.
                    """)
        
        with gr.Tab("4. apply_effects"):
            gr.Markdown("### Apply Screen Effects to Image")
            gr.Markdown("Load an image and apply captured camera effects with adjustable intensity")
            
            with gr.Row():
                with gr.Column(scale=1):
                    input_image_effects = gr.Image(
                        label="Upload Image to Process",
                        type="pil",
                        height=400
                    )
                    
                    gr.Markdown("#### Effect Intensity Controls")
                    
                    blur_strength_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.3,
                        step=0.05,
                        label="Focus Blur Strength",
                        info="0 = no blur, 1 = maximum spatially-varying blur"
                    )
                    
                    ca_strength_slider = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=2,
                        step=0.5,
                        label="Chromatic Aberration Strength",
                        info="RGB channel separation (pixels)"
                    )
                    
                    banding_strength_slider = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1,
                        step=0.1,
                        label="Color Banding Strength",
                        info="Multiplier for captured banding pattern"
                    )
                    
                    moire_strength_slider = gr.Slider(
                        minimum=0,
                        maximum=5,
                        value=1,
                        step=0.1,
                        label="Moire Pattern Strength",
                        info="Multiplier for captured moire patterns"
                    )
                    
                    apply_effects_btn = gr.Button("Apply Effects", variant="primary", size="lg")
                    save_image_btn = gr.Button("Save as PNG", variant="secondary")
                    
                    save_image_status = gr.Textbox(
                        label="Save Status",
                        lines=2,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    processed_image_output = gr.Image(
                        label="Processed Image",
                        type="pil",
                        height=600
                    )
                    
                    processing_stats = gr.Textbox(
                        label="Processing Statistics",
                        lines=10,
                        interactive=False
                    )
                    
                    gr.Markdown("""
                    **Notes:**
                    - Image dimensions are automatically handled (effects are scaled to match)
                    - **Focus blur**: Uses analyzed focus map from Tab 2 (falls back to synthetic if not available)
                    - **Chromatic aberration**: Uses analyzed RGB shifts from Tab 2 (falls back to synthetic if not available)
                    - **Banding & Moire**: Uses captured data from Tab 3 (resized to fit)
                    - All saves are PNG format (never webp)
                    - Run Tab 2 and Tab 3 analyses first to capture real camera characteristics
                    - Adjust sliders and click 'Apply Effects' to see results in real-time
                    """)
        
        # Event handlers
        generate_btn.click(
            fn=generate_dot_image,
            inputs=[density_slider, dot_size_slider],
            outputs=[dot_image_output, stats_output]
        )
        
        analyze_focus_btn.click(
            fn=analyze_focus_quality,
            inputs=[upload_image],
            outputs=[focus_output, focus_stats]
        )
        
        analyze_ca_btn.click(
            fn=analyze_chromatic_aberration,
            inputs=[upload_image],
            outputs=[ca_output, ca_stats]
        )
        
        extract_btn.click(
            fn=extract_moire_and_banding,
            inputs=[white_screen_image, moire_threshold_slider, banding_sigma_slider],
            outputs=[banding_output, moire_output, combined_output, artifact_stats]
        )
        
        save_data_btn.click(
            fn=save_moire_banding_data,
            inputs=[white_screen_image, moire_threshold_slider, banding_sigma_slider],
            outputs=[save_status]
        )
        
        apply_effects_btn.click(
            fn=apply_screen_effects,
            inputs=[input_image_effects, blur_strength_slider, ca_strength_slider, 
                   banding_strength_slider, moire_strength_slider],
            outputs=[processed_image_output, processing_stats]
        )
        
        save_image_btn.click(
            fn=save_processed_image,
            inputs=[processed_image_output],
            outputs=[save_image_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(inbrowser=True)
