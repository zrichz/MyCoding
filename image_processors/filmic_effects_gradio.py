"""
Filmic Effects Processor - Gradio Version
A streamlined web application for applying cinematic effects to 720x1600 images.

Features:
- Film grain, vignette, saturation reduction, chromatic aberration
- Face-centered effects using MediaPipe detection with fallback
- Vintage borders, unsharp sharpening, auto-contrast
- Batch processing capabilities
- Real-time preview with face outline debugging
"""

import gradio as gr
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import zipfile
import tempfile

# Face detection imports
try:
    import cv2
    import mediapipe as mp
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    print("Warning: Face detection not available. Install opencv-python and mediapipe for face-centered effects.")

class FilmicEffectsProcessor:
    def __init__(self):
        # Default values for effects
        self.DEFAULT_GRAIN = 0.22
        self.DEFAULT_VIGNETTE = 0.34
        self.DEFAULT_SATURATION = 0.30
        self.DEFAULT_CHROMATIC = 0.09
        
        # Internal parameters (not exposed in GUI)
        self.unsharp_radius = 1.0
        self.unsharp_amount = 1.3
        self.unsharp_threshold = 3
        self.contrast_percentile = 0.5
        
        # Face detection setup
        self.face_mesh = None
        if FACE_DETECTION_AVAILABLE:
            try:
                mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"Face detection initialization failed: {e}")
                self.face_mesh = None
        
        # Current image state
        self.effect_center = (360, 360)
        self.face_detected = False
        self.face_landmarks = None

    def detect_face_center(self, image):
        """Detect face center using MediaPipe with progressive fallback"""
        if not FACE_DETECTION_AVAILABLE or self.face_mesh is None:
            self.face_detected = False
            self.face_landmarks = None
            return (360, 360)
            
        try:
            # Convert PIL Image to numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image.copy()
            
            # Ensure RGB format
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                rgb_image = image_np
            else:
                rgb_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # First attempt: Standard confidence (0.5)
            results = self.face_mesh.process(rgb_image)
            confidence_used = 0.5
            
            # Progressive fallback with lower confidence
            if not (results.multi_face_landmarks and len(results.multi_face_landmarks) > 0):
                try:
                    mp_face_mesh = mp.solutions.face_mesh
                    
                    # Try medium confidence (0.3)
                    fallback_face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                        min_detection_confidence=0.3, min_tracking_confidence=0.3
                    )
                    results = fallback_face_mesh.process(rgb_image)
                    confidence_used = 0.3
                    
                    # Try very low confidence (0.1) if still no face
                    if not (results.multi_face_landmarks and len(results.multi_face_landmarks) > 0):
                        very_low_face_mesh = mp_face_mesh.FaceMesh(
                            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                            min_detection_confidence=0.1, min_tracking_confidence=0.1
                        )
                        results = very_low_face_mesh.process(rgb_image)
                        confidence_used = 0.1
                        
                except Exception as fallback_error:
                    print(f"Fallback face detection failed: {fallback_error}")
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = face_landmarks.landmark
                self.face_landmarks = face_landmarks
                
                h, w = rgb_image.shape[:2]
                
                # Calculate eye centers and mouth corners
                left_eye_center = (int(landmarks[159].x * w), int(landmarks[159].y * h))
                right_eye_center = (int(landmarks[386].x * w), int(landmarks[386].y * h))
                left_mouth_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
                right_mouth_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
                
                # Calculate face center point
                eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
                eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
                mouth_center_y = (left_mouth_corner[1] + right_mouth_corner[1]) // 2
                
                center_x = eye_center_x
                center_y = (eye_center_y + mouth_center_y) // 2
                
                print(f"Face detected! (confidence: {confidence_used}) Center: ({center_x}, {center_y})")
                self.face_detected = True
                return (center_x, center_y)
                
        except Exception as e:
            print(f"Face detection error: {e}")
            
        # Return default center if all attempts fail
        self.face_detected = False
        self.face_landmarks = None
        return (360, 360)

    def draw_face_outline(self, draw, image_size):
        """Draw red outline of detected face using MediaPipe landmarks"""
        if not self.face_landmarks:
            return
            
        landmarks = self.face_landmarks.landmark
        h, w = image_size[1], image_size[0]
        
        # Face oval contour points
        face_oval = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        
        # Draw face outline
        outline_points = []
        for idx in face_oval:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                outline_points.append((x, y))
        
        if len(outline_points) > 2:
            for i in range(len(outline_points)):
                start_point = outline_points[i]
                end_point = outline_points[(i + 1) % len(outline_points)]
                draw.line([start_point, end_point], fill=(255, 0, 0), width=2)
        
        # Draw center point
        center_x, center_y = self.effect_center
        radius = 4
        draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                    fill=(255, 0, 0), outline=None)

    def apply_film_grain_rgb(self, image, intensity):
        """Apply film grain to RGB image"""
        if intensity == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        rgb_array = np.array(image, dtype=np.float32)
        width, height = image.size
        
        # Create grain noise
        grain = np.random.normal(0, intensity, (height, width, 3))
        grain *= 25.0
        rgb_array += grain
        rgb_array = np.clip(rgb_array, 0, 255)
        
        return Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')

    def apply_saturation_reduction_rgb(self, image, reduction):
        """Apply saturation reduction"""
        if reduction == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image, dtype=np.float32)
        
        saturation_factor = 1.0 - reduction
        hsv_array[:, :, 1] *= saturation_factor
        hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1], 0, 255)
        
        hsv_result = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        return hsv_result.convert('RGB')

    def create_vignette(self, width, height, strength):
        """Create circular vignette effect"""
        center_x, center_y = width // 2, height // 2
        y, x = np.ogrid[:height, :width]
        
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        vignette = distance / max_distance
        vignette = np.clip(vignette, 0, 1)
        vignette = 1 - (vignette ** 2 * strength)
        
        return vignette

    def apply_chromatic_aberration(self, image, strength):
        """Apply chromatic aberration effect"""
        if strength == 0.0:
            return image
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        center_x, center_y = self.effect_center
        
        r, g, b = image.split()
        r_array = np.array(r)
        g_array = np.array(g)
        b_array = np.array(b)
        
        # Create distance-based aberration
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        aberration_factor = np.zeros_like(distance)
        mask = distance > 250
        aberration_factor[mask] = np.minimum((distance[mask] - 250) / (800 - 250), 1.0)
        
        shift_amount = aberration_factor * strength * 30.0
        r_shifted = r_array.copy()
        b_shifted = b_array.copy()
        
        max_shift = int(np.ceil(np.max(shift_amount)))
        if max_shift > 0:
            x_shift = np.round(shift_amount).astype(int)
            
            for shift_val in range(1, max_shift + 1):
                shift_mask = (x_shift == shift_val)
                
                if np.any(shift_mask):
                    # Red channel shift right
                    red_rolled = np.roll(r_array, shift_val, axis=1)
                    red_rolled[:, :shift_val] = r_array[:, :shift_val]
                    r_shifted = np.where(shift_mask, red_rolled, r_shifted)
                    
                    # Blue channel shift left
                    blue_rolled = np.roll(b_array, -shift_val, axis=1)
                    blue_rolled[:, -shift_val:] = b_array[:, -shift_val:]
                    b_shifted = np.where(shift_mask, blue_rolled, b_shifted)
        
        return Image.merge('RGB', (
            Image.fromarray(r_shifted.astype(np.uint8)),
            Image.fromarray(g_array.astype(np.uint8)),
            Image.fromarray(b_shifted.astype(np.uint8))
        ))

    def create_vintage_border(self, image):
        """Create vintage photo border"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        border_width = 12
        corner_radius = 6
        
        img_array = np.array(image)
        y, x = np.ogrid[:height, :width]
        
        inner_left, inner_right = border_width, width - border_width
        inner_top, inner_bottom = border_width, height - border_width
        
        central_rect = (x >= inner_left) & (x < inner_right) & (y >= inner_top) & (y < inner_bottom)
        
        # Create rounded corner cutouts
        corners = [
            (inner_left, inner_top), (inner_right, inner_top),
            (inner_left, inner_bottom), (inner_right, inner_bottom)
        ]
        
        corner_cutouts = np.zeros_like(central_rect)
        for corner_x, corner_y in corners:
            distance = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
            corner_mask = (distance < corner_radius)
            if corner_x == inner_left:
                corner_mask &= (x >= inner_left) & (x < inner_left + corner_radius)
            else:
                corner_mask &= (x >= inner_right - corner_radius) & (x < inner_right)
            if corner_y == inner_top:
                corner_mask &= (y >= inner_top) & (y < inner_top + corner_radius)
            else:
                corner_mask &= (y >= inner_bottom - corner_radius) & (y < inner_bottom)
            corner_cutouts |= corner_mask
        
        inner_area = central_rect & ~corner_cutouts
        border_mask = ~inner_area
        
        img_array[border_mask] = [230, 225, 215]
        return Image.fromarray(img_array)

    def apply_unsharp_sharpening(self, image):
        """Apply unsharp mask sharpening"""
        try:
            return image.filter(ImageFilter.UnsharpMask(
                radius=int(self.unsharp_radius),
                percent=int(self.unsharp_amount * 100),
                threshold=int(self.unsharp_threshold)
            ))
        except Exception as e:
            print(f"Error applying unsharp sharpening: {e}")
            return image

    def apply_auto_contrast_stretch(self, image):
        """Apply automatic contrast stretching"""
        try:
            img_array = np.array(image)
            percentile = self.contrast_percentile
            
            low_percentiles = np.percentile(img_array, percentile, axis=(0, 1))
            high_percentiles = np.percentile(img_array, 100 - percentile, axis=(0, 1))
            
            stretched_array = np.zeros_like(img_array, dtype=np.float32)
            
            for i in range(img_array.shape[2]):
                channel = img_array[:, :, i].astype(np.float32)
                low_val = float(low_percentiles[i])
                high_val = float(high_percentiles[i])
                
                if high_val > low_val:
                    stretched_channel = (channel - low_val) * 255.0 / (high_val - low_val)
                    stretched_channel = np.clip(stretched_channel, 0, 255)
                else:
                    stretched_channel = channel
                
                stretched_array[:, :, i] = stretched_channel
            
            return Image.fromarray(stretched_array.astype(np.uint8))
        except Exception as e:
            print(f"Error applying auto contrast stretch: {e}")
            return image

    def apply_filmic_effects(self, image, grain_intensity=0.22, grain_enabled=True,
                           vignette_strength=0.34, vignette_enabled=True,
                           saturation_reduction=0.30, saturation_enabled=True,
                           chromatic_aberration=0.09, chromatic_enabled=True,
                           vintage_border=True, unsharp_sharpening=True,
                           auto_contrast_stretch=True):
        """Apply all filmic effects to image"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply vignette in HSV space
        if vignette_enabled:
            hsv_image = image.convert('HSV')
            hsv_array = np.array(hsv_image, dtype=np.float32)
            width, height = image.size
            
            hsv_array /= 255.0
            vignette = self.create_vignette(width, height, vignette_strength)
            hsv_array[:, :, 2] *= vignette
            hsv_array = np.clip(hsv_array, 0, 1) * 255.0
            
            hsv_result = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
            image = hsv_result.convert('RGB')
        
        # Apply other effects
        if chromatic_enabled:
            image = self.apply_chromatic_aberration(image, chromatic_aberration)
        
        if grain_enabled:
            image = self.apply_film_grain_rgb(image, grain_intensity)
        
        if saturation_enabled:
            image = self.apply_saturation_reduction_rgb(image, saturation_reduction)
        
        if unsharp_sharpening:
            image = self.apply_unsharp_sharpening(image)
        
        if auto_contrast_stretch:
            image = self.apply_auto_contrast_stretch(image)
        
        if vintage_border:
            image = self.create_vintage_border(image)
        
        return image

# Initialize processor
processor = FilmicEffectsProcessor()

def process_image(image, grain_intensity, grain_enabled, vignette_strength, vignette_enabled,
                 saturation_reduction, saturation_enabled, chromatic_aberration, chromatic_enabled,
                 vintage_border, unsharp_sharpening, auto_contrast_stretch, show_face_outline):
    """Process single image with effects"""
    if image is None:
        return None, None, "Please upload an image first."
    
    # Check image dimensions
    if image.size != (720, 1600):
        return None, None, f"⚠️ Image size is {image.size}, expected 720x1600 pixels"
    
    # Detect face center
    processor.effect_center = processor.detect_face_center(image)
    
    # Apply effects
    processed = processor.apply_filmic_effects(
        image, grain_intensity, grain_enabled, vignette_strength, vignette_enabled,
        saturation_reduction, saturation_enabled, chromatic_aberration, chromatic_enabled,
        vintage_border, unsharp_sharpening, auto_contrast_stretch
    )
    
    # Add face outline if requested
    if show_face_outline and processor.face_landmarks is not None:
        processed_with_outline = processed.copy()
        draw = ImageDraw.Draw(processed_with_outline)
        processor.draw_face_outline(draw, processed_with_outline.size)
        processed = processed_with_outline
    
    # Create status message
    face_status = "✅ Face detected" if processor.face_detected else "❌ No face detected"
    effects_list = []
    if grain_enabled: effects_list.append(f"Grain: {grain_intensity:.2f}")
    if vignette_enabled: effects_list.append(f"Vignette: {vignette_strength:.2f}")
    if saturation_enabled: effects_list.append(f"Saturation: -{saturation_reduction:.2f}")
    if chromatic_enabled: effects_list.append(f"Chromatic: {chromatic_aberration:.2f}")
    
    status = f"{face_status}\n\nEffects applied: {', '.join(effects_list) if effects_list else 'None'}"
    if vintage_border: status += "\n+ Vintage border"
    if unsharp_sharpening: status += "\n+ Unsharp sharpening"
    if auto_contrast_stretch: status += "\n+ Auto contrast"
    
    return image, processed, status

def process_batch(files, grain_intensity, grain_enabled, vignette_strength, vignette_enabled,
                 saturation_reduction, saturation_enabled, chromatic_aberration, chromatic_enabled,
                 vintage_border, unsharp_sharpening, auto_contrast_stretch, progress=gr.Progress()):
    """Process multiple images and return as zip file"""
    if not files:
        return None, "No files uploaded"
    
    progress(0, desc="Starting batch processing...")
    
    # Create temporary directory for processed images
    with tempfile.TemporaryDirectory() as temp_dir:
        processed_count = 0
        total_files = len(files)
        
        for i, file in enumerate(files):
            try:
                progress((i + 1) / total_files, desc=f"Processing {file.name}...")
                
                # Load and check image
                image = Image.open(file.name)
                if image.size != (720, 1600):
                    print(f"Skipping {file.name} - wrong dimensions: {image.size}")
                    continue
                
                # Detect face for this image
                processor.effect_center = processor.detect_face_center(image)
                
                # Apply effects
                processed = processor.apply_filmic_effects(
                    image, grain_intensity, grain_enabled, vignette_strength, vignette_enabled,
                    saturation_reduction, saturation_enabled, chromatic_aberration, chromatic_enabled,
                    vintage_border, unsharp_sharpening, auto_contrast_stretch
                )
                
                # Save processed image
                input_path = Path(file.name)
                output_filename = f"{input_path.stem}_filmic{input_path.suffix}"
                output_path = Path(temp_dir) / output_filename
                
                if input_path.suffix.lower() in ['.jpg', '.jpeg']:
                    processed.save(output_path, 'JPEG', quality=95, optimize=True)
                else:
                    processed.save(output_path, 'PNG', optimize=True)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {file.name}: {e}")
                continue
        
        if processed_count == 0:
            return None, "No images were processed successfully"
        
        # Create zip file
        progress(1, desc="Creating zip file...")
        zip_path = Path(temp_dir) / "filmic_processed.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in Path(temp_dir).glob("*_filmic.*"):
                zipf.write(file_path, file_path.name)
        
        return str(zip_path), f"Successfully processed {processed_count}/{total_files} images"

def reset_to_defaults():
    """Reset all sliders to default values"""
    return (
        processor.DEFAULT_GRAIN,      # grain_intensity
        processor.DEFAULT_VIGNETTE,   # vignette_strength  
        processor.DEFAULT_SATURATION, # saturation_reduction
        processor.DEFAULT_CHROMATIC   # chromatic_aberration
    )

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="🎬 Filmic Effects Processor",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .image-container {
            max-height: 600px;
        }
        """
    ) as interface:
        
        gr.Markdown(
            """
            # 🎬 Filmic Effects Processor
            
            Apply cinematic effects to 720×1600 images with face-centered positioning.
            Upload single images for preview or multiple images for batch processing.
            """
        )
        
        face_detection_status = "✅ Face Detection Available" if FACE_DETECTION_AVAILABLE else "❌ Face Detection Unavailable"
        gr.Markdown(f"**Status:** {face_detection_status}")
        
        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("🖼️ Single Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Image (720×1600)",
                            type="pil",
                            sources=["upload", "clipboard"]
                        )
                        
                        with gr.Group():
                            gr.Markdown("### **Effect Controls**")
                            
                            # Main effects
                            with gr.Row():
                                grain_enabled = gr.Checkbox(label="Film Grain", value=True)
                                vignette_enabled = gr.Checkbox(label="Vignette", value=True)
                                saturation_enabled = gr.Checkbox(label="Saturation Reduction", value=True)
                                chromatic_enabled = gr.Checkbox(label="Chromatic Aberration", value=True)
                            
                            grain_intensity = gr.Slider(0, 0.56, value=processor.DEFAULT_GRAIN, step=0.01,
                                                       label="Grain Intensity")
                            vignette_strength = gr.Slider(0, 0.75, value=processor.DEFAULT_VIGNETTE, step=0.01,
                                                         label="Vignette Strength")
                            saturation_reduction = gr.Slider(0, 1.0, value=processor.DEFAULT_SATURATION, step=0.01,
                                                            label="Saturation Reduction")
                            chromatic_aberration = gr.Slider(0, 1.0, value=processor.DEFAULT_CHROMATIC, step=0.01,
                                                            label="Chromatic Aberration")
                            
                            # Additional options
                            with gr.Row():
                                vintage_border = gr.Checkbox(label="Vintage Border", value=True)
                                unsharp_sharpening = gr.Checkbox(label="Unsharp Sharpening", value=True)
                                auto_contrast_stretch = gr.Checkbox(label="Auto Contrast", value=True)
                                show_face_outline = gr.Checkbox(label="Show Face Outline", value=False)
                            
                            reset_btn = gr.Button("🔄 Reset to Defaults", variant="secondary")
                    
                    with gr.Column(scale=2):
                        with gr.Row():
                            original_output = gr.Image(label="Original", type="pil", height=400)
                            processed_output = gr.Image(label="Processed", type="pil", height=400)
                        
                        status_output = gr.Textbox(
                            label="Processing Status",
                            value="Upload a 720×1600 image to begin",
                            max_lines=5
                        )
            
            # Batch Processing Tab
            with gr.TabItem("📁 Batch Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        files_input = gr.File(
                            label="Upload Images (720×1600)",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        gr.Markdown("### **Batch Effect Settings**")
                        gr.Markdown("*Use same controls as single image mode*")
                        
                        # Batch controls (same as single image)
                        with gr.Row():
                            batch_grain_enabled = gr.Checkbox(label="Film Grain", value=True)
                            batch_vignette_enabled = gr.Checkbox(label="Vignette", value=True)
                            batch_saturation_enabled = gr.Checkbox(label="Saturation Reduction", value=True)
                            batch_chromatic_enabled = gr.Checkbox(label="Chromatic Aberration", value=True)
                        
                        batch_grain_intensity = gr.Slider(0, 0.56, value=processor.DEFAULT_GRAIN, step=0.01,
                                                         label="Grain Intensity")
                        batch_vignette_strength = gr.Slider(0, 0.75, value=processor.DEFAULT_VIGNETTE, step=0.01,
                                                           label="Vignette Strength")
                        batch_saturation_reduction = gr.Slider(0, 1.0, value=processor.DEFAULT_SATURATION, step=0.01,
                                                              label="Saturation Reduction")
                        batch_chromatic_aberration = gr.Slider(0, 1.0, value=processor.DEFAULT_CHROMATIC, step=0.01,
                                                              label="Chromatic Aberration")
                        
                        with gr.Row():
                            batch_vintage_border = gr.Checkbox(label="Vintage Border", value=True)
                            batch_unsharp_sharpening = gr.Checkbox(label="Unsharp Sharpening", value=True)
                            batch_auto_contrast_stretch = gr.Checkbox(label="Auto Contrast", value=True)
                        
                        process_batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
                        batch_reset_btn = gr.Button("🔄 Reset Batch Defaults", variant="secondary")
                    
                    with gr.Column(scale=1):
                        download_output = gr.File(label="Download Processed Images")
                        batch_status_output = gr.Textbox(
                            label="Batch Status",
                            value="Upload multiple 720×1600 images and click 'Process Batch'",
                            max_lines=10
                        )
        
        # Event handlers for single image
        inputs_single = [
            image_input, grain_intensity, grain_enabled, vignette_strength, vignette_enabled,
            saturation_reduction, saturation_enabled, chromatic_aberration, chromatic_enabled,
            vintage_border, unsharp_sharpening, auto_contrast_stretch, show_face_outline
        ]
        outputs_single = [original_output, processed_output, status_output]
        
        # Auto-update on any change
        for component in inputs_single:
            component.change(fn=process_image, inputs=inputs_single, outputs=outputs_single)
        
        # Reset buttons
        reset_btn.click(
            fn=reset_to_defaults,
            outputs=[grain_intensity, vignette_strength, saturation_reduction, chromatic_aberration]
        )
        
        # Batch processing
        inputs_batch = [
            files_input, batch_grain_intensity, batch_grain_enabled, batch_vignette_strength, batch_vignette_enabled,
            batch_saturation_reduction, batch_saturation_enabled, batch_chromatic_aberration, batch_chromatic_enabled,
            batch_vintage_border, batch_unsharp_sharpening, batch_auto_contrast_stretch
        ]
        
        process_batch_btn.click(
            fn=process_batch,
            inputs=inputs_batch,
            outputs=[download_output, batch_status_output]
        )
        
        batch_reset_btn.click(
            fn=reset_to_defaults,
            outputs=[batch_grain_intensity, batch_vignette_strength, batch_saturation_reduction, batch_chromatic_aberration]
        )
    
    return interface

def main():
    """Launch the Gradio application"""
    try:
        interface = create_interface()
        
        print("🎬 Filmic Effects Processor - Gradio Version")
        print("=" * 50)
        print("Starting web interface...")
        if FACE_DETECTION_AVAILABLE:
            print("✅ Face detection available")
        else:
            print("❌ Face detection unavailable - install opencv-python and mediapipe")
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            inbrowser=True
        )
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("\nPlease install required packages:")
        print("pip install gradio pillow numpy")
        print("pip install opencv-python mediapipe  # For face detection")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
