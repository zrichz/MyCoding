# Filmic Effects Processor - Gradio Version 1
# Applies film grain, vignette, and photographic effects to images

import gradio as gr
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import os
from pathlib import Path
import cv2
#import mediapipe as mp

# Face detection setup
try:
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,  # Long-range model
        min_detection_confidence=0.3
    )
    FACE_DETECTION_AVAILABLE = True
    print("✓ Face detection initialized")
except Exception as e:
    FACE_DETECTION_AVAILABLE = False
    face_detection = None
    print(f"✗ Face detection unavailable: {e}")

# Global variables for face detection state
current_face_center = (360, 360)
current_face_detected = False


def detect_face_center(image):
    """Detect face center using MediaPipe Face Detection"""
    global current_face_center, current_face_detected
    
    if not FACE_DETECTION_AVAILABLE or face_detection is None:
        current_face_detected = False
        width, height = image.size
        current_face_center = (width // 2, int(height * 0.25))
        return current_face_center, False
        
    try:
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Ensure RGB
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            rgb_image = image_np[:, :, :3]
        elif len(image_np.shape) == 3 and image_np.shape[2] == 3:
            rgb_image = image_np
        else:
            rgb_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        
        h, w = rgb_image.shape[:2]
        
        # Detect faces
        detection_results = face_detection.process(rgb_image)
        
        if detection_results.detections and len(detection_results.detections) > 0:
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            center_x = int((bbox.xmin + bbox.width / 2) * w)
            center_y = int((bbox.ymin + bbox.height / 2) * h)
            
            face_width = int(bbox.width * w)
            face_height = int(bbox.height * h)
            
            print(f"Face detected! Size: {face_width}x{face_height}px, Center: ({center_x}, {center_y})")
            current_face_center = (center_x, center_y)
            current_face_detected = True
            return current_face_center, True
        else:
            print("No face detected")
            
    except Exception as e:
        print(f"Face detection error: {e}")
    
    # Fallback
    current_face_detected = False
    current_face_center = (360, 360)
    return current_face_center, False


def create_vignette(width, height, strength):
    """Create circular vignette effect"""
    center_x, center_y = width // 2, height // 2
    y, x = np.ogrid[:height, :width]
    
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    vignette = distance / max_distance
    vignette = np.clip(vignette, 0, 1)
    vignette = 1 - (vignette ** 2 * strength)
    
    return vignette


def apply_chromatic_aberration(image, strength, center):
    """Apply chromatic aberration effect"""
    if strength == 0.0:
        return image
        
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    center_x, center_y = center
    
    r, g, b = image.split()
    r_array = np.array(r)
    g_array = np.array(g)
    b_array = np.array(b)
    
    # Create distance map
    y, x = np.ogrid[:height, :width]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Aberration factor
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
                red_rolled = np.roll(r_array, shift_val, axis=1)
                red_rolled[:, :shift_val] = r_array[:, :shift_val]
                r_shifted = np.where(shift_mask, red_rolled, r_shifted)
                
                blue_rolled = np.roll(b_array, -shift_val, axis=1)
                blue_rolled[:, -shift_val:] = b_array[:, -shift_val:]
                b_shifted = np.where(shift_mask, blue_rolled, b_shifted)
    
    result = Image.merge('RGB', (
        Image.fromarray(r_shifted.astype(np.uint8)),
        Image.fromarray(g_array.astype(np.uint8)),
        Image.fromarray(b_shifted.astype(np.uint8))
    ))
    
    return result


def apply_saturation_reduction(image, reduction):
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


def create_vintage_border(image):
    """Create white border with rounded corners"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    border_width = 12
    corner_radius = 6
    
    bordered_image = image.copy()
    img_array = np.array(bordered_image)
    
    y, x = np.ogrid[:height, :width]
    border_mask = np.zeros((height, width), dtype=bool)
    
    inner_left = border_width
    inner_right = width - border_width
    inner_top = border_width
    inner_bottom = height - border_width
    
    central_rect = (x >= inner_left) & (x < inner_right) & (y >= inner_top) & (y < inner_bottom)
    
    # Rounded corners
    tl_corner_x, tl_corner_y = inner_left, inner_top
    tl_distance = np.sqrt((x - tl_corner_x)**2 + (y - tl_corner_y)**2)
    tl_cutout = (x >= inner_left) & (x < inner_left + corner_radius) & \
               (y >= inner_top) & (y < inner_top + corner_radius) & \
               (tl_distance < corner_radius)
    
    tr_corner_x, tr_corner_y = inner_right, inner_top
    tr_distance = np.sqrt((x - tr_corner_x)**2 + (y - tr_corner_y)**2)
    tr_cutout = (x >= inner_right - corner_radius) & (x < inner_right) & \
               (y >= inner_top) & (y < inner_top + corner_radius) & \
               (tr_distance < corner_radius)
    
    bl_corner_x, bl_corner_y = inner_left, inner_bottom
    bl_distance = np.sqrt((x - bl_corner_x)**2 + (y - bl_corner_y)**2)
    bl_cutout = (x >= inner_left) & (x < inner_left + corner_radius) & \
               (y >= inner_bottom - corner_radius) & (y < inner_bottom) & \
               (bl_distance < corner_radius)
    
    br_corner_x, br_corner_y = inner_right, inner_bottom
    br_distance = np.sqrt((x - br_corner_x)**2 + (y - br_corner_y)**2)
    br_cutout = (x >= inner_right - corner_radius) & (x < inner_right) & \
               (y >= inner_bottom - corner_radius) & (y < inner_bottom) & \
               (br_distance < corner_radius)
    
    inner_area = central_rect & ~(tl_cutout | tr_cutout | bl_cutout | br_cutout)
    border_mask = ~inner_area
    
    img_array[border_mask] = [230, 225, 215]
    
    return Image.fromarray(img_array)


def apply_unsharp_sharpening(image, half_strength=True):
    """Apply unsharp mask sharpening"""
    try:
        radius = 1.0
        amount = 1.3
        threshold = 3
        
        sharpened = image.filter(ImageFilter.UnsharpMask(
            radius=int(radius),
            percent=int(amount * 100),
            threshold=int(threshold)
        ))
        
        if half_strength:
            return Image.blend(image, sharpened, 0.5)
        else:
            return sharpened
            
    except Exception as e:
        print(f"Sharpening error: {e}")
        return image


def apply_auto_contrast_stretch(image, strength_percent):
    """Apply automatic contrast stretching"""
    try:
        strength = strength_percent / 100.0
        
        if strength <= 0:
            return image
        
        img_array = np.array(image).astype(np.float32)
        percentile = 0.5
        
        low_percentiles = np.percentile(img_array, percentile, axis=(0, 1))
        high_percentiles = np.percentile(img_array, 100 - percentile, axis=(0, 1))
        
        stretched_array = np.zeros_like(img_array, dtype=np.float32)
        
        for i in range(img_array.shape[2]):
            channel = img_array[:, :, i]
            low_val = float(low_percentiles[i])
            high_val = float(high_percentiles[i])
            
            if high_val > low_val:
                stretched_channel = (channel - low_val) * 255.0 / (high_val - low_val)
                stretched_channel = np.clip(stretched_channel, 0, 255)
            else:
                stretched_channel = channel
            
            stretched_array[:, :, i] = stretched_channel
        
        if strength < 1.0:
            result_array = img_array * (1.0 - strength) + stretched_array * strength
        else:
            result_array = stretched_array
        
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        return Image.fromarray(result_array)
        
    except Exception as e:
        print(f"Auto-contrast error: {e}")
        return image


def apply_dithering_effect(image, dither_type, colors_per_channel):
    """Apply dithering effect"""
    try:
        img_array = np.array(image).astype(np.float32)
        h, w, c = img_array.shape
        colors_per_channel = max(2, min(256, colors_per_channel))
        
        if dither_type == "floyd-steinberg":
            result = img_array.copy()
            
            scale_factor = (colors_per_channel - 1) / 255.0
            inv_scale_factor = 255.0 / (colors_per_channel - 1)
            
            for y in range(h):
                for x in range(w):
                    old_pixel = result[y, x].copy()
                    
                    new_pixel = np.round(old_pixel * scale_factor) * inv_scale_factor
                    result[y, x] = new_pixel
                    
                    quant_error = old_pixel - new_pixel
                    
                    if x + 1 < w:
                        result[y, x + 1] += quant_error * 7/16
                    if y + 1 < h:
                        if x > 0:
                            result[y + 1, x - 1] += quant_error * 3/16
                        result[y + 1, x] += quant_error * 5/16
                        if x + 1 < w:
                            result[y + 1, x + 1] += quant_error * 1/16
            
            result = np.clip(result, 0, 255).astype(np.uint8)
            
        else:  # Bayer
            bayer_matrix = np.array([
                [ 0, 32,  8, 40,  2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44,  4, 36, 14, 46,  6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [ 3, 35, 11, 43,  1, 33,  9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47,  7, 39, 13, 45,  5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ]) / 64.0
            
            tile_h = (h + 7) // 8
            tile_w = (w + 7) // 8
            tiled_bayer = np.tile(bayer_matrix, (tile_h, tile_w))[:h, :w]
            tiled_bayer = tiled_bayer[:, :, np.newaxis]
            
            threshold = tiled_bayer * (255.0 / colors_per_channel)
            adjusted = img_array + threshold - 127.5 / colors_per_channel
            
            scale_factor = (colors_per_channel - 1) / 255.0
            inv_scale_factor = 255.0 / (colors_per_channel - 1)
            result = np.round(adjusted * scale_factor) * inv_scale_factor
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
        
    except Exception as e:
        print(f"Dithering error: {e}")
        return image


def make_image_look_photographed(
    img,
    apply_tone_curve=True,
    tone_strength=0.5,
    apply_grain=True,
    grain_strength=0.04,
    apply_chromatic_aberration=True,
    ca_shift=1,
    apply_halation=True,
    halation_strength=0.2,
    apply_vignette=True,
    vignette_strength=0.5,
    apply_scan_banding=True,
    banding_strength=0.01,
    apply_rgb_misalignment=True,
    misalign_px=1,
    paper_texture=None,
    texture_strength=0.1,
    apply_dust=True,
    dust_amount=0.001,
    scratch_amount=0.0002,
    apply_jpeg_artifacts=True,
    jpeg_quality=70,
    apply_lens_distortion=True,
    distortion_strength=-0.0005,
    apply_lut=False,
    lut=None,
):
    """Apply photographic effects to image (numpy BGR array)"""
    
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    # Tone curve
    if apply_tone_curve:
        def s_curve(x, s=tone_strength):
            return 1 / (1 + np.exp(-s * (x - 0.5)))
        img = s_curve(img)

    # Film grain
    if apply_grain:
        grain = np.random.normal(0, grain_strength, (h, w, 1))
        grain = np.repeat(grain, 3, axis=2)
        luma = img.mean(axis=2, keepdims=True)
        grain *= (0.5 - np.abs(luma - 0.5)) * 2
        img = np.clip(img + grain, 0, 1)

    # Chromatic aberration
    if apply_chromatic_aberration:
        def shift_channel(ch, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(ch, M, (w, h))

        b, g, r = cv2.split(img)
        r = shift_channel(r, ca_shift, 0)
        b = shift_channel(b, -ca_shift, 0)
        img = cv2.merge([b, g, r])

    # Halation
    if apply_halation:
        blur = cv2.GaussianBlur(img, (0, 0), 5)
        img = np.clip(img + blur * halation_strength, 0, 1)

    # Vignette
    if apply_vignette:
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        vignette = 1 - (dist / dist.max())**2 * vignette_strength
        vignette = vignette[..., None]
        img *= vignette

    # Scan banding
    if apply_scan_banding:
        bands = (np.sin(np.linspace(0, 50, h)) * banding_strength).reshape(h, 1, 1)
        img = np.clip(img + bands, 0, 1)

    # RGB misalignment
    if apply_rgb_misalignment:
        def shift(ch, dx, dy):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            return cv2.warpAffine(ch, M, (w, h))

        b, g, r = cv2.split(img)
        r = shift(r, misalign_px, 0)
        g = shift(g, 0, -misalign_px)
        b = shift(b, -misalign_px, misalign_px)
        img = cv2.merge([b, g, r])

    # Paper texture
    if paper_texture is not None:
        tex = cv2.resize(paper_texture.astype(np.float32) / 255.0, (w, h))
        img = img * (1 - texture_strength) + tex * texture_strength

    # Dust & scratches
    if apply_dust and (dust_amount > 0 or scratch_amount > 0):
        if dust_amount > 0:
            num_dust = int(h * w * dust_amount)
            if num_dust > 0:
                dust_y = np.random.randint(0, h, num_dust)
                dust_x = np.random.randint(0, w, num_dust)
                dust_sizes = np.random.randint(1, 4, num_dust)
                
                for y, x, size in zip(dust_y, dust_x, dust_sizes):
                    y1, y2 = max(0, y-size), min(h, y+size)
                    x1, x2 = max(0, x-size), min(w, x+size)
                    img[y1:y2, x1:x2] *= 0.7
        
        if scratch_amount > 0:
            num_scratches = int(h * scratch_amount * 10)
            if num_scratches > 0:
                for _ in range(num_scratches):
                    x = np.random.randint(0, w)
                    y_start = np.random.randint(0, h // 2)
                    y_end = y_start + np.random.randint(h // 4, h)
                    y_end = min(y_end, h)
                    thickness = np.random.randint(1, 2)
                    
                    x1, x2 = max(0, x-thickness), min(w, x+thickness)
                    img[y_start:y_end, x1:x2] = np.clip(img[y_start:y_end, x1:x2] + 0.15, 0, 1)

    # JPEG artifacts
    if apply_jpeg_artifacts:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, enc = cv2.imencode(".jpg", (img * 255).astype(np.uint8), encode_param)
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # Lens distortion
    if apply_lens_distortion:
        K = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
        D = np.array([distortion_strength, 0, 0, 0], dtype=np.float32)
        img = cv2.undistort(img, K, D)

    # LUT
    if apply_lut and lut is not None:
        img_uint = (img * 255).astype(np.uint8)
        img = cv2.LUT(img_uint, lut).astype(np.float32) / 255.0

    return (img * 255).astype(np.uint8)


def apply_photographic_effects(pil_image, apply_grain, grain_strength, 
                               apply_halation, halation_strength,
                               apply_jpeg_artifacts, jpeg_quality,
                               apply_lens_distortion, distortion_strength):
    """Apply photographic effects wrapper"""
    try:
        img_array = np.array(pil_image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        result_bgr = make_image_look_photographed(
            img_bgr,
            apply_tone_curve=False,
            tone_strength=0.5,
            apply_grain=apply_grain,
            grain_strength=grain_strength,
            apply_chromatic_aberration=False,
            ca_shift=1,
            apply_halation=apply_halation,
            halation_strength=halation_strength,
            apply_vignette=False,
            vignette_strength=0.5,
            apply_scan_banding=False,
            banding_strength=0.01,
            apply_rgb_misalignment=False,
            misalign_px=1,
            apply_dust=False,
            dust_amount=0.001,
            scratch_amount=0.0002,
            apply_jpeg_artifacts=apply_jpeg_artifacts,
            jpeg_quality=jpeg_quality,
            apply_lens_distortion=apply_lens_distortion,
            distortion_strength=distortion_strength,
            apply_lut=False,
            lut=None
        )
        
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
        
    except Exception as e:
        print(f"Photographic effects error: {e}")
        return pil_image


def process_image(
    input_image,
    # Original effects
    vignette_enabled,
    vignette_strength,
    saturation_enabled,
    saturation_reduction,
    chromatic_enabled,
    chromatic_aberration,
    show_ca_center,
    vintage_border,
    unsharp_sharpening,
    unsharp_half_strength,
    auto_contrast_stretch,
    auto_contrast_strength,
    # Photo effects
    apply_photo_grain,
    photo_grain_strength,
    apply_halation,
    halation_strength,
    apply_jpeg_artifacts,
    jpeg_quality,
    apply_lens_distortion,
    distortion_strength,
    # Dithering
    apply_dithering,
    dithering_type,
    dithering_colors
):
    """Main processing function"""
    
    if input_image is None:
        return None, "No image provided"
    
    try:
        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            image = Image.fromarray(input_image)
        else:
            image = input_image
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Detect face for effect centering
        face_center, face_detected = detect_face_center(image)
        face_status = "✓ Face detected" if face_detected else "✗ No face detected"
        
        width, height = image.size
        
        # Apply vignette to HSV
        hsv_image = image.convert('HSV')
        hsv_array = np.array(hsv_image, dtype=np.float32)
        
        hsv_array[:, :, 0] /= 255.0
        hsv_array[:, :, 1] /= 255.0
        hsv_array[:, :, 2] /= 255.0
        
        if vignette_enabled:
            vignette = create_vignette(width, height, vignette_strength)
            hsv_array[:, :, 2] *= vignette
        
        hsv_array = np.clip(hsv_array, 0, 1)
        hsv_array *= 255.0
        
        hsv_result = Image.fromarray(hsv_array.astype(np.uint8), mode='HSV')
        result = hsv_result.convert('RGB')
        
        # Apply effects
        if chromatic_enabled:
            result = apply_chromatic_aberration(result, chromatic_aberration, face_center)
        
        if saturation_enabled:
            result = apply_saturation_reduction(result, saturation_reduction)
        
        if unsharp_sharpening:
            result = apply_unsharp_sharpening(result, unsharp_half_strength)
        
        if auto_contrast_stretch:
            result = apply_auto_contrast_stretch(result, auto_contrast_strength)
        
        if vintage_border:
            result = create_vintage_border(result)
        
        # Photographic effects
        if apply_photo_grain or apply_halation or apply_jpeg_artifacts or apply_lens_distortion:
            result = apply_photographic_effects(
                result, 
                apply_photo_grain, 
                photo_grain_strength,
                apply_halation, 
                halation_strength,
                apply_jpeg_artifacts, 
                jpeg_quality,
                apply_lens_distortion, 
                distortion_strength
            )
        
        # Dithering
        if apply_dithering:
            result = apply_dithering_effect(result, dithering_type, dithering_colors)
        
        # Show face center indicator
        if show_ca_center:
            result_with_indicator = result.copy()
            draw = ImageDraw.Draw(result_with_indicator)
            
            center_x, center_y = face_center
            radius = 8
            draw.ellipse([center_x-radius-1, center_y-radius-1, center_x+radius+1, center_y+radius+1], 
                       fill=None, outline=(0, 0, 0), width=2)
            draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                       fill=(0, 255, 0), outline=None)
            
            result = result_with_indicator
        
        info_text = f"Processed {width}x{height} | {face_status}"
        return result, info_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def batch_process_folder(
    folder_path,
    # All the same parameters as process_image
    vignette_enabled, vignette_strength,
    saturation_enabled, saturation_reduction,
    chromatic_enabled, chromatic_aberration,
    show_ca_center, vintage_border,
    unsharp_sharpening, unsharp_half_strength,
    auto_contrast_stretch, auto_contrast_strength,
    apply_photo_grain, photo_grain_strength,
    apply_halation, halation_strength,
    apply_jpeg_artifacts, jpeg_quality,
    apply_lens_distortion, distortion_strength,
    apply_dithering, dithering_type, dithering_colors
):
    """Batch process all images in a folder"""
    
    if not folder_path or not os.path.isdir(folder_path):
        return "Please provide a valid folder path"
    
    try:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file_path in Path(folder_path).iterdir():
            if file_path.suffix.lower() in extensions:
                image_files.append(file_path)
        
        if not image_files:
            return "No valid images found in folder"
        
        # Create output directory
        output_dir = Path(folder_path) / "filmic"
        output_dir.mkdir(exist_ok=True)
        
        processed_count = 0
        
        for image_path in image_files:
            try:
                image = Image.open(image_path)
                
                # Detect face for this image
                face_center, _ = detect_face_center(image)
                
                # Process with current settings
                result, _ = process_image(
                    image,
                    vignette_enabled, vignette_strength,
                    saturation_enabled, saturation_reduction,
                    chromatic_enabled, chromatic_aberration,
                    False,  # Don't show center indicator in batch
                    vintage_border,
                    unsharp_sharpening, unsharp_half_strength,
                    auto_contrast_stretch, auto_contrast_strength,
                    apply_photo_grain, photo_grain_strength,
                    apply_halation, halation_strength,
                    apply_jpeg_artifacts, jpeg_quality,
                    apply_lens_distortion, distortion_strength,
                    apply_dithering, dithering_type, dithering_colors
                )
                
                if result:
                    output_path = output_dir / f"{image_path.stem}_filmic.jpg"
                    result.save(output_path, quality=88)
                    processed_count += 1
                    
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        return f"Successfully processed {processed_count}/{len(image_files)} images!\nOutput saved to: {output_dir}"
        
    except Exception as e:
        return f"Batch processing error: {str(e)}"


# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Filmic Effects Processor") as demo:
    gr.Markdown("# 🎬 Filmic Effects Processor")
    gr.Markdown("Apply film grain, vignette, chromatic aberration, and other photographic effects to your images")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="pil")
            
            with gr.Tabs():
                with gr.Tab("Original Effects"):
                    with gr.Group():
                        vignette_enabled = gr.Checkbox(label="Vignette", value=True)
                        vignette_strength = gr.Slider(0, 1, value=0.15, step=0.01, label="Vignette Strength")
                    
                    with gr.Group():
                        saturation_enabled = gr.Checkbox(label="Saturation Reduction", value=True)
                        saturation_reduction = gr.Slider(0, 1, value=0.15, step=0.01, label="Saturation Amount")
                    
                    with gr.Group():
                        chromatic_enabled = gr.Checkbox(label="Chromatic Aberration", value=True)
                        chromatic_aberration = gr.Slider(0, 0.1, value=0.03, step=0.001, label="CA Amount")
                        show_ca_center = gr.Checkbox(label="Show Face Center", value=False)
                    
                    vintage_border = gr.Checkbox(label="Photo Border", value=False)
                    unsharp_sharpening = gr.Checkbox(label="Unsharp Sharpening", value=True)
                    unsharp_half_strength = gr.Checkbox(label="Apply at 50%", value=True)
                    
                    with gr.Group():
                        auto_contrast_stretch = gr.Checkbox(label="Auto-Contrast Stretch", value=True)
                        auto_contrast_strength = gr.Slider(0, 100, value=50, step=1, label="Contrast Strength %")
                
                with gr.Tab("Photo Effects 1"):
                    with gr.Group():
                        apply_photo_grain = gr.Checkbox(label="Photo Grain", value=True)
                        photo_grain_strength = gr.Slider(0, 0.1, value=0.025, step=0.001, label="Grain Strength")
                    
                    with gr.Group():
                        apply_halation = gr.Checkbox(label="Halation (Glow)", value=False)
                        halation_strength = gr.Slider(0, 0.2, value=0.05, step=0.01, label="Halation Strength")
                
                with gr.Tab("Photo Effects 2"):
                    with gr.Group():
                        apply_jpeg_artifacts = gr.Checkbox(label="JPEG Artifacts", value=False)
                        jpeg_quality = gr.Slider(1, 100, value=20, step=1, label="JPEG Quality")
                    
                    with gr.Group():
                        apply_lens_distortion = gr.Checkbox(label="Lens Distortion", value=False)
                        distortion_strength = gr.Slider(-0.002, 0.002, value=-0.0005, step=0.0001, 
                                                       label="Distortion (-ve=barrel, +ve=pincushion)")
                    
                    with gr.Group():
                        apply_dithering = gr.Checkbox(label="Color Dithering", value=False)
                        dithering_type = gr.Radio(["floyd-steinberg", "bayer"], 
                                                  value="floyd-steinberg", label="Dithering Type")
                        dithering_colors = gr.Slider(2, 16, value=4, step=1, label="Colors per Channel")
            
            process_btn = gr.Button("Process Image", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Processed Image", type="pil")
            info_text = gr.Textbox(label="Info", interactive=False)
    
    gr.Markdown("---")
    gr.Markdown("### 📁 Batch Processing")
    
    with gr.Row():
        folder_input = gr.Textbox(label="Folder Path", placeholder="/path/to/images")
        batch_btn = gr.Button("Process All Images in Folder", variant="secondary")
    
    batch_output = gr.Textbox(label="Batch Processing Status", interactive=False)
    
    # Connect events
    process_btn.click(
        fn=process_image,
        inputs=[
            input_image,
            vignette_enabled, vignette_strength,
            saturation_enabled, saturation_reduction,
            chromatic_enabled, chromatic_aberration,
            show_ca_center, vintage_border,
            unsharp_sharpening, unsharp_half_strength,
            auto_contrast_stretch, auto_contrast_strength,
            apply_photo_grain, photo_grain_strength,
            apply_halation, halation_strength,
            apply_jpeg_artifacts, jpeg_quality,
            apply_lens_distortion, distortion_strength,
            apply_dithering, dithering_type, dithering_colors
        ],
        outputs=[output_image, info_text]
    )
    
    batch_btn.click(
        fn=batch_process_folder,
        inputs=[
            folder_input,
            vignette_enabled, vignette_strength,
            saturation_enabled, saturation_reduction,
            chromatic_enabled, chromatic_aberration,
            show_ca_center, vintage_border,
            unsharp_sharpening, unsharp_half_strength,
            auto_contrast_stretch, auto_contrast_strength,
            apply_photo_grain, photo_grain_strength,
            apply_halation, halation_strength,
            apply_jpeg_artifacts, jpeg_quality,
            apply_lens_distortion, distortion_strength,
            apply_dithering, dithering_type, dithering_colors
        ],
        outputs=batch_output
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
