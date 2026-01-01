"""
Face Morphing Video Creator

Creates smooth morphing videos between two face images using facial landmarks
and Delaunay triangulation for accurate facial feature alignment.

Features:
- Detects 468 facial landmarks using MediaPipe
- Aligns faces automatically
- Smooth morphing with Delaunay triangulation
- Configurable morph duration and video settings
- Exports as 24fps MP4 video
"""

# Suppress verbose logging from MediaPipe and TensorFlow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['GLOG_minloglevel'] = '2'  # Suppress Glog logging (MediaPipe)

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import cv2
import numpy as np
from PIL import Image
import gradio as gr
from scipy.spatial import Delaunay
import tkinter as tk
from tkinter import filedialog
import mediapipe as mp


class FaceMorphVideoCreator:
    def __init__(self):
        # Initialize MediaPipe Face Mesh for accurate landmark detection (468 points)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,  # Lower threshold for small faces
            min_tracking_confidence=0.3
        )
        print("✓ Using MediaPipe Face Mesh with 468-point landmark model")
        
        # Video settings
        self.fps = 24
        self.morph_duration = 3.0  # seconds
    
    def draw_landmarks_on_image(self, image, points):
        """Draw facial landmarks on image for preview"""
        img_copy = image.copy()
        
        # Draw all landmark points
        for point in points:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img_copy, (x, y), 1, (0, 255, 0), -1)
        
        # MediaPipe Face Mesh connections for better visualization
        # Key facial features using MediaPipe landmark indices
        
        # Face oval (outer contour)
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        # Left eye
        LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Right eye  
        RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Lips outer
        LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409]
        
        # Lips inner
        LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        
        # Left eyebrow
        LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
        
        # Right eyebrow
        RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
        
        # Nose
        NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
        
        # Draw contours for key features
        contours = {
            'face': FACE_OVAL,
            'left_eye': LEFT_EYE,
            'right_eye': RIGHT_EYE,
            'lips_outer': LIPS_OUTER,
            'lips_inner': LIPS_INNER,
            'left_eyebrow': LEFT_EYEBROW,
            'right_eyebrow': RIGHT_EYEBROW,
            'nose': NOSE
        }
        
        for feature_name, indices in contours.items():
            for i in range(len(indices)):
                idx1 = indices[i]
                idx2 = indices[(i + 1) % len(indices)]
                if idx1 < len(points) and idx2 < len(points):
                    pt1 = (int(points[idx1][0]), int(points[idx1][1]))
                    pt2 = (int(points[idx2][0]), int(points[idx2][1]))
                    cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1)
        
        return img_copy
    
    def draw_key_landmarks_debug(self, frame, points):
        """Draw key facial landmarks for debugging (eyes, nose, mouth)"""
        frame_copy = frame.copy()
        
        # Key landmark indices for MediaPipe 468-point model
        key_points = {
            'left_eye_center': 468,      # Left iris center (if available)
            'right_eye_center': 473,     # Right iris center
            'nose_tip': 4,               # Nose tip
            'left_mouth': 61,            # Left mouth corner
            'right_mouth': 291,          # Right mouth corner
            'chin': 152,                 # Chin center
            'left_cheek': 234,           # Left cheek
            'right_cheek': 454,          # Right cheek
        }
        
        # Draw circles for key points with labels
        for name, idx in key_points.items():
            if idx < len(points):
                x, y = int(points[idx][0]), int(points[idx][1])
                # Draw larger, more visible circles
                cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)  # Green filled circle
                cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), 2)   # Red outline
        
        return frame_copy
        
    def detect_face_landmarks(self, image, face_hint=None):
        """
        Detect facial landmarks in an image using MediaPipe
        
        Args:
            image: Input image (PIL Image or numpy array)
            face_hint: Optional tuple (x, y) indicating approximate face center (currently unused with MediaPipe)
        """
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Ensure it's a valid numpy array
        if not isinstance(image_np, np.ndarray):
            return None, None
        
        # Ensure uint8 format first
        if image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        # Handle different image formats - MediaPipe expects RGB
        if len(image_np.shape) == 2:
            # Grayscale image - convert to RGB
            image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif len(image_np.shape) == 3:
            if image_np.shape[2] == 4:
                # RGBA image - convert to RGB
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            elif image_np.shape[2] == 3:
                # Already RGB (from PIL/Gradio)
                image_rgb = image_np.copy()
            else:
                return None, None
        else:
            return None, None
        
        # Ensure RGB is uint8
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
        
        h, w = image_rgb.shape[:2]
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, None
        
        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized landmarks to pixel coordinates
        points = []
        for landmark in face_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            points.append([x, y])
        
        points = np.array(points, dtype=np.float32)
        
        # Return RGB image for consistency
        return points, image_rgb
    
    def add_boundary_points(self, points, img_shape):
        """Add boundary points for complete image coverage"""
        h, w = img_shape[:2]
        
        # Add corner and edge points
        boundary_points = [
            [0, 0], [w//2, 0], [w-1, 0],
            [0, h//2], [w-1, h//2],
            [0, h-1], [w//2, h-1], [w-1, h-1]
        ]
        
        return np.vstack([points, boundary_points])
    
    def ease_in_out(self, t):
        """Smooth ease-in-out interpolation (cubic easing)"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    def resize_to_match(self, img1, img2, points2, target_size=None, half_resolution=False):
        """Resize first image and crop second image to match dimensions, adjusting points"""
        if target_size:
            target_w, target_h = target_size
        else:
            # Use first image dimensions
            target_h, target_w = img1.shape[:2]
        
        # Apply half resolution if requested
        if half_resolution:
            target_w = target_w // 2
            target_h = target_h // 2
        
        # Resize first image
        img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Crop second image to match first image's aspect ratio and dimensions
        h2, w2 = img2.shape[:2]
        target_aspect = target_w / target_h
        current_aspect = w2 / h2
        
        # Track crop offsets for adjusting landmark points
        crop_x_offset = 0
        crop_y_offset = 0
        
        if abs(current_aspect - target_aspect) > 0.01:  # Different aspect ratios
            # Crop to match aspect ratio first
            if current_aspect > target_aspect:
                # Image is wider, crop width
                new_w2 = int(h2 * target_aspect)
                crop_x_offset = (w2 - new_w2) // 2
                img2 = img2[:, crop_x_offset:crop_x_offset + new_w2]
            else:
                # Image is taller, crop height
                new_h2 = int(w2 / target_aspect)
                crop_y_offset = (h2 - new_h2) // 2
                img2 = img2[crop_y_offset:crop_y_offset + new_h2, :]
        
        # Adjust points2 for crop offset
        points2_adjusted = points2.copy()
        points2_adjusted[:, 0] -= crop_x_offset
        points2_adjusted[:, 1] -= crop_y_offset
        
        # Now resize to target dimensions
        img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Scale the adjusted points for the resize
        h2_cropped, w2_cropped = img2.shape[:2]
        scale_x = target_w / w2_cropped
        scale_y = target_h / h2_cropped
        points2_adjusted[:, 0] *= scale_x
        points2_adjusted[:, 1] *= scale_y
        
        return img1_resized, img2_resized, points2_adjusted, (target_w, target_h)
    
    def morph_triangle(self, img1, img2, img_morph, tri1, tri2, tri_morph, alpha):
        """Morph a single triangle between two images"""
        # Get bounding rectangles
        rect1 = cv2.boundingRect(tri1)
        rect2 = cv2.boundingRect(tri2)
        rect_morph = cv2.boundingRect(tri_morph)
        
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        x, y, w, h = rect_morph
        
        if w <= 0 or h <= 0 or w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return
        
        # Convert triangle coordinates to be relative to bounding rectangles
        tri1_rect = np.array([[tri1[i][0] - x1, tri1[i][1] - y1] for i in range(3)], dtype=np.float32)
        tri2_rect = np.array([[tri2[i][0] - x2, tri2[i][1] - y2] for i in range(3)], dtype=np.float32)
        tri_morph_rect = np.array([[tri_morph[i][0] - x, tri_morph[i][1] - y] for i in range(3)], dtype=np.float32)
        
        # Create mask for the morphed triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, tri_morph_rect.astype(np.int32), 255)
        
        # Get affine transformation matrices
        try:
            mat1 = cv2.getAffineTransform(tri1_rect, tri_morph_rect)
            mat2 = cv2.getAffineTransform(tri2_rect, tri_morph_rect)
        except:
            return
        
        # Extract and warp rectangle regions from source images
        img1_rect = img1[y1:y1+h1, x1:x1+w1]
        img2_rect = img2[y2:y2+h2, x2:x2+w2]
        
        warp1 = cv2.warpAffine(img1_rect, mat1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warp2 = cv2.warpAffine(img2_rect, mat2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Blend the two warped images
        blended = cv2.addWeighted(warp1, 1.0 - alpha, warp2, alpha, 0)
        
        # Copy blended triangle to output using mask
        if x >= 0 and y >= 0 and x + w <= img_morph.shape[1] and y + h <= img_morph.shape[0]:
            # Apply mask to blend only the triangle region
            for c in range(3):
                img_morph[y:y+h, x:x+w, c] = np.where(mask > 0, blended[:, :, c], img_morph[y:y+h, x:x+w, c])
    
    def create_morph_sequence(self, img1, points1, img2, points2, num_frames, target_size=None, half_resolution=False, debug_overlay=False, progress_callback=None):
        """Create morphing sequence between two images with easing"""
        frames = []
        
        # Resize images to match target resolution and adjust points2 for cropping
        img1_resized, img2_resized, points2_adjusted, final_size = self.resize_to_match(
            img1, img2, points2, target_size, half_resolution
        )
        target_w, target_h = final_size
        
        # Scale points1 to match resized image1
        h1, w1 = img1.shape[:2]
        scale_x = target_w / w1
        scale_y = target_h / h1
        
        points1_scaled = points1 * [scale_x, scale_y]
        
        # points2_adjusted already accounts for crop and scale from resize_to_match
        points2_scaled = points2_adjusted
        
        # Add boundary points for complete coverage
        points1_full = self.add_boundary_points(points1_scaled, img1_resized.shape)
        points2_full = self.add_boundary_points(points2_scaled, img2_resized.shape)
        
        # Compute Delaunay triangulation on average shape
        points_avg = (points1_full + points2_full) / 2.0
        triangulation = Delaunay(points_avg)
        simplices = triangulation.simplices
        
        # Convert images to uint8 for faster processing
        img1_uint8 = img1_resized.astype(np.uint8)
        img2_uint8 = img2_resized.astype(np.uint8)
        
        for frame_idx in range(num_frames):
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx / num_frames, f"Generating frame {frame_idx+1}/{num_frames}")
            
            # Calculate linear alpha
            t = frame_idx / (num_frames - 1) if num_frames > 1 else 0.0
            
            # Apply ease-in-out
            alpha = self.ease_in_out(t)
            
            # Interpolate points
            points_morph = (1.0 - alpha) * points1_full + alpha * points2_full
            
            # Create morphed image
            img_morph = np.zeros_like(img1_uint8, dtype=np.uint8)
            
            # Morph each triangle
            for simplex in simplices:
                tri1 = points1_full[simplex].astype(np.float32)
                tri2 = points2_full[simplex].astype(np.float32)
                tri_morph = points_morph[simplex].astype(np.float32)
                
                self.morph_triangle(
                    img1_uint8,
                    img2_uint8,
                    img_morph,
                    tri1, tri2, tri_morph,
                    alpha
                )
            
            # Add debug overlay if enabled
            if debug_overlay:
                img_morph = self.draw_key_landmarks_debug(img_morph, points_morph)
            
            frames.append(img_morph)
        
        return frames
    
    def save_video(self, frames, output_path, fps=24):
        """Save frames as MP4 video with multiple codec fallbacks"""
        if not frames:
            return False, "No frames to save"
        
        try:
            # Check first frame dimensions and format
            first_frame = frames[0]
            if len(first_frame.shape) != 3:
                return False, f"Invalid frame format: expected 3D array, got {first_frame.shape}"
            
            h, w = first_frame.shape[:2]
            
            # Ensure dimensions are even (required by many codecs)
            if w % 2 != 0:
                w = w - 1
            if h % 2 != 0:
                h = h - 1
            
            # Try multiple codecs in order of preference
            codecs_to_try = [
                ('X264', 'H.264'),  # H.264 (best quality, widely compatible)
                ('mp4v', 'MPEG-4'),  # MPEG-4 Part 2 (good compatibility)
                ('MJPG', 'MJPEG'),  # Motion JPEG (always available)
            ]
            
            out = None
            codec_used = None
            
            for codec_name, codec_desc in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec_name)
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    
                    if out.isOpened():
                        codec_used = codec_desc
                        break
                    else:
                        out.release()
                        out = None
                except Exception:
                    if out is not None:
                        out.release()
                    out = None
                    continue
            
            if out is None or not out.isOpened():
                return False, "Failed to initialize video writer with any codec"
            
            # Write all frames
            for i, frame in enumerate(frames):
                # Ensure frame is correct size
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Ensure frame is BGR format (OpenCV requirement)
                if len(frame.shape) == 3:
                    if frame.shape[2] == 3:
                        # Assume it's RGB, convert to BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    elif frame.shape[2] == 4:
                        # RGBA, convert to BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    else:
                        return False, f"Unexpected number of channels in frame {i}: {frame.shape[2]}"
                else:
                    return False, f"Frame {i} has invalid shape: {frame.shape}"
                
                # Ensure uint8 dtype
                if frame_bgr.dtype != np.uint8:
                    frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
                
                out.write(frame_bgr)
            
            out.release()
            return True, f"Video saved: {output_path} (codec: {codec_used})"
            
        except Exception as e:
            if 'out' in locals() and out is not None:
                out.release()
            import traceback
            return False, f"Error saving video: {e}\n{traceback.format_exc()}"


# Global instance
morpher = FaceMorphVideoCreator()


def detect_faces_preview(image1, image2, hint1_str="", hint2_str=""):
    """Detect and preview face landmarks on both images"""
    try:
        if image1 is None or image2 is None:
            return None, None, "Please upload both images"
        
        # Get image dimensions for debug info
        if isinstance(image1, Image.Image):
            img1_size = f"{image1.width}x{image1.height}"
        elif isinstance(image1, np.ndarray):
            img1_size = f"{image1.shape[1]}x{image1.shape[0]}"
        else:
            img1_size = "unknown"
            
        if isinstance(image2, Image.Image):
            img2_size = f"{image2.width}x{image2.height}"
        elif isinstance(image2, np.ndarray):
            img2_size = f"{image2.shape[1]}x{image2.shape[0]}"
        else:
            img2_size = "unknown"
        
        # Parse face hints from coordinate strings
        hint1_x, hint1_y = parse_hint_coords(hint1_str)
        hint1 = None
        if hint1_x is not None and hint1_y is not None and hint1_x > 0 and hint1_y > 0:
            hint1 = (int(hint1_x), int(hint1_y))
            
        hint2_x, hint2_y = parse_hint_coords(hint2_str)
        hint2 = None
        if hint2_x is not None and hint2_y is not None and hint2_x > 0 and hint2_y > 0:
            hint2 = (int(hint2_x), int(hint2_y))
        
        # Detect landmarks
        points1, img1 = morpher.detect_face_landmarks(image1, face_hint=hint1)
        points2, img2 = morpher.detect_face_landmarks(image2, face_hint=hint2)
        
        if points1 is None:
            hint_msg = "\n\n💡 TIP: If automatic detection fails, try entering face center coordinates (X, Y) below the image." if hint1 is None else ""
            return None, None, f"No face detected in first image (size: {img1_size})\nTry: 1) Ensure face is clearly visible, 2) Face should be at least 80x80 pixels, 3) Good lighting and contrast{hint_msg}"
        if points2 is None:
            hint_msg = "\n\n💡 TIP: If automatic detection fails, try entering face center coordinates (X, Y) below the image." if hint2 is None else ""
            return None, None, f"No face detected in second image (size: {img2_size})\nTry: 1) Ensure face is clearly visible, 2) Face should be at least 80x80 pixels, 3) Good lighting and contrast{hint_msg}"
        
        # Draw landmarks on images
        preview1 = morpher.draw_landmarks_on_image(img1, points1)
        preview2 = morpher.draw_landmarks_on_image(img2, points2)
        
        # Draw hint markers if they were used
        if hint1 is not None:
            cv2.circle(preview1, hint1, 10, (0, 255, 0), 2)
            cv2.drawMarker(preview1, hint1, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(preview1, "Hint", (hint1[0] + 15, hint1[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        if hint2 is not None:
            cv2.circle(preview2, hint2, 10, (0, 255, 0), 2)
            cv2.drawMarker(preview2, hint2, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(preview2, "Hint", (hint2[0] + 15, hint2[1] - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        hint_status = ""
        if hint1 is not None or hint2 is not None:
            hint_status = " (with face hints)"
        
        status = f"✓ Face detected in both images{hint_status}\n" \
                f"Image 1: {img1.shape[1]}x{img1.shape[0]} - {len(points1)} landmarks\n" \
                f"Image 2: {img2.shape[1]}x{img2.shape[0]} - {len(points2)} landmarks"
        
        return preview1, preview2, status
        
    except Exception as e:
        import traceback
        return None, None, f"Error: {str(e)}\n{traceback.format_exc()}"


def create_face_morph_video(image1, image2, duration, half_resolution, debug_overlay, output_folder, hint1_str="", hint2_str="", progress=gr.Progress()):
    """Main function to create face morph video"""
    try:
        if image1 is None or image2 is None:
            return "Please upload both images", None
        
        if not output_folder or not os.path.isdir(output_folder):
            return "Please select a valid output folder", None
        
        progress(0, desc="Detecting faces...")
        
        # Parse face hints from coordinate strings
        hint1_x, hint1_y = parse_hint_coords(hint1_str)
        hint1 = None
        if hint1_x is not None and hint1_y is not None and hint1_x > 0 and hint1_y > 0:
            hint1 = (int(hint1_x), int(hint1_y))
            
        hint2_x, hint2_y = parse_hint_coords(hint2_str)
        hint2 = None
        if hint2_x is not None and hint2_y is not None and hint2_x > 0 and hint2_y > 0:
            hint2 = (int(hint2_x), int(hint2_y))
        
        # Detect landmarks in both images
        points1, img1 = morpher.detect_face_landmarks(image1, face_hint=hint1)
        points2, img2 = morpher.detect_face_landmarks(image2, face_hint=hint2)
        
        if points1 is None:
            hint_msg = "\n\n💡 TIP: Try entering face center coordinates (X, Y) to help detection." if hint1 is None else ""
            return f"No face detected in first image{hint_msg}", None
        if points2 is None:
            hint_msg = "\n\n💡 TIP: Try entering face center coordinates (X, Y) to help detection." if hint2 is None else ""
            return f"No face detected in second image{hint_msg}", None
        
        progress(0.2, desc="Creating morph sequence...")
        
        # Get target size from first image
        target_size = (img1.shape[1], img1.shape[0])  # (width, height)
        
        # Calculate number of frames
        num_frames = int(duration * morpher.fps)
        
        # Create morph sequence using original images with target resolution
        frames = morpher.create_morph_sequence(
            img1, points1,
            img2, points2,
            num_frames,
            target_size=target_size,
            half_resolution=half_resolution,
            debug_overlay=debug_overlay,
            progress_callback=lambda p, desc: progress(0.2 + p * 0.7, desc=desc)
        )
        
        progress(0.9, desc="Saving video...")
        
        # Create a temporary file for Gradio (will be auto-managed by Gradio's cache)
        import tempfile
        temp_dir = tempfile.gettempdir()
        temp_output = os.path.join(temp_dir, "face_morph_temp.mp4")
        
        # Save video to temp location first
        success, message = morpher.save_video(frames, temp_output, morpher.fps)
        
        if not success:
            progress(1.0, desc="Error!")
            return message, None
        
        # Also save to user's selected folder
        output_path = os.path.join(output_folder, "face_morph_output.mp4")
        try:
            import shutil
            shutil.copy2(temp_output, output_path)
            final_message = f"Video saved to: {output_path}"
        except Exception as e:
            final_message = f"Video created but copy failed: {e}\nTemp file at: {temp_output}"
            output_path = temp_output
        
        progress(1.0, desc="Complete!")
        
        # Return temp file path for Gradio to display
        return final_message, temp_output
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error: {str(e)}\n\nDetails:\n{error_details}", None


def parse_hint_coords(hint_str):
    """Parse hint coordinate string into x, y tuple or return None"""
    if not hint_str or hint_str.strip() == "":
        return None, None
    try:
        parts = hint_str.strip().split(',')
        if len(parts) == 2:
            x = int(parts[0].strip())
            y = int(parts[1].strip())
            return x, y
    except:
        pass
    return None, None


def handle_image_select(image, evt: gr.SelectData):
    """Handle click on image to set face hint"""
    if image is None:
        return ""
    
    # Get click coordinates from event
    x, y = evt.index[0], evt.index[1]
    
    # Return formatted coordinates
    return f"{x}, {y}"


def clear_hint():
    """Clear face hint coordinates"""
    return ""


def browse_folder():
    """Open folder selection dialog"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title="Select Output Folder")
    root.destroy()
    return folder if folder else ""


# Gradio Interface
def launch_gradio():
    with gr.Blocks(title="Face Morphing Video Creator") as app:
        gr.Markdown("# Face Morphing Video Creator")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Images")
                gr.Markdown("💡 **Tip:** Click on the nose area if automatic detection fails")
                
                image1_input = gr.Image(
                    label="First Face - Click on nose to set hint",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    height=400,
                    width=400
                )
                
                hint1_coords = gr.Textbox(
                    label="Face 1 Hint (auto-filled on click)",
                    value="",
                    interactive=False,
                    visible=True
                )
                
                clear_hint1_btn = gr.Button("Clear Face 1 Hint", size="sm", variant="secondary")
                
                image2_input = gr.Image(
                    label="Second Face - Click on nose to set hint",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    height=400,
                    width=400
                )
                
                hint2_coords = gr.Textbox(
                    label="Face 2 Hint (auto-filled on click)",
                    value="",
                    interactive=False,
                    visible=True
                )
                
                clear_hint2_btn = gr.Button("Clear Face 2 Hint", size="sm", variant="secondary")
                
                detect_btn = gr.Button("Detect Faces", variant="secondary", size="lg")
                
                gr.Markdown("### Settings")
                
                duration_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.0,
                    step=0.5,
                    label="Morph Duration (seconds)"
                )
                
                half_res_checkbox = gr.Checkbox(
                    label="Half Resolution Output",
                    value=False,
                    info="Render at 50% resolution for faster processing"
                )
                
                debug_checkbox = gr.Checkbox(
                    label="Debug Mode (Show Key Landmarks)",
                    value=False,
                    info="Overlay key facial points on output video for debugging"
                )
                
                # Set default output folder to be in the image_processors directory
                default_output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
                os.makedirs(default_output_folder, exist_ok=True)  # Create if it doesn't exist
                
                output_folder = gr.Textbox(
                    label="Output Folder",
                    value=default_output_folder,
                    placeholder="Click below to browse",
                    info="Where to save the output video"
                )
                
                browse_btn = gr.Button("Browse for Folder", size="sm")
                
                create_btn = gr.Button("Create Morph Video", variant="primary", size="lg")
            
            with gr.Column():
                gr.Markdown("### Face Detection Preview")
                
                with gr.Row():
                    preview1_output = gr.Image(
                        label="Face 1 Landmarks",
                        type="numpy"
                    )
                    
                    preview2_output = gr.Image(
                        label="Face 2 Landmarks",
                        type="numpy"
                    )
                
                detection_status = gr.Textbox(
                    label="Detection Status",
                    lines=3,
                    interactive=False
                )
                
                status_text = gr.Textbox(
                    label="Processing Status",
                    lines=2,
                    interactive=False
                )
                
                video_output = gr.Video(
                    label="Output Video",
                    interactive=False
                )
        
        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. Upload two face images (frontal faces work best)
        2. **Optional:** If automatic face detection fails, **click on the nose area** of each face to set a detection hint
           - Click directly on the image to set coordinates
           - A hint marker will appear when you detect faces
           - Click "Clear Hint" to remove and try automatic detection again
        3. Click 'Detect Faces' to verify face detection (green markers show hints if used)
        4. Adjust settings (duration, resolution)
        5. Select output folder (can be anywhere - Downloads, Documents, etc.)
        6. Click 'Create Morph Video' and wait for processing
        7. Video will be saved as `face_morph_output.mp4` in your selected folder
        8. You can also download the video using the download button below the preview
        
        **Note:** Small faces work best if they're at least 80x80 pixels. The app automatically 
        upscales images during detection to find smaller faces. If detection still fails, click 
        on the nose area to help guide the algorithm.
        """)
        
        # Event handlers
        # Handle clicks on images to set face hints
        image1_input.select(
            fn=handle_image_select,
            inputs=[image1_input],
            outputs=[hint1_coords]
        )
        
        image2_input.select(
            fn=handle_image_select,
            inputs=[image2_input],
            outputs=[hint2_coords]
        )
        
        # Clear hint buttons
        clear_hint1_btn.click(
            fn=clear_hint,
            inputs=[],
            outputs=[hint1_coords]
        )
        
        clear_hint2_btn.click(
            fn=clear_hint,
            inputs=[],
            outputs=[hint2_coords]
        )
        
        detect_btn.click(
            fn=detect_faces_preview,
            inputs=[image1_input, image2_input, hint1_coords, hint2_coords],
            outputs=[preview1_output, preview2_output, detection_status]
        )
        
        browse_btn.click(
            fn=browse_folder,
            inputs=[],
            outputs=[output_folder]
        )
        
        create_btn.click(
            fn=create_face_morph_video,
            inputs=[image1_input, image2_input, duration_slider, half_res_checkbox, debug_checkbox, output_folder, hint1_coords, hint2_coords],
            outputs=[status_text, video_output]
        )
    
    return app


if __name__ == "__main__":
    app = launch_gradio()
    app.queue(max_size=20)
    
    # Note: Videos are saved to temp directory for Gradio compatibility
    # Then copied to user's selected output folder
    # This avoids Gradio's InvalidPathError for non-allowed directories
    app.launch(
        max_threads=10,
        show_error=True,
        inbrowser=True
    )
