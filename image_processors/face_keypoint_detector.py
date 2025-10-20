"""
Face Keypoint Detection using MediaPipe
A Gradio web application that detects facial landmarks and keypoints using Google's MediaPipe framework.
"""

import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
import webbrowser
import os

class FaceKeypointDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure face mesh detection
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,  # Focus on main face only
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect_face_keypoints(self, image, show_keypoints=True, keypoint_style="landmarks"):
        """
        Detect face keypoints using MediaPipe
        
        Args:
            image: PIL Image or numpy array
            show_keypoints: Whether to draw keypoints on the image
            keypoint_style: Style of keypoints - "landmarks", "contours", or "both"
        
        Returns:
            processed_image: Image with or without keypoints
            projected_face: Normalized frontal projection of the face
            keypoints_info: Dictionary with keypoint information
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_np = np.array(image)
            else:
                image_np = image.copy()
            
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                # Check if it's BGR (common with OpenCV) or RGB
                # For safety, assume it's RGB since Gradio typically provides RGB
                rgb_image = image_np
            else:
                rgb_image = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_image)
            
            # Prepare output image
            output_image = rgb_image.copy()
            projected_face = None
            keypoints_info = {
                "faces_detected": 0,
                "total_landmarks": 0,
                "face_dimensions": None,
                "confidence": None
            }
            
            if results.multi_face_landmarks:
                keypoints_info["faces_detected"] = len(results.multi_face_landmarks)
                
                # Process the main face (first detected face)
                face_landmarks = results.multi_face_landmarks[0]
                keypoints_info["total_landmarks"] = len(face_landmarks.landmark)
                
                # Calculate face bounding box
                h, w, _ = rgb_image.shape
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                face_width = max(x_coords) - min(x_coords)
                face_height = max(y_coords) - min(y_coords)
                keypoints_info["face_dimensions"] = {
                    "width": int(face_width),
                    "height": int(face_height),
                    "center_x": int(sum(x_coords) / len(x_coords)),
                    "center_y": int(sum(y_coords) / len(y_coords))
                }
                
                # Create projected face
                projected_face = self.create_projected_face(rgb_image, face_landmarks)
                
                if show_keypoints:
                    # Calculate center point between eyes and mouth
                    landmarks = face_landmarks.landmark
                    
                    # Get eye pupil positions (approximate center of eyes)
                    # Left eye center: landmark 468 (left pupil) - using 159 (left eye center)
                    # Right eye center: landmark 469 (right pupil) - using 386 (right eye center)
                    left_eye_center = (int(landmarks[159].x * w), int(landmarks[159].y * h))
                    right_eye_center = (int(landmarks[386].x * w), int(landmarks[386].y * h))
                    
                    # Get mouth corner positions
                    left_mouth_corner = (int(landmarks[61].x * w), int(landmarks[61].y * h))
                    right_mouth_corner = (int(landmarks[291].x * w), int(landmarks[291].y * h))
                    
                    # Calculate center positions
                    eye_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
                    eye_center_y = (left_eye_center[1] + right_eye_center[1]) // 2
                    
                    mouth_center_x = (left_mouth_corner[0] + right_mouth_corner[0]) // 2
                    mouth_center_y = (left_mouth_corner[1] + right_mouth_corner[1]) // 2
                    
                    # Calculate the single center point
                    center_x = eye_center_x  # Horizontally centered between pupils
                    center_y = (eye_center_y + mouth_center_y) // 2  # Vertically centered between eyes and mouth
                    
                    # Draw single green dot (8 pixel diameter = 4 pixel radius)
                    cv2.circle(output_image, (center_x, center_y), 4, (0, 255, 0), -1)
            
            return output_image, projected_face, keypoints_info
            
        except Exception as e:
            # Return original image with error info
            if isinstance(image, np.ndarray):
                original_img = image
            else:
                original_img = np.array(image)
            return original_img, None, {
                "error": str(e),
                "faces_detected": 0,
                "total_landmarks": 0
            }
    
    def create_projected_face(self, image, face_landmarks):
        """Create a normalized frontal projection of the detected face"""
        try:
            h, w, _ = image.shape
            landmarks = face_landmarks.landmark
            
            # Get more accurate key facial landmarks for better face extraction
            # Eye centers for better alignment
            left_eye_center = (int(landmarks[159].x * w), int(landmarks[159].y * h))  # Left eye center
            right_eye_center = (int(landmarks[386].x * w), int(landmarks[386].y * h))  # Right eye center
            
            # Nose tip and base for vertical reference
            nose_tip = (int(landmarks[1].x * w), int(landmarks[1].y * h))  # Nose tip
            nose_base = (int(landmarks[2].x * w), int(landmarks[2].y * h))  # Nose base
            
            # Mouth corners for width reference
            left_mouth = (int(landmarks[61].x * w), int(landmarks[61].y * h))  # Left mouth corner
            right_mouth = (int(landmarks[291].x * w), int(landmarks[291].y * h))  # Right mouth corner
            
            # Forehead and chin for height reference
            forehead_center = (int(landmarks[9].x * w), int(landmarks[9].y * h))  # Forehead center
            chin_center = (int(landmarks[175].x * w), int(landmarks[175].y * h))  # Chin center
            
            # Calculate face center using eye midpoint and nose
            face_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
            face_center_y = (left_eye_center[1] + nose_tip[1]) // 2
            
            # Calculate face dimensions more accurately
            # Width: distance between eyes * 2.8 (accounts for face width beyond eyes)
            eye_distance = np.sqrt((right_eye_center[0] - left_eye_center[0])**2 + 
                                 (right_eye_center[1] - left_eye_center[1])**2)
            face_width = int(eye_distance * 2.8)
            
            # Height: use forehead to chin distance, or estimate from eye to mouth distance
            forehead_to_chin = np.sqrt((chin_center[0] - forehead_center[0])**2 + 
                                     (chin_center[1] - forehead_center[1])**2)
            
            # Fallback: estimate height from eye to mouth distance * 2.5
            eye_to_mouth = abs(left_eye_center[1] - left_mouth[1])
            estimated_height = int(eye_to_mouth * 2.5)
            
            # Use the larger of the two height estimates for better coverage
            face_height = int(max(forehead_to_chin, estimated_height))
            
            # Ensure minimum face size
            face_width = max(face_width, 100)
            face_height = max(face_height, 120)
            
            # Calculate face bounding box with some padding
            padding_x = int(face_width * 0.15)  # 15% padding on sides
            padding_y = int(face_height * 0.1)   # 10% padding top/bottom
            
            x1 = max(0, face_center_x - face_width // 2 - padding_x)
            y1 = max(0, face_center_y - face_height // 2 - padding_y)
            x2 = min(w, face_center_x + face_width // 2 + padding_x)
            y2 = min(h, face_center_y + face_height // 2 + padding_y)
            
            # Extract face region
            face_roi = image[y1:y2, x1:x2]
            
            if face_roi.size == 0:
                return None
            
            # Resize to standard size for consistency (maintaining aspect ratio)
            target_width = 256
            target_height = 320
            
            # Calculate resize dimensions maintaining aspect ratio
            roi_h, roi_w = face_roi.shape[:2]
            scale = min(target_width / roi_w, target_height / roi_h)
            
            new_w = int(roi_w * scale)
            new_h = int(roi_h * scale)
            
            # Resize the face
            resized_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Create a centered image on the target canvas
            projected_face = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate centering offsets
            y_offset = (target_height - new_h) // 2
            x_offset = (target_width - new_w) // 2
            
            # Place the resized face in the center
            projected_face[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_face
            
            # Apply slight contrast and brightness enhancement
            projected_face = cv2.convertScaleAbs(projected_face, alpha=1.05, beta=5)
            
            return projected_face
            
        except Exception as e:
            print(f"Error creating projected face: {e}")
            return None
    
    def format_keypoints_info(self, info):
        """Format keypoints information for display"""
        if "error" in info:
            return f"❌ Error: {info['error']}"
        
        if info["faces_detected"] == 0:
            return "❌ No faces detected in the image"
        
        output = []
        output.append(f"✅ **Face Detection Results:**")
        output.append(f"• **Faces detected:** {info['faces_detected']}")
        output.append(f"• **Total landmarks:** {info['total_landmarks']}")
        
        if info["face_dimensions"]:
            dims = info["face_dimensions"]
            output.append(f"• **Face size:** {dims['width']} × {dims['height']} pixels")
            output.append(f"• **Face center:** ({dims['center_x']}, {dims['center_y']})")
        
        return "\n".join(output)

# Initialize the detector
detector = FaceKeypointDetector()

def process_image(image, show_keypoints):
    """Process uploaded image and return results"""
    if image is None:
        return None, None, "Please upload an image first."
    
    # Process the image (keypoint_style is no longer needed)
    processed_image, projected_face, keypoints_info = detector.detect_face_keypoints(
        image, show_keypoints, "single_dot"
    )
    
    # Format information text
    info_text = detector.format_keypoints_info(keypoints_info)
    
    return processed_image, projected_face, info_text

# Create Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(
        title="Face Keypoint Detection",
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .image-container {
            max-height: 600px;
        }
        """
    ) as iface:
        
        gr.Markdown(
            """
            # 🎭 Face Keypoint Detection with MediaPipe
            
            Upload an image to detect facial landmarks and keypoints using Google's MediaPipe framework.
            The system will identify up to 468 facial landmarks on the main face in the image and create
            a normalized frontal projection for analysis.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                image_input = gr.Image(
                    label="Upload Image",
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    elem_classes=["image-container"]
                )
                
                with gr.Group():
                    gr.Markdown("### **Detection Options**")
                    
                    show_keypoints = gr.Checkbox(
                        label="Show center point on image",
                        value=True
                    )
                
                process_btn = gr.Button(
                    "🔍 Detect Face Keypoints",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output
                image_output = gr.Image(
                    label="Processed Image with Keypoints",
                    type="pil",
                    elem_classes=["image-container"]
                )
                
                projected_output = gr.Image(
                    label="Projected Face (Normalized)",
                    type="numpy",
                    elem_classes=["image-container"]
                )
                
                info_output = gr.Markdown(
                    label="Detection Information",
                    value="Upload an image and click 'Detect Face Keypoints' to see results."
                )
        

        

        
        # Event handlers
        process_btn.click(
            fn=process_image,
            inputs=[image_input, show_keypoints],
            outputs=[image_output, projected_output, info_output]
        )
        
        # Auto-process when image is uploaded
        image_input.change(
            fn=process_image,
            inputs=[image_input, show_keypoints],
            outputs=[image_output, projected_output, info_output]
        )
        
        # Auto-process when settings change
        show_keypoints.change(
            fn=process_image,
            inputs=[image_input, show_keypoints],
            outputs=[image_output, projected_output, info_output]
        )
    
    return iface

def main():
    """Main function to launch the Gradio app"""
    try:
        # Configure Firefox as the preferred browser
        firefox_path = 'firefox'  # Default fallback
        try:
            # Try to register Firefox browser
            possible_firefox_paths = [
                '/usr/bin/firefox',
                '/usr/bin/firefox-esr', 
                '/opt/firefox/firefox',
                '/snap/bin/firefox',
                'firefox'  # In PATH
            ]
            
            for path in possible_firefox_paths:
                if os.path.exists(path) or path == 'firefox':
                    firefox_path = path
                    break
            
            if firefox_path:
                webbrowser.register('firefox', None, webbrowser.BackgroundBrowser(firefox_path))
                webbrowser.get('firefox')
                print(f"✅ Firefox browser configured: {firefox_path}")
            else:
                print("⚠️ Firefox not found, using default browser")
        except Exception as e:
            print(f"⚠️ Could not configure Firefox: {e}, using default browser")
        
        # Set environment variable to prefer Firefox
        os.environ['BROWSER'] = firefox_path
        
        # Create and launch interface
        iface = create_interface()
        
        print("🎭 Face Keypoint Detection App")
        print("=" * 50)
        print("Starting Gradio interface...")
        print("MediaPipe Face Mesh initialized successfully!")
        
        # Launch with appropriate settings
        iface.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Standard Gradio port
            share=False,            # Set to True to create public link
            show_error=True,        # Show errors in interface
            quiet=False,            # Show startup logs
            inbrowser=True          # Open browser automatically
        )
        
    except ImportError as e:
        print(f"❌ Missing required packages: {e}")
        print("\nPlease install required packages:")
        print("pip install gradio opencv-python mediapipe pillow numpy")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()
