"""
Demo script for testing slitscanner functionality without GUI
Creates sample video and processes it to demonstrate the slitscanning effect
"""

import cv2
import numpy as np
from PIL import Image
import os


def create_demo_video(filename="demo_video.mp4", duration=3, fps=30):
    """Create a simple demo video with moving shapes"""
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    for frame_num in range(total_frames):
        # Create a frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            frame[y, :] = [int(255 * y / height), 50, 100]
        
        # Add moving circle
        center_x = int(width * 0.2 + (width * 0.6) * (frame_num / total_frames))
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Add moving rectangle
        rect_x = int(width * 0.8 - (width * 0.6) * (frame_num / total_frames))
        rect_y = height // 3
        cv2.rectangle(frame, (rect_x - 20, rect_y - 20), (rect_x + 20, rect_y + 20), (0, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✅ Demo video created: {filename}")
    return filename


def slitscan_video(video_path, frame_step=1, max_width=3000):
    """Process video using slitscanning technique"""
    print(f"🎬 Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video info: {frame_width}x{frame_height}, {total_frames} frames @ {fps} FPS")
    
    # Calculate center column and frames to process
    center_x = frame_width // 2
    frames_to_process = min(total_frames // frame_step, max_width)
    
    print(f"🎯 Processing {frames_to_process} frames (every {frame_step} frame{'s' if frame_step > 1 else ''})")
    print(f"📍 Extracting center column at x={center_x}")
    
    # Create output image array
    slitscanned_image = np.zeros((frame_height, frames_to_process, 3), dtype=np.uint8)
    
    frame_count = 0
    processed_count = 0
    
    while cap.isOpened() and processed_count < frames_to_process:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if we should process this frame
        if frame_count % frame_step == 0:
            # Extract center column
            center_column = frame[:, center_x, :]
            
            # Add to slitscanned image
            slitscanned_image[:, processed_count, :] = center_column
            
            processed_count += 1
            
            # Show progress
            if processed_count % 10 == 0 or processed_count == frames_to_process:
                progress = processed_count / frames_to_process * 100
                print(f"⏳ Progress: {processed_count}/{frames_to_process} ({progress:.1f}%)")
        
        frame_count += 1
    
    cap.release()
    
    # Convert BGR to RGB
    slitscanned_image = cv2.cvtColor(slitscanned_image, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Slitscanning complete! Result: {processed_count}x{frame_height} pixels")
    
    return Image.fromarray(slitscanned_image)


def main():
    """Run the demo"""
    print("🎬 Slitscanner Demo")
    print("=" * 50)
    
    # Create demo video
    video_file = create_demo_video()
    
    try:
        # Process with different frame steps
        for step, name in [(1, "every_frame"), (2, "every_2_frames"), (5, "every_5_frames")]:
            print(f"\n🎯 Processing with frame step {step} ({name})...")
            
            result_image = slitscan_video(video_file, frame_step=step)
            
            output_filename = f"slitscan_demo_{name}.png"
            result_image.save(output_filename)
            
            print(f"💾 Saved: {output_filename} ({result_image.width}x{result_image.height})")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    finally:
        # Clean up demo video
        if os.path.exists(video_file):
            os.remove(video_file)
            print(f"🧹 Cleaned up: {video_file}")
    
    print("\n🎉 Demo complete! Check the generated PNG files to see the slitscanning effect.")


if __name__ == "__main__":
    main()
