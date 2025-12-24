"""
Quick test for video generation fix
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import io

def test_video_frame_generation():
    """Test the video frame generation method"""
    print("Testing video frame generation with matplotlib fix...")
    
    # Create a simple test plot
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)  # 512x512
    fig.patch.set_facecolor('black')
    
    # Create some test data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, color='cyan', linewidth=2)
    ax.set_title('Test Frame', color='white')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    
    try:
        # Test the new frame conversion method
        fig.canvas.draw()
        
        # Save to temporary buffer and read back
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        
        # Read back as image
        pil_img = PIL.Image.open(buf)
        pil_img = pil_img.resize((512, 512), PIL.Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        frame_rgb = np.array(pil_img)
        if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 4:  # RGBA
            frame_rgb = frame_rgb[:, :, :3]  # Remove alpha channel
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        buf.close()
        plt.close(fig)
        
        print(f"✅ Frame generated successfully: {frame_bgr.shape}")
        print(f"   Data type: {frame_bgr.dtype}")
        print(f"   Min/Max values: {frame_bgr.min()}/{frame_bgr.max()}")
        
        # Save a test frame
        cv2.imwrite('test_frame.png', frame_bgr)
        print("✅ Test frame saved as 'test_frame.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Frame generation failed: {e}")
        return False

if __name__ == "__main__":
    success = test_video_frame_generation()
    if success:
        print("\n🎉 Video frame generation should now work in the music visualizer!")
    else:
        print("\n❌ There may still be issues with video generation.")