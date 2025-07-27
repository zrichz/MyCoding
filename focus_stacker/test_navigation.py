#!/usr/bin/env python3
"""
Test navigation in debug window
"""

import tkinter as tk
import customtkinter as ctk
import cv2
import numpy as np
from gradient_debug_window import GradientStackingDebugWindow

def test_navigation():
    """Test debug window navigation independently."""
    
    # Create simple test images
    images = []
    for i in range(4):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        # Add different colored rectangles to distinguish images
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        img[50:150, 50:250] = colors[i]
        # Add text to identify image
        cv2.putText(img, f"Image {i+1}", (80, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        images.append(img)
    
    # Create root window
    root = ctk.CTk()
    root.title("Navigation Test")
    root.geometry("300x200")
    
    def open_debug():
        debug_window = GradientStackingDebugWindow(root, images)
    
    # Add button to open debug window
    open_button = ctk.CTkButton(root, text="Open Debug Window", command=open_debug)
    open_button.pack(pady=50)
    
    print("Click 'Open Debug Window' to test navigation")
    print("The debug window should allow you to navigate between 4 test images")
    
    root.mainloop()

if __name__ == "__main__":
    test_navigation()
