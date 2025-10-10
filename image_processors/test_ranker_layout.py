#!/usr/bin/env python3
"""
Test script for the redesigned image ranker layout.
This script launches the image ranker with the new 2560x1440 optimized layout.
"""

import tkinter as tk
from image_ranker import ImageRanker

def main():
    root = tk.Tk()
    app = ImageRanker(root)
    
    # Set window to appear on screen (not withdrawn)
    root.deiconify()
    
    print("Image Ranker launched with streamlined workflow:")
    print("- Window size: 2560x1440")
    print("- Left sidebar: 400px wide with controls")
    print("- Right area: Maximized for image comparison")
    print("- Canvas size: 1000x1300 pixels each")
    print("- Simple Add/Cancel dialog instead of ranking results window")
    print("\nTo test:")
    print("1. Select a directory with images")
    print("2. Click 'Start Ranking' to begin comparisons")
    print("3. Use mouse clicks or 'T' key for ties")
    print("4. At completion, simple dialog asks 'Add ranks or Cancel'")
    print("5. Files are automatically renamed with completion message")
    
    root.mainloop()

if __name__ == "__main__":
    main()