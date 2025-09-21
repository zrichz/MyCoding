"""
Demonstration of New NCA Animator Features
==========================================

This script demonstrates the new features added to the NCA animator:
- 16x16 pixel support
- Frame interval control
- Size information display
"""

import tkinter as tk
from nca_animator import NCAAnimationGUI

def show_features():
    """Show the new features in action"""
    
    # Create the GUI
    root = tk.Tk()
    
    # Create and display the animator
    app = NCAAnimationGUI(root)
    
    # Add a welcome message
    welcome_msg = """
New Features Demo:

✅ 16x16 Support: Now supports tiny 16x16 models (scaled to 64x64)
✅ Frame Interval: Control how many steps to skip (1-10)
✅ Size Info: Real-time display of input → output dimensions
✅ Improved GUI: Cleaner layout with better information

Try different size options to see the scaling information update!
"""
    
    print(welcome_msg)
    
    # Run the GUI
    root.mainloop()

if __name__ == "__main__":
    show_features()
