# Imports first
import xml.etree.ElementTree as ET
import os
from svgoutline import svg_to_outlines
import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np
from PyQt5.QtWidgets import QApplication

# Helper functions
def sanitize_filename(filename):
    """Replace spaces with underscores in filename"""
    return filename.replace(' ', '_')

def plot_outlines(outlines):
    """Plot the extracted outlines using matplotlib"""
    if not outlines:
        print("No outlines to plot")
        return
    
    # Initialize Qt application if not already running
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
    fig, ax = plt.subplots(figsize=(10, 10))
    for outline in outlines:
        vertices = np.array(outline)
        n_points = len(vertices)
        codes = [Path.MOVETO] + [Path.LINETO] * (n_points-2) + [Path.CLOSEPOLY]
        path = Path(vertices, codes)
        patch = PathPatch(path, facecolor='none', edgecolor='black')
        ax.add_patch(patch)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title('SVG Outlines')
    plt.show()

def add_dimensions_to_svg(svg_path):
    abs_path = os.path.abspath(svg_path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"SVG file not found: {abs_path}")
    
    if os.path.getsize(abs_path) == 0:
        raise ValueError(f"SVG file is empty: {abs_path}")
        
    print(f"Processing SVG file: {abs_path}")
    
    try:
        tree = ET.parse(abs_path)
        root = tree.getroot()
        if 'width' not in root.attrib or 'height' not in root.attrib:
            root.set('width', '100mm')
            root.set('height', '100mm')
            tree.write(abs_path)
        return root
    except ET.ParseError as e:
        print(f"Error parsing SVG file: {e}")
        raise

if __name__ == "__main__":
    testfile = "gimp_exported_path.svg"
    test_file = os.path.join(os.getcwd(), "images_general", testfile)
    print(f"Looking for file at: {test_file}")

    try:
        root = add_dimensions_to_svg(test_file)
        print("\nNote: Any 'qt.svg: QSvgHandler' messages below are just warnings and can be ignored:")
        print("-" * 70)
        
        # Extract outlines here
        outlines = svg_to_outlines(root)
        
        # Debug info
        print("SVG Structure:")
        for elem in root.iter():
            print(f"Tag: {elem.tag}, Attributes: {elem.attrib}")
        
        # Plot outlines
        if outlines:
            print("\nPlotting outlines...")
            plot_outlines(outlines)
        else:
            print("\nNo outlines found in SVG")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)