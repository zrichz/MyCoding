# Imports first
import xml.etree.ElementTree as ET
import os
from svgoutline import svg_to_outlines
import sys

# Helper functions
def sanitize_filename(filename):
    """Replace spaces with underscores in filename"""
    return filename.replace(' ', '_')



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


# Main execution block
if __name__ == "__main__":
    try:
        testfile = "gimp_exported_path.svg"
        test_file = os.path.join(os.getcwd(), "images_general", testfile)
        print(f"Looking for file at: {test_file}")

        root = add_dimensions_to_svg(test_file)
        outlines = svg_to_outlines(root)
        
        print("\nSVG Structure:")
        for elem in root.iter():
            print(f"Tag: {elem.tag}, Attributes: {elem.attrib}")
            
        print("\nExtracted Outlines:")
        for i, outline in enumerate(outlines):
            print(f"\nOutline {i+1}:")
            formatted_outline = [f"{item:.2f}" if isinstance(item, float) else item for item in outline]
            print(formatted_outline)
            
        if outlines:
            print("\nthere are outlines present, should you wish to process further")
            
        
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)