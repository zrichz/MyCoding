import json
import os

def load_json(json_path):
    """Load JSON file and return data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def manipulate_data(data):
    """Placeholder for data manipulation."""
    # TODO: Add manipulation logic here
    shapes = data.get('shapes', [])
    return shapes

def save_to_obj(shapes, obj_path):
    """Save shapes data to OBJ file."""
    if not shapes:
        print("No data to save")
        return
    
    with open(obj_path, 'w') as objfile:
        objfile.write("# OBJ file generated from myfacecircles.json\n")
        objfile.write("# Format: v x y z r g b\n\n")
        
        for shape in shapes:
            # Get data as x, y, z coordinates
            data = shape.get('data', [])
            if len(data) < 3:
                continue  # Skip if not enough coordinates
            
            x, y, z = data[0], data[1], data[2]
            
            # Get color and discard alpha (4th element), normalize to 0-1 range
            color = shape.get('color', [])
            if len(color) >= 3:
                r = color[0] / 255.0
                g = color[1] / 255.0
                b = color[2] / 255.0
            else:
                r, g, b = 1.0, 1.0, 1.0  # Default white
            
            # Write vertex with color
            objfile.write(f"v {x} {y} {z} {r:.6f} {g:.6f} {b:.6f}\n")
    
    print(f"OBJ saved to: {obj_path}")

def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'myfacecircles.json')
    obj_path = os.path.join(script_dir, 'myfacecircles.obj')
    
    # Load JSON
    print(f"Loading JSON from: {json_path}")
    data = load_json(json_path)
    
    # Manipulate data
    print("Processing data...")
    shapes = manipulate_data(data)
    print(f"Found {len(shapes)} shapes")
    
    # Save to OBJ
    save_to_obj(shapes, obj_path)

if __name__ == "__main__":
    main()
