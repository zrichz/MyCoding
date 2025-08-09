#!/usr/bin/env python3
"""Test script to verify the timestamp filename generation works correctly."""

import sys
import os
import tempfile
from datetime import datetime

# Add the current directory to the path so we can import the expander
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the ImageExpander class
from image_expander_720x1600 import ImageExpander

def test_timestamp_filename_generation():
    """Test the timestamp filename generation method."""
    print("Testing timestamp filename generation...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an instance of ImageExpander (without starting the GUI)
        expander = ImageExpander()
        
        # Test generating filenames
        print(f"Test directory: {temp_dir}")
        
        # Test with different extensions
        for ext in ['.jpg', '.png', '.jpeg']:
            try:
                filename, output_path = expander.generate_timestamp_filename(temp_dir, ext)
                print(f"Generated filename: {filename}")
                print(f"Full path: {output_path}")
                
                # Verify the format
                parts = filename.split('_')
                if len(parts) == 3 and parts[2] == f"720x1600{ext}":
                    timestamp_part = parts[0]
                    counter_part = parts[1]
                    
                    # Check timestamp format (YYYYMMDDHHMMSS)
                    if len(timestamp_part) == 14 and timestamp_part.isdigit():
                        print(f"✓ Timestamp format correct: {timestamp_part}")
                    else:
                        print(f"✗ Timestamp format incorrect: {timestamp_part}")
                    
                    # Check counter format (001-999)
                    if len(counter_part) == 3 and counter_part.isdigit():
                        print(f"✓ Counter format correct: {counter_part}")
                    else:
                        print(f"✗ Counter format incorrect: {counter_part}")
                    
                    # Check that file doesn't exist yet
                    if not os.path.exists(output_path):
                        print(f"✓ File doesn't exist yet (as expected)")
                    else:
                        print(f"✗ File already exists: {output_path}")
                        
                    print()
                else:
                    print(f"✗ Filename format incorrect: {filename}")
                    print()
                    
            except Exception as e:
                print(f"✗ Error generating filename for {ext}: {e}")
                print()
        
        # Test conflict resolution
        print("Testing conflict resolution...")
        try:
            # Generate first filename
            filename1, path1 = expander.generate_timestamp_filename(temp_dir, '.jpg')
            print(f"First filename: {filename1}")
            
            # Create the file to simulate conflict
            with open(path1, 'w') as f:
                f.write("test")
            
            # Generate second filename (should increment counter)
            filename2, path2 = expander.generate_timestamp_filename(temp_dir, '.jpg')
            print(f"Second filename: {filename2}")
            
            # Check that filenames are different
            if filename1 != filename2:
                print("✓ Conflict resolution working - generated different filenames")
                
                # Check that counter incremented
                parts1 = filename1.split('_')
                parts2 = filename2.split('_')
                
                if len(parts1) == 3 and len(parts2) == 3:
                    counter1 = int(parts1[1])
                    counter2 = int(parts2[1])
                    
                    if counter2 == counter1 + 1:
                        print(f"✓ Counter incremented correctly: {counter1:03d} -> {counter2:03d}")
                    else:
                        print(f"✗ Counter didn't increment correctly: {counter1:03d} -> {counter2:03d}")
                else:
                    print("✗ Filename format issue in conflict test")
            else:
                print("✗ Conflict resolution failed - generated same filename")
                
        except Exception as e:
            print(f"✗ Error in conflict resolution test: {e}")
    
    print("\nTimestamp filename generation test completed!")

if __name__ == "__main__":
    test_timestamp_filename_generation()
