#!/usr/bin/env python3
"""
Generate .data files containing random numbers.
Creates files with 1024 * 2^n numbers.
line 45 determines the range of n.
"""

import random
import os
from pathlib import Path


def generate_data_file(num_count, exponent, output_dir="data_files"):
    """
    Generate a .data file with random numbers.
    
    Args:
        num_count: Number of random numbers to generate
        exponent: The exponent n (for 2^n)
        output_dir: Directory to save the file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate filename
    filename = os.path.join(output_dir, f"log2pow_{exponent:04d}.data")
    
    print(f"Generating {num_count:,} random numbers -> {filename}")
    
    # Generate random numbers and write to file
    with open(filename, 'w') as f:
        for i in range(num_count):
            # Generate random float between 0 and 1
            f.write(f"{random.random()}\n")
    
    print(f"  Completed: {filename}")


def main():
    """Generate all data files in the sequence 2^n where n ranges from 0 to 10"""
    
    print("Random Data File Generator")
    print("=" * 50)
    
    # Generate files for n = 0 to 10
    for n in range(11):  # 0 to 10 inclusive
        num_count = 2 ** n
        generate_data_file(num_count, n)
    
    print("=" * 50)
    print(f"Generated {11} files")
    print(f"Total numbers across all files: {sum(2**i for i in range(11)):,}")


if __name__ == "__main__":
    main()
