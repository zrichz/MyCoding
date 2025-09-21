#!/usr/bin/env python3
"""
Combination Generator Script
Generates all k-combinations of an n-element set based on user input.

Usage:
1. Enter a base phrase (e.g., "8vSw-0600")
2. Enter an integer n
3. Script generates all combinations from k=1 to k=n
4. Outputs to a .txt file with comma-separated values

Example:
Input: "8vSw-0600", n=3, k_min=1
Output: "8vSw-0600_vector_01.pt","8vSw-0600_vector_02.pt","8vSw-0600_vector_03.pt",
        "8vSw-0600_vector_01.pt 8vSw-0600_vector_02.pt","8vSw-0600_vector_01.pt 8vSw-0600_vector_03.pt",
        "8vSw-0600_vector_02.pt 8vSw-0600_vector_03.pt","8vSw-0600_vector_01.pt 8vSw-0600_vector_02.pt 8vSw-0600_vector_03.pt"

Input: "8vSw-0600", n=3, k_min=2  
Output: "8vSw-0600_vector_01.pt 8vSw-0600_vector_02.pt","8vSw-0600_vector_01.pt 8vSw-0600_vector_03.pt",
        "8vSw-0600_vector_02.pt 8vSw-0600_vector_03.pt","8vSw-0600_vector_01.pt 8vSw-0600_vector_02.pt 8vSw-0600_vector_03.pt"
"""

import itertools
import os


def generate_combinations(base_phrase, n, k_min=1):
    """
    Generate all k-combinations for k=k_min to n of the n-element set
    
    Args:
        base_phrase: Base phrase like "8vSw-0600"
        n: Integer representing the size of the set
        k_min: Minimum combination size (default: 1)
    
    Returns:
        List of all combination strings
    """
    # Generate the base elements (e.g., 8vSw-0600_vector_01.pt, 8vSw-0600_vector_02.pt, etc.)
    elements = [f"{base_phrase}_vector_{i+1:02d}.pt" for i in range(n)]
    
    print(f"\nGenerated base elements: {elements}")
    
    all_combinations = []
    
    # Generate combinations for each k from k_min to n
    for k in range(k_min, n + 1):
        combinations = list(itertools.combinations(elements, k))
        
        print(f"\nCombinations of size {k} ({len(combinations)} total):")
        
        for combo in combinations:
            if k == 1:
                # For single elements, just use the element itself
                combo_str = combo[0]
            else:
                # For multiple elements, join with space
                combo_str = " ".join(combo)
            
            all_combinations.append(combo_str)
            print(f"  {combo_str}")
    
    return all_combinations


def save_to_file(combinations, base_phrase, n, k_min=1):
    """
    Save combinations to a comma-separated .txt file
    
    Args:
        combinations: List of combination strings
        base_phrase: Base phrase used for filename
        n: Integer n for filename
        k_min: Minimum combination size for filename
    """
    # Create filename
    safe_phrase = base_phrase.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if k_min == 1:
        filename = f"{safe_phrase}_n{n}_combinations.txt"
    else:
        filename = f"{safe_phrase}_n{n}_k{k_min}-{n}_combinations.txt"
    
    # Create output directory if it doesn't exist
    output_dir = "combination_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, filename)
    
    # Create comma-separated string with quotes around each combination
    quoted_combinations = [f'"{combo}"' for combo in combinations]
    output_content = ",".join(quoted_combinations)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(output_content)
    
    print(f"\n✅ Combinations saved to: {filepath}")
    print(f"📄 Total combinations: {len(combinations)}")
    print(f"📊 File size: {len(output_content)} characters")
    
    return filepath


def main():
    """Main function to execute the combination generation workflow"""
    
    print("=" * 60)
    print("COMBINATION GENERATOR")
    print("=" * 60)
    print("Generate all k-combinations of an n-element set")
    print("Output format: comma-separated quoted strings in .txt file")
    print("=" * 60)
    
    # Get user input for base phrase
    while True:
        base_phrase = input("\nEnter base phrase (e.g., '8vSw-0600'): ").strip()
        if base_phrase:
            break
        print("Please enter a valid phrase.")
    
    # Get user input for n
    while True:
        try:
            n = int(input(f"Enter integer n (number of elements): ").strip())
            if n > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    # Get user input for k_min
    while True:
        try:
            k_min_input = input(f"Enter minimum k (1-{n}, press Enter for 1): ").strip()
            if k_min_input == "":
                k_min = 1
                break
            k_min = int(k_min_input)
            if 1 <= k_min <= n:
                break
            else:
                print(f"Please enter an integer between 1 and {n}.")
        except ValueError:
            print("Please enter a valid integer or press Enter for default.")
    
    print(f"\n📝 Configuration:")
    print(f"   Base phrase: '{base_phrase}'")
    print(f"   Set size (n): {n}")
    print(f"   Minimum combination size (k): {k_min}")
    print(f"   Will generate combinations for k = {k_min} to {n}")
    
    # Calculate total number of combinations
    total_combinations = sum(len(list(itertools.combinations(range(n), k))) for k in range(k_min, n + 1))
    print(f"   Total combinations to generate: {total_combinations}")
    
    # Confirm with user
    proceed = input(f"\nProceed with generation? (y/n): ").lower().strip()
    if proceed not in ['y', 'yes']:
        print("Generation cancelled.")
        return
    
    # Generate combinations
    print(f"\n🔄 Generating combinations...")
    combinations = generate_combinations(base_phrase, n, k_min)
    
    # Save to file
    print(f"\n💾 Saving to file...")
    filepath = save_to_file(combinations, base_phrase, n, k_min)
    
    # Show sample of output
    print(f"\n📋 Sample output (first 3 combinations):")
    for i, combo in enumerate(combinations[:3]):
        print(f"   {i+1}: \"{combo}\"")
    
    if len(combinations) > 3:
        print(f"   ... and {len(combinations) - 3} more")
    
    print(f"\n✅ Generation complete!")
    print(f"📁 Output file: {filepath}")
    
    # Show file content preview
    print(f"\n🔍 File content preview (first 200 characters):")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        preview = content[:200]
        if len(content) > 200:
            preview += "..."
        print(f"   {preview}")


if __name__ == "__main__":
    main()
