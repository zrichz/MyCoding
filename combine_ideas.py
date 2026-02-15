#!/usr/bin/env python3
"""
Combine ideas from ideas.txt into groups of 5.
Reads ideas.txt (one phrase per line) and creates five_ideas.txt
where every 5 lines are combined into one comma-separated line.
"""

def combine_ideas(input_file='ideas.txt', output_file='five_ideas.txt', group_size=5):
    """Combine lines from input file into groups."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        combined_lines = []
        for i in range(0, len(lines), group_size):
            group = lines[i:i+group_size]
            combined = ', '.join(group)
            combined_lines.append(combined)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(combined_lines))
        
        print(f"✓ Processed {len(lines)} lines from {input_file}")
        print(f"✓ Created {len(combined_lines)} combined lines in {output_file}")
        
    except FileNotFoundError:
        print(f"Error: {input_file} not found")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    import sys
    exit(combine_ideas())
