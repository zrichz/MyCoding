import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, getcontext

# Configuration - Modify these values to experiment
NUM_DIGITS = 200  # Number of pi digits to generate (more digits = longer path)
BASE = 10        # Base to convert pi to (2-36, try 4, 6, 8, 10, 16)

# Display optimized for 1920x1080 screen with 16:9 aspect ratio
# Debug feature: When NUM_DIGITS < 20, digit labels are shown on each line segment

# Popular bases to try:
# BASE = 4   # Quaternary (0,1,2,3)
# BASE = 6   # Senary (0,1,2,3,4,5) 
# BASE = 8   # Octal (0,1,2,3,4,5,6,7)
# BASE = 10  # Decimal (0,1,2,3,4,5,6,7,8,9)
# BASE = 16  # Hexadecimal (0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F)

# Set high precision for decimal calculations
getcontext().prec = NUM_DIGITS + 20

def calculate_pi(precision):
    """
    Calculate pi using the Bailey–Borwein–Plouffe formula (simpler than Chudnovsky)
    Returns pi as a Decimal with the specified number of decimal places
    """
    getcontext().prec = precision + 20
    
    pi_sum = Decimal(0)
    
    # BBP formula: π = Σ(k=0 to ∞) [1/16^k * (4/(8k+1) - 2/(8k+4) - 1/(8k+5) - 1/(8k+6))]
    for k in range(precision + 10):  # More iterations for higher precision
        term1 = Decimal(4) / (8*k + 1)
        term2 = Decimal(2) / (8*k + 4) 
        term3 = Decimal(1) / (8*k + 5)
        term4 = Decimal(1) / (8*k + 6)
        
        term = (term1 - term2 - term3 - term4) / (Decimal(16) ** k)
        pi_sum += term
        
        # Early termination if term becomes negligible
        if abs(term) < Decimal(10) ** (-precision - 5):
            break
    
    return pi_sum

def decimal_to_base(decimal_str, base, num_digits):
    """
    Convert a decimal number string to any base (2-36)
    Returns the string representation in the target base
    """
    if base < 2 or base > 36:
        raise ValueError("Base must be between 2 and 36")
    
    # Split into integer and fractional parts
    if '.' in decimal_str:
        integer_part, fractional_part = decimal_str.split('.')
    else:
        integer_part, fractional_part = decimal_str, '0'
    
    # Convert integer part
    integer_val = int(integer_part)
    if integer_val == 0:
        integer_result = '0'
    else:
        digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        integer_result = ''
        while integer_val > 0:
            integer_result = digits[integer_val % base] + integer_result
            integer_val //= base
    
    # Convert fractional part
    fractional_val = Decimal('0.' + fractional_part)
    fractional_result = ''
    digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for _ in range(num_digits):
        fractional_val *= base
        digit = int(fractional_val)
        fractional_result += digits[digit]
        fractional_val -= digit
        
        if fractional_val == 0:
            break
    
    return integer_result + fractional_result

def generate_pi_config(base, num_digits):
    """
    Generate pi configuration for a given base and number of digits
    """
    # Calculate pi with high precision
    pi_decimal = calculate_pi(num_digits + 10)
    
    # Convert to string and get the required precision
    pi_str = str(pi_decimal)
    
    # Convert to target base
    pi_base = decimal_to_base(pi_str, base, num_digits)
    
    # Configuration based on base (optimized for 1920x1080 screen)
    configs = {
        4: {"w": 1600, "h": 900, "oX": 400, "oY": 450, "step": 40},
        6: {"w": 1600, "h": 900, "oX": 800, "oY": 450, "step": 70},
        8: {"w": 1600, "h": 900, "oX": 200, "oY": 200, "step": 32},
        10: {"w": 1600, "h": 900, "oX": 800, "oY": 200, "step": 60},
        16: {"w": 1600, "h": 900, "oX": 800, "oY": 450, "step": 50},
    }
    
    # Use default config for unlisted bases
    default_config = {"w": 1600, "h": 900, "oX": 800, "oY": 450, "step": 50}
    config = configs.get(base, default_config)
    
    return {
        "base": base,
        "value": pi_base,
        "w": config["w"],
        "h": config["h"],
        "oX": config["oX"],
        "oY": config["oY"],
        "step": config["step"]
    }

# Generate pi configuration
pi = generate_pi_config(BASE, NUM_DIGITS)
print(f"Generated π in base {BASE} with {len(pi['value'])} digits: {pi['value'][:50]}...")

def path_finder(pi):
    x, y = 0, 0
    path = [(x, y)]
    for digit in pi["value"]:
        n = int(digit)
        r = n * 2 * np.pi / pi["base"]
        x += pi["step"] * np.cos(r)
        y += pi["step"] * np.sin(r)
        path.append((x, y))
    return np.array(path)

def plot_pi_path(pi_config, path, title=None):
    """
    Plot the pi path with nice formatting
    """
    if title is None:
        title = f'π Path in Base {pi_config["base"]} ({len(pi_config["value"])} digits)'
    
    # Create larger figure optimized for 1920x1080 screen
    fig, ax = plt.subplots(figsize=(16, 9))  # 16:9 aspect ratio for widescreen
    ax.set_facecolor('#f0f0f0')
    
    # Plot the path
    ax.plot(path[:, 0] + pi_config["oX"], path[:, 1] + pi_config["oY"], 
            color='gray', linewidth=2, alpha=0.8)
    
    # Add dots along the path (smaller for larger display)
    ax.scatter(path[:, 0] + pi_config["oX"], path[:, 1] + pi_config["oY"], 
               color='#f0f0f0', s=15, alpha=0.6)
    
    # Add digit labels at line centers for debugging (only when digits < 20)
    if len(pi_config["value"]) < 20:
        for i in range(len(path) - 1):
            # Calculate center point of each line segment
            start_x = path[i, 0] + pi_config["oX"]
            start_y = path[i, 1] + pi_config["oY"]
            end_x = path[i + 1, 0] + pi_config["oX"]
            end_y = path[i + 1, 1] + pi_config["oY"]
            
            center_x = (start_x + end_x) / 2
            center_y = (start_y + end_y) / 2
            
            # Get the digit that created this line segment
            digit = pi_config["value"][i]
            
            # Add text label at the center of the line
            ax.text(center_x, center_y, digit, fontsize=12, fontweight='bold',
                   ha='center', va='center', color='red', 
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', alpha=0.8))
    
    # Highlight start and end points (smaller markers)
    ax.scatter(path[0, 0] + pi_config["oX"], path[0, 1] + pi_config["oY"], 
               color='red', s=40, label='Start', zorder=5)
    ax.scatter(path[-1, 0] + pi_config["oX"], path[-1, 1] + pi_config["oY"], 
               color='blue', s=40, label='End', zorder=5)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, pad=20)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Generate and plot the pi path
path = path_finder(pi)
plot_pi_path(pi, path)

def test_different_bases():
    """
    Function to easily test different bases - uncomment to use
    """
    bases_to_test = [4, 6, 8, 10, 16]
    
    for base in bases_to_test:
        print(f"\n=== Testing Base {base} ===")
        pi_config = generate_pi_config(base, 50)  # Use 50 digits for quick testing
        print(f"π in base {base}: {pi_config['value'][:30]}...")
        
        # You can plot each one by uncommenting the lines below:
        # path = path_finder(pi_config)
        # plot_pi_path(pi_config, path, f"Pi in Base {base}")

# Uncomment the line below to test all bases at once:
# test_different_bases()
