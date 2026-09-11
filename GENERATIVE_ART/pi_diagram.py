import io
import math
import matplotlib.pyplot as plt
import mpmath
import numpy as np
import gradio as gr
from decimal import Decimal, localcontext
from PIL import Image

# Config
NUM_DIGITS = 200  # Number of digits to generate (more = longer path)
BASE = 10        # Base to convert pi to (2-10)
MAX_DIGITS = 1_000_000
MAX_PLOT_POINTS = 100_000
PATH_COLORS = (
    "#e63946",  # red
    "#2a9d8f",  # teal
    "#457b9d",  # blue
    "#f4a261",  # orange
    "#8338ec",  # violet
    "#ff006e",  # magenta
)

# Display optimized for 1920x1080
# When NUM_DIGITS < 20, digit labels are shown on each line segment

def calculate_pi(precision):
    """Return pi to the requested number of decimal places using mpmath."""
    with mpmath.workdps(precision + 10):
        return mpmath.nstr(mpmath.pi, n=precision + 1, strip_zeros=False)

def decimal_to_base(decimal_str, base, num_digits):
    """
    Convert a decimal number string to a base from 2 through 10.
    Returns the string representation in the target base
    """
    if base < 2 or base > 10:
        raise ValueError("Base must be between 2 and 10")
    
    # Split into integer and fractional parts
    if '.' in decimal_str:
        integer_part, fractional_part = decimal_str.split('.')
    else:
        integer_part, fractional_part = decimal_str, '0'
    
    if base == 10:
        return integer_part + fractional_part[:num_digits]

    # Convert integer part
    integer_val = int(integer_part)
    integer_result = str(integer_val)
    
    # Convert fractional part
    fractional_val = Decimal('0.' + fractional_part)
    fractional_result = []

    for _ in range(num_digits):
        fractional_val *= base
        digit = int(fractional_val)
        fractional_result.append(str(digit))
        fractional_val -= digit
        
        if fractional_val == 0:
            break
    
    return integer_result + ''.join(fractional_result)

def generate_pi_config(base, num_digits):
    """
    Generate pi configuration for a given base and number of digits
    """
    if not 2 <= base <= 10:
        raise ValueError("Base must be between 2 and 10")
    if not 1 <= num_digits <= MAX_DIGITS:
        raise ValueError(f"Number of digits must be between 1 and {MAX_DIGITS:,}")

    decimal_precision = (
        num_digits if base == 10 else math.ceil(num_digits * math.log10(base)) + 10
    )
    pi_decimal = calculate_pi(decimal_precision)
    
    # Convert to string and get the required precision
    pi_str = str(pi_decimal)
    
    # Convert to target base
    pi_base = decimal_to_base(pi_str, base, num_digits)
    
    # Configuration based on base (optimized for 1920x1080 screen)
    configs = {
        2: {"w": 1600, "h": 900, "oX": 800, "oY": 450, "step": 50},
        4: {"w": 1600, "h": 900, "oX": 400, "oY": 450, "step": 40},
        6: {"w": 1600, "h": 900, "oX": 800, "oY": 450, "step": 70},
        8: {"w": 1600, "h": 900, "oX": 200, "oY": 200, "step": 32},
        10: {"w": 1600, "h": 900, "oX": 800, "oY": 200, "step": 60},
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

def path_finder(pi):
    digits = np.fromiter((int(digit) for digit in pi["value"]), dtype=np.float64)
    angles = digits * 2 * np.pi / pi["base"]
    steps = np.column_stack((np.cos(angles), np.sin(angles))) * pi["step"]
    return np.vstack((np.zeros((1, 2)), np.cumsum(steps, axis=0)))

def plot_pi_path(pi_config, path, title=None):
    """
    Plot the pi path with nice formatting
    """
    if title is None:
        title = f'π Path in Base {pi_config["base"]} ({len(pi_config["value"])} digits)'
    
    # Create larger figure optimized for 1920x1080 screen
    fig, ax = plt.subplots(figsize=(16, 9))  # 16:9 aspect ratio for widescreen
    ax.set_facecolor('#f0f0f0')
    
    segment_count = len(path) - 1
    section_indices = np.array_split(np.arange(segment_count), len(PATH_COLORS))
    points_per_section = max(1, MAX_PLOT_POINTS // len(PATH_COLORS))

    for section_number, indices in enumerate(section_indices, start=1):
        if len(indices) == 0:
            continue

        start_index = int(indices[0])
        end_index = int(indices[-1]) + 1
        section_path = path[start_index:end_index + 1]
        plot_step = max(1, math.ceil(len(section_path) / points_per_section))
        plot_section = section_path[::plot_step]
        if not np.array_equal(plot_section[-1], section_path[-1]):
            plot_section = np.vstack((plot_section, section_path[-1]))

        first_digit = start_index + 1
        last_digit = end_index
        label = f"Section {section_number} (digits {first_digit}-{last_digit})"
        color = PATH_COLORS[section_number - 1]
        x_values = plot_section[:, 0] + pi_config["oX"]
        y_values = plot_section[:, 1] + pi_config["oY"]

        ax.plot(x_values, y_values, color=color, linewidth=2, alpha=0.9, label=label)
        ax.scatter(x_values, y_values, color=color, s=15, alpha=0.7)
    
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
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    plt.tight_layout()
    return fig


def generate_pi_diagram(base, num_digits):
    """Generate a PNG diagram and summary text for the Gradio interface."""
    base = int(base)
    num_digits = int(num_digits)

    if not 2 <= base <= 10:
        raise ValueError("Base must be between 2 and 10")
    if not 1 <= num_digits <= MAX_DIGITS:
        raise ValueError(f"Number of digits must be between 1 and {MAX_DIGITS:,}")

    pi_config = generate_pi_config(base, num_digits)
    path = path_finder(pi_config)
    figure = plot_pi_path(pi_config, path)

    image_buffer = io.BytesIO()
    figure.savefig(image_buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(figure)
    image_buffer.seek(0)
    diagram_image = Image.open(image_buffer).convert("RGB")

    summary = (
        f"Generated {len(pi_config['value'])} digits of pi in base {base}.\n"
        f"Sequence: {pi_config['value'][:80]}"
    )
    return diagram_image, summary


def create_interface():
    """Create the interactive pi path diagram interface."""
    with gr.Blocks(
        title="Pi Path Diagram",
        theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("Pi Paths")
        gr.Markdown(
            "Convert pi to another base and trace each digit as a turn in a path. "
            "Short sequences are labeled with the digit that created each segment."
        )

        with gr.Row():
            with gr.Column(scale=1):
                base_input = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=BASE,
                    step=1,
                    label="Base",
                    info="Use bases 2 through 10. Common choices: 4, 6, 8, and 10."
                )
                digits_input = gr.Slider(
                    minimum=1,
                    maximum=MAX_DIGITS,
                    value=NUM_DIGITS,
                    step=1,
                    label="Number of digits",
                    info="Fewer than 20 digits displays segment labels."
                )
                generate_button = gr.Button("Generate Diagram", variant="primary")
                summary_output = gr.Textbox(
                    label="Generation Summary",
                    lines=4,
                    interactive=False
                )

            with gr.Column(scale=2):
                diagram_output = gr.Image(
                    label="Pi Path",
                    type="pil",
                    format="png",
                    height=650,
                    interactive=False
                )

        generate_button.click(
            fn=generate_pi_diagram,
            inputs=[base_input, digits_input],
            outputs=[diagram_output, summary_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(inbrowser=True)
