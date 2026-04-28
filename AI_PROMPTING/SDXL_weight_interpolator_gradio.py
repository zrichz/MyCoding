import gradio as gr
import numpy as np


def generate_weight_strings(string1, w1_1, w2_1, steps_1, string2, w1_2, w2_2, steps_2):
    """
    Generate two interpolated weight strings based on input parameters.
    
    Args:
        string1: First string
        w1_1: Starting weight for string 1
        w2_1: Ending weight for string 1
        steps_1: Number of steps for string 1
        string2: Second string
        w1_2: Starting weight for string 2
        w2_2: Ending weight for string 2
        steps_2: Number of steps for string 2
    
    Returns:
        Tuple of two formatted strings
    """
    # Generate interpolated weights for string 1
    weights_1 = np.linspace(w1_1, w2_1, steps_1)
    parts_1 = [f"{string1}:{weight:.2f}" for weight in weights_1]
    result_1 = ", ".join(parts_1)
    
    # Generate interpolated weights for string 2
    weights_2 = np.linspace(w1_2, w2_2, steps_2)
    parts_2 = [f"{string2}:{weight:.2f}" for weight in weights_2]
    result_2 = ", ".join(parts_2)
    
    return result_1, result_2


# Create Gradio interface
with gr.Blocks(title="Weight Interpolator") as demo:
    gr.Markdown("# Weight Interpolator")
    gr.Markdown("Enter two sets of parameters to generate interpolated weight strings")
    
    with gr.Row():
        # First set of inputs
        with gr.Column():
            gr.Markdown("### Set 1")
            string1 = gr.Textbox(label="String 1", placeholder="Enter first string")
            w1_1 = gr.Slider(minimum=0.00, maximum=1.00, step=0.05, value=0.00, 
                           label="W1 (Starting Weight)")
            w2_1 = gr.Slider(minimum=0.00, maximum=1.00, step=0.05, value=1.00, 
                           label="W2 (Ending Weight)")
            steps_1 = gr.Slider(minimum=2, maximum=10, step=1, value=5, 
                              label="Steps")
        
        # Second set of inputs
        with gr.Column():
            gr.Markdown("### Set 2")
            string2 = gr.Textbox(label="String 2", placeholder="Enter second string")
            w1_2 = gr.Slider(minimum=0.00, maximum=1.00, step=0.05, value=0.00, 
                           label="W1 (Starting Weight)")
            w2_2 = gr.Slider(minimum=0.00, maximum=1.00, step=0.05, value=1.00, 
                           label="W2 (Ending Weight)")
            steps_2 = gr.Slider(minimum=2, maximum=10, step=1, value=5, 
                              label="Steps")
    
    generate_btn = gr.Button("Generate Weight Strings", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Output 1")
            output1 = gr.Textbox(label="String 1 Interpolation", lines=3)
        
        with gr.Column():
            gr.Markdown("### Output 2")
            output2 = gr.Textbox(label="String 2 Interpolation", lines=3)
    
    # Connect the button to the function
    generate_btn.click(
        fn=generate_weight_strings,
        inputs=[string1, w1_1, w2_1, steps_1, string2, w1_2, w2_2, steps_2],
        outputs=[output1, output2]
    )
    
    # Example
    gr.Markdown("### Example")
    gr.Markdown("String: 'cat', W1: 0.50, W2: 0.90, Steps: 3 → Output: `cat:0.50, cat:0.70, cat:0.90`")


if __name__ == "__main__":
    demo.launch()
