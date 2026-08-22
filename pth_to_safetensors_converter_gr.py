"""
PTH to SafeTensors Converter
Converts PyTorch .pth pickle files to the safer .safetensors format.
Uses Gradio interface for easy file conversion.
"""

import os
import gradio as gr
import torch
from safetensors.torch import save_file


def convert_pth_to_safetensors(pth_path, use_safe_load=True):
    """
    Convert a PyTorch .pth file to safetensors format.
    
    Args:
        pth_path: Path to the input .pth file
        use_safe_load: If True, use weights_only=True for safer loading
    
    Returns:
        tuple: (output_path, status_message)
    """
    if not pth_path:
        return None, "Error: No file selected"
    
    if not os.path.exists(pth_path):
        return None, f"Error: File not found: {pth_path}"
    
    if not pth_path.endswith('.pth'):
        return None, "Error: Input file must have .pth extension"
    
    # Generate output path
    base_name = os.path.splitext(pth_path)[0]
    output_path = f"{base_name}.safetensors"
    
    status_messages = []
    status_messages.append(f"Loading {os.path.basename(pth_path)}...")
    
    # Load the .pth file
    try:
        if use_safe_load:
            status_messages.append("Using safe load mode (weights_only=True)")
            pth_model = torch.load(pth_path, map_location="cpu", weights_only=True)
        else:
            status_messages.append("Warning: Using standard load mode. Only do this if you trust the source")
            pth_model = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if use_safe_load:
            return None, f"Safe load failed: {str(e)}\n\nTry again with 'Use Safe Load' unchecked if you trust the source."
        else:
            return None, f"Load failed: {str(e)}"
    
    # Extract state dictionary
    try:
        if isinstance(pth_model, dict):
            # Check for common wrapper keys
            if "state_dict" in pth_model:
                state_dict = pth_model["state_dict"]
                status_messages.append("Extracted state_dict from 'state_dict' key")
            elif "model" in pth_model:
                state_dict = pth_model["model"]
                status_messages.append("Extracted state_dict from 'model' key")
            elif "generator" in pth_model:
                state_dict = pth_model["generator"]
                status_messages.append("Extracted state_dict from 'generator' key")
            else:
                state_dict = pth_model
                status_messages.append("Using root dictionary as state_dict")
        else:
            # If it loaded as a raw model object
            if hasattr(pth_model, "state_dict"):
                state_dict = pth_model.state_dict()
                status_messages.append("Extracted state_dict from model object")
            else:
                return None, "Error: Could not find a valid state_dict or model object structure"
    except Exception as e:
        return None, f"Error extracting state_dict: {str(e)}"
    
    # Clean up tensors
    status_messages.append(f"Processing {len(state_dict)} tensors...")
    cleaned_state_dict = {}
    tensor_count = 0
    
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            cleaned_state_dict[str(k)] = v.contiguous()
            tensor_count += 1
    
    if tensor_count == 0:
        return None, "Error: No tensors found in the state dictionary"
    
    status_messages.append(f"Cleaned {tensor_count} tensors")
    
    # Save to safetensors format
    try:
        status_messages.append(f"Saving to {os.path.basename(output_path)}...")
        save_file(cleaned_state_dict, output_path)
        status_messages.append(f"Success. Conversion complete")
        status_messages.append(f"Output saved to: {output_path}")
        
        # Calculate file sizes
        input_size = os.path.getsize(pth_path) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        status_messages.append(f"Input size: {input_size:.2f} MB")
        status_messages.append(f"Output size: {output_size:.2f} MB")
        
        return output_path, "\n".join(status_messages)
    
    except Exception as e:
        return None, f"Error saving safetensors file: {str(e)}"


def convert_interface(input_file, use_safe_load):
    """Wrapper function for Gradio interface"""
    if input_file is None:
        return None, "Please select a .pth file to convert"
    
    output_path, message = convert_pth_to_safetensors(input_file.name, use_safe_load)
    
    if output_path and os.path.exists(output_path):
        return output_path, message
    else:
        return None, message


# Create Gradio interface
with gr.Blocks(title="PTH to SafeTensors Converter") as demo:
    gr.Markdown("# PTH to SafeTensors Converter")
    gr.Markdown(
        "Convert PyTorch .pth pickle files to the safer .safetensors format. "
        "This protects against malicious code execution hidden in pickle files."
    )
    
    with gr.Row():
        with gr.Column():
            input_file = gr.File(
                label="Select .pth File",
                file_types=[".pth"],
                type="filepath"
            )
            
            use_safe_load = gr.Checkbox(
                label="Use Safe Load (weights_only=True)",
                value=True,
                info="Recommended: Blocks malicious code execution. Uncheck only if you trust the source."
            )
            
            convert_btn = gr.Button("Convert to SafeTensors", variant="primary")
        
        with gr.Column():
            output_file = gr.File(label="Output SafeTensors File")
            status_output = gr.Textbox(
                label="Conversion Status",
                lines=10,
                max_lines=15
            )
    
    gr.Markdown("### Notes")
    gr.Markdown(
        "- Safe Load mode (recommended) prevents execution of malicious code in .pth files\n"
        "- If Safe Load fails, you can retry with it disabled, but only for trusted files\n"
        "- The output .safetensors file will be created in the same directory as the input\n"
        "- SafeTensors format is faster to load and safer than pickle-based .pth files"
    )
    
    # Connect the button
    convert_btn.click(
        fn=convert_interface,
        inputs=[input_file, use_safe_load],
        outputs=[output_file, status_output]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
