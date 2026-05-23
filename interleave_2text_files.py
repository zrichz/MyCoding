#!/home/rich/MyCoding/venvmycoding313/bin/python
"""
Text File Interleaver - Gradio GUI
Interleaves lines from two text files alternately.
Output format: line1_file1, line1_file2, line2_file1, line2_file2, etc.
"""

import gradio as gr
import tempfile
import os
from pathlib import Path


def interleave_files(file1, file2, output_filename):
    """
    Interleave lines from two uploaded text files.
    
    Args:
        file1: Gradio file object for first input file
        file2: Gradio file object for second input file
        output_filename: Desired output filename
    
    Returns:
        Tuple of (output_file_path, status_message)
    """
    if file1 is None or file2 is None:
        return None, "Error: Please upload both files"
    
    if not output_filename:
        output_filename = "interleaved_output.txt"
    
    # Ensure .txt extension
    if not output_filename.endswith('.txt'):
        output_filename += '.txt'
    
    try:
        # Read both files
        with open(file1.name, 'r', encoding='utf-8') as f1:
            lines1 = f1.readlines()
        
        with open(file2.name, 'r', encoding='utf-8') as f2:
            lines2 = f2.readlines()
        
        # Create output file
        output_path = os.path.join(tempfile.gettempdir(), output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as out:
            max_lines = max(len(lines1), len(lines2))
            
            for i in range(max_lines):
                if i < len(lines1):
                    out.write(lines1[i])
                if i < len(lines2):
                    out.write(lines2[i])
        
        status = (f"Successfully interleaved {len(lines1)} lines from file 1 "
                 f"and {len(lines2)} lines from file 2\n"
                 f"Total output lines: {len(lines1) + len(lines2)}")
        
        return output_path, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Text File Interleaver")
    gr.Markdown("Upload two text files to interleave their lines alternately")
    gr.Markdown("Output format: line 1 from file 1, line 1 from file 2, line 2 from file 1, line 2 from file 2, etc.")
    
    with gr.Row():
        with gr.Column():
            file1_input = gr.File(
                label="File 1",
                file_types=[".txt"],
                type="filepath"
            )
        
        with gr.Column():
            file2_input = gr.File(
                label="File 2",
                file_types=[".txt"],
                type="filepath"
            )
    
    output_name_input = gr.Textbox(
        label="Output Filename",
        value="interleaved_output.txt",
        placeholder="Enter output filename"
    )
    
    process_btn = gr.Button("Interleave Files", variant="primary", size="lg")
    
    status_output = gr.Textbox(
        label="Status",
        interactive=False,
        lines=3
    )
    
    file_output = gr.File(
        label="Download Interleaved File"
    )
    
    # Wire up the button
    process_btn.click(
        fn=interleave_files,
        inputs=[file1_input, file2_input, output_name_input],
        outputs=[file_output, status_output]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
