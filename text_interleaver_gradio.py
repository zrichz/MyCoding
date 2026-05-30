#!/usr/bin/env python3
"""
Text File Interleaver - Gradio version
Interleaves lines from two text files into a single output file.
"""

import gradio as gr


def interleave_files(file1, file2):
    """Interleave lines from two text files."""
    if file1 is None or file2 is None:
        return None, "Please upload both text files."
    
    try:
        # Read both files
        with open(file1.name, 'r', encoding='utf-8') as f:
            lines1 = f.readlines()
        
        with open(file2.name, 'r', encoding='utf-8') as f:
            lines2 = f.readlines()
        
        # Interleave lines
        interleaved = []
        max_len = max(len(lines1), len(lines2))
        
        for i in range(max_len):
            if i < len(lines1):
                interleaved.append(lines1[i])
            if i < len(lines2):
                interleaved.append(lines2[i])
        
        # Write to output file
        output_path = "/tmp/interleaved_output.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(interleaved)
        
        status = f"Interleaved {len(lines1)} lines from file 1 and {len(lines2)} lines from file 2\nTotal output lines: {len(interleaved)}"
        
        return output_path, status
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Text File Interleaver") as demo:
        gr.Markdown("# Text File Interleaver")
        gr.Markdown("Upload two text files to interleave their lines into a single output file.")
        
        with gr.Row():
            with gr.Column():
                file1 = gr.File(
                    label="Text File 1",
                    file_types=[".txt"]
                )
                
                file2 = gr.File(
                    label="Text File 2",
                    file_types=[".txt"]
                )
                
                process_btn = gr.Button("Interleave Files", variant="primary", size="lg")
            
            with gr.Column():
                output_file = gr.File(
                    label="Interleaved Output File"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    lines=3
                )
        
        gr.Markdown("""
        ### How it works:
        - Upload two text files
        - Lines are interleaved: line 1 from file 1, line 1 from file 2, line 2 from file 1, etc.
        - If files have different lengths, all lines from both files are included
        - Download the resulting interleaved text file
        """)
        
        process_btn.click(
            fn=interleave_files,
            inputs=[file1, file2],
            outputs=[output_file, status_output]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, inbrowser=True)
