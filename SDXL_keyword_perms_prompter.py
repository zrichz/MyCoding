"""
Keyword Permutation Prompt Generator
Generates all 16 permutations (combinations) of 4 keywords with prefix and suffix.
Based on SDXL_combined_prompter.py

Author: GitHub Copilot
Date: 2026-02-15
"""

import gradio as gr
from itertools import combinations
from datetime import datetime


def generate_all_permutations(prefix, keyword1, keyword2, keyword3, keyword4, suffix):
    """
    Generate all 16 combinations of 4 keywords (all possible subsets).
    
    The 16 combinations are:
    - Empty set (just prefix + suffix)
    - 4 single keywords
    - 6 pairs (C(4,2))
    - 4 triples (C(4,3))
    - 1 quadruple (all 4)
    """
    keywords = [keyword1, keyword2, keyword3, keyword4]
    prompts = []
    
    # Generate all possible combinations (empty set through all 4)
    for r in range(5):  # 0, 1, 2, 3, 4 elements
        for combo in combinations(keywords, r):
            # Build the prompt
            parts = []
            if prefix.strip():
                parts.append(prefix.strip())
            
            if combo:  # If there are keywords in this combination
                parts.append(", ".join(combo))
            
            if suffix.strip():
                parts.append(suffix.strip())
            
            # Join with ", " separator
            prompt = ", ".join(parts)
            prompts.append(prompt)
    
    return prompts


def generate_and_display(prefix, kw1, kw2, kw3, kw4, suffix):
    """Generate all 16 permutations and format for display."""
    # Validate that all keywords are provided
    if not all([kw1.strip(), kw2.strip(), kw3.strip(), kw4.strip()]):
        return "⚠️ Please provide all 4 keywords", []
    
    # Generate prompts
    prompts = generate_all_permutations(prefix, kw1, kw2, kw3, kw4, suffix)
    
    # Format output with negative prompt (unchanged from combined prompter)
    output_lines = []
    for i, prompt in enumerate(prompts, 1):
        output_lines.append(f'{i}. --prompt "{prompt}" --negative_prompt "asian, poor quality"')
    
    output = "\n\n".join(output_lines)
    
    return output, prompts


def save_prompts(prompts_data):
    """Save prompts to file."""
    if not prompts_data:
        return "No prompts to save. Generate prompts first."
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"keyword_permutation_prompts_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for prompt in prompts_data:
            f.write(f'--prompt "{prompt}" --negative_prompt "asian, poor quality"\n')
    
    return f"✓ Saved {len(prompts_data)} prompts to {filename}"


# Build Gradio interface
with gr.Blocks(title="Keyword Permutation Prompter") as demo:
    gr.Markdown("# Keyword Permutation Prompt Generator")
    gr.Markdown("Generate all **16 combinations** of 4 keywords with prefix and suffix.")
    gr.Markdown("Format: `prefix, <keyword combinations>, suffix`")
    gr.Markdown("Negative prompt: `asian, poor quality` (unchanged from SDXL Combined Prompter)")
    
    with gr.Row():
        with gr.Column():
            prefix_input = gr.Textbox(
                label="Prefix",
                placeholder="e.g., A photo of",
                lines=2
            )
            
            gr.Markdown("### Keywords (all 4 required)")
            keyword1 = gr.Textbox(
                label="Keyword 1",
                placeholder="e.g., sunset"
            )
            keyword2 = gr.Textbox(
                label="Keyword 2",
                placeholder="e.g., ocean"
            )
            keyword3 = gr.Textbox(
                label="Keyword 3",
                placeholder="e.g., palm trees"
            )
            keyword4 = gr.Textbox(
                label="Keyword 4",
                placeholder="e.g., golden hour"
            )
            
            suffix_input = gr.Textbox(
                label="Suffix",
                placeholder="e.g., detailed",
                lines=2
            )
            
            generate_btn = gr.Button("Generate 16 Prompts", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Generated Prompts (16 total)",
                lines=30,
                max_lines=40,
                interactive=False
            )
    
    with gr.Row():
        save_btn = gr.Button("Save to File", size="sm")
        save_status = gr.Textbox(label="Save Status", interactive=False, scale=3)
    
    # Hidden state to store prompts for saving
    prompts_state = gr.State([])
    
    # Wire up interactions
    generate_btn.click(
        fn=generate_and_display,
        inputs=[prefix_input, keyword1, keyword2, keyword3, keyword4, suffix_input],
        outputs=[output_text, prompts_state]
    )
    
    save_btn.click(
        fn=save_prompts,
        inputs=[prompts_state],
        outputs=[save_status]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
