import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import gradio as gr
import os

# Load the BLIP model and processor with CPU fallback
print("Loading BLIP model...")

# Load processor and model on CPU first
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Try to use CUDA, but fall back to CPU if there are any issues
device = "cpu"  # Default to CPU
if torch.cuda.is_available():
    try:
        print("Attempting to move model to CUDA...")
        model = model.to("cuda")  # type: ignore
        # Test with actual inference to catch kernel errors
        test_tensor = torch.zeros(1, 3, 224, 224).cuda()
        _ = model.vision_model(test_tensor)
        device = "cuda"
        print("✓ CUDA is working - using GPU")
    except Exception as e:
        print(f"✗ CUDA error detected: {str(e)[:100]}")
        print("→ Falling back to CPU")
        model = model.to("cpu")  # type: ignore
        device = "cpu"
else:
    print("CUDA not available - using CPU")
    model = model.to("cpu")  # type: ignore

print(f"Final device: {device}")
print("BLIP model loaded successfully!")

# Function to generate captions using BLIP
def generate_caption_blip(image, max_length, min_length, num_beams, length_penalty):
    global device, model
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore
        out = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True
        )
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except RuntimeError as e:
        if "CUDA" in str(e) and device == "cuda":
            print(f"CUDA runtime error detected: {str(e)[:100]}")
            print("Switching to CPU for all future operations...")
            device = "cpu"
            model = model.to("cpu")  # type: ignore
            # Retry on CPU
            inputs = processor(images=image, return_tensors="pt").to(device)  # type: ignore
            out = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=True
            )
            caption = processor.decode(out[0], skip_special_tokens=True)
            return caption
        else:
            raise

# Store generated captions globally for saving
current_captions = []
current_image_path = None

# Function to handle file upload and set original path
def handle_file_upload(file_obj):
    if file_obj is None:
        return ""
    # Get the original filename from the temp path
    temp_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    original_name = os.path.basename(temp_path)
    # Return a suggested path (user can edit)
    return f"~/images_general/{original_name}"

# Main function for Gradio
def generate_captions(file_obj):
    global current_captions, current_image_path
    current_captions = []
    current_image_path = None
    
    if file_obj is None:
        yield None, "", gr.update(choices=[], visible=False), gr.update(value="", visible=False)
        return
    
    try:
        # Store the original file path for later saving
        current_image_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        # Load and convert image
        image = Image.open(current_image_path).convert("RGB")
        
        results = []
        results.append(f"Using device: {device}\n")
        results.append("=" * 60 + "\n\n")
        yield image, "".join(results), gr.update(choices=[], visible=False), gr.update(value="", visible=False)
        
        captions_list = []
        # Generate captions with increasing lengths (skip first 3 iterations)
        for i in range(3, 6):  # Start from i=3 (only last 3 iterations)
            max_length = (i + 1) * 8
            min_length = (i + 1) * 4
            num_beams = 8
            length_penalty = 1.25
            
            caption = generate_caption_blip(image, max_length=max_length, min_length=min_length, 
                                          num_beams=num_beams, length_penalty=length_penalty)
            
            captions_list.append(caption)
            results.append(f"Caption {i - 2} (max_len={max_length}, min_len={min_length}):\n")
            results.append(f"{caption}\n\n")
            
            # Yield progressive results
            yield image, "".join(results), gr.update(choices=[], visible=False), gr.update(value="", visible=False)
        
        # Store captions for selection
        current_captions = captions_list
        
        # Show radio buttons with captions
        caption_choices = [f"Caption {i+1}: {cap}" for i, cap in enumerate(captions_list)]
        yield image, "".join(results), gr.update(choices=caption_choices, value=None, visible=True), gr.update(value="", visible=True)
        
    except Exception as e:
        yield None, f"Error: {str(e)}", gr.update(choices=[], visible=False), gr.update(value="", visible=False)

# Function to save selected caption
def save_caption(selected_caption_text, original_path):
    global current_captions
    
    if not original_path:
        return "⚠ Please specify the original image path."
    
    if not selected_caption_text:
        return "Please select a caption first."
    
    try:
        # Extract caption index from selection
        caption_idx = int(selected_caption_text.split(":")[0].replace("Caption ", "")) - 1
        selected_caption = current_captions[caption_idx]
        
        # Expand user path (~/... -> /home/user/...)
        expanded_path = os.path.expanduser(original_path)
        
        # Create .txt file path in same directory as original image
        base_path = os.path.splitext(expanded_path)[0]
        txt_path = base_path + ".txt"
        
        # Save caption
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(selected_caption)
        
        return f"✓ Caption saved to: {txt_path}"
        
    except Exception as e:
        return f"Error saving caption: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="BLIP Image Captioner") as demo:
    gr.Markdown("# BLIP Image Captioner")
    gr.Markdown(f"**Current Device:** {device}")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.File(
                label="Upload Image File",
                file_types=["image"]
            )
            original_path_input = gr.Textbox(
                label="Original Image Path (edit if needed)",
                placeholder="e.g., /home/rich/MyCoding/images_general/image.jpg",
                info="Caption will be saved as .txt in same directory"
            )
            image_preview = gr.Image(
                label="Image Preview",
                height=400,
                interactive=False
            )
            generate_btn = gr.Button("Generate Captions", variant="primary")
        
        with gr.Column(scale=1):
            output_text = gr.Textbox(
                label="Generated Captions",
                lines=15,
                max_lines=20
            )
    
    # Auto-populate original path when file is uploaded
    image_input.change(
        fn=handle_file_upload,
        inputs=image_input,
        outputs=original_path_input
    )
    
    with gr.Column():
        caption_selector = gr.Radio(
            label="Select Caption to Save (will auto-save on selection)",
            choices=[],
            visible=False
        )
        save_status = gr.Textbox(
            label="Save Status",
            lines=2,
            visible=False
        )
    
    generate_btn.click(
        fn=generate_captions,
        inputs=image_input,
        outputs=[image_preview, output_text, caption_selector, save_status]
    )
    
    # Auto-save when caption is selected
    caption_selector.change(
        fn=save_caption,
        inputs=[caption_selector, original_path_input],
        outputs=save_status
    )
    
    gr.Markdown("---")
    gr.Markdown("Upload an image file, edit the original path if needed, click 'Generate Captions', then select one to auto-save it.")

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)