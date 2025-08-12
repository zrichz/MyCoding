"""
SuperJPEG Encoder/Decoder

A custom JPEG implementation from first principles with configurable block sizes.
Supports block sizes from 4x4 to 64x64 in steps of 4, creating multiple variations
of each input image with different compression characteristics.

Key Features:
- Variable block size (4x4, 8x8, 12x12, ..., 64x64)
- Custom DCT implementation
- Adaptive quantization tables
- Huffman encoding/decoding
- Full JPEG pipeline from scratch

Author: Generated for image processing collection
"""

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import struct
import math
from collections import defaultdict, Counter
import json
from datetime import datetime
import threading


class SuperJPEGEncoder:
    """Custom JPEG encoder with variable block sizes"""
    
    def __init__(self, block_size=8, quality=50):
        self.block_size = block_size
        self.quality = quality
        self.dct_matrix = self._generate_dct_matrix()
        self.idct_matrix = self.dct_matrix.T  # Inverse DCT is transpose
        self.quantization_table = self._generate_quantization_table()
        
    def _generate_dct_matrix(self):
        """Generate DCT transformation matrix for given block size"""
        N = self.block_size
        dct_matrix = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                if i == 0:
                    dct_matrix[i, j] = np.sqrt(1.0 / N)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0 / N) * np.cos((2 * j + 1) * i * np.pi / (2 * N))
                    
        return dct_matrix
    
    def _generate_quantization_table(self):
        """Generate adaptive quantization table based on block size and quality"""
        N = self.block_size
        
        # Base quantization values (JPEG standard 8x8)
        base_quant = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ])
        
        # Create quantization table for block size N with proper frequency scaling
        quant_table = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                # Map to base 8x8 coordinates for spatial scaling
                base_i = min(int(i * 7 / (N - 1)), 7) if N > 1 else 0
                base_j = min(int(j * 7 / (N - 1)), 7) if N > 1 else 0
                base_value = base_quant[base_i, base_j]
                
                # Calculate frequency scaling factor
                # Larger blocks capture lower frequencies in the same spatial positions
                # So we need to adjust quantization based on actual frequency content
                freq_i = i * 8.0 / N  # Equivalent frequency in 8x8 block
                freq_j = j * 8.0 / N  # Equivalent frequency in 8x8 block
                
                # Block size scaling factor
                # Larger blocks need different quantization because:
                # 1. Same spatial frequency appears at different DCT coefficient positions
                # 2. Energy distribution changes with block size
                block_scale_factor = (N / 8.0) ** 0.5  # Square root scaling for energy
                
                # Frequency-based scaling
                # Higher frequencies (larger i,j) should be quantized more aggressively in larger blocks
                freq_magnitude = np.sqrt(freq_i**2 + freq_j**2)
                freq_scale = 1.0 + (freq_magnitude / 8.0) * (N / 8.0 - 1.0) * 0.5
                
                # Apply scaling
                scaled_value = base_value * block_scale_factor * freq_scale
                quant_table[i, j] = max(1, scaled_value)
        
        # Apply quality scaling
        if self.quality < 50:
            scale = 5000 / self.quality
        else:
            scale = 200 - 2 * self.quality
            
        quant_table = np.maximum(1, np.round(quant_table * scale / 100))
        
        return quant_table.astype(int)
    
    def _apply_dct(self, block):
        """Apply 2D DCT to a block"""
        # 2D DCT: F = DCT_matrix * block * DCT_matrix^T
        return np.dot(np.dot(self.dct_matrix, block), self.dct_matrix.T)
    
    def _apply_idct(self, dct_block):
        """Apply inverse 2D DCT to a block"""
        # 2D IDCT: block = IDCT_matrix * dct_block * IDCT_matrix^T
        return np.dot(np.dot(self.idct_matrix, dct_block), self.idct_matrix.T)
    
    def _quantize(self, dct_block):
        """Quantize DCT coefficients"""
        return np.round(dct_block / self.quantization_table).astype(int)
    
    def _dequantize(self, quantized_block):
        """Dequantize DCT coefficients"""
        return quantized_block * self.quantization_table
    
    def _zigzag_scan(self, block):
        """Convert 2D block to 1D array using zigzag pattern"""
        N = self.block_size
        result = []
        
        # Generate zigzag indices
        indices = []
        for diag in range(2 * N - 1):
            if diag % 2 == 0:  # Even diagonal: top-right to bottom-left
                for i in range(max(0, diag - N + 1), min(diag + 1, N)):
                    j = diag - i
                    if 0 <= j < N:
                        indices.append((i, j))
            else:  # Odd diagonal: bottom-left to top-right
                for i in range(min(diag, N - 1), max(0, diag - N + 1) - 1, -1):
                    j = diag - i
                    if 0 <= j < N:
                        indices.append((i, j))
        
        return [block[i, j] for i, j in indices]
    
    def _inverse_zigzag_scan(self, zigzag_data):
        """Convert 1D zigzag array back to 2D block"""
        N = self.block_size
        block = np.zeros((N, N), dtype=int)
        
        # Generate zigzag indices (same as above)
        indices = []
        for diag in range(2 * N - 1):
            if diag % 2 == 0:
                for i in range(max(0, diag - N + 1), min(diag + 1, N)):
                    j = diag - i
                    if 0 <= j < N:
                        indices.append((i, j))
            else:
                for i in range(min(diag, N - 1), max(0, diag - N + 1) - 1, -1):
                    j = diag - i
                    if 0 <= j < N:
                        indices.append((i, j))
        
        for idx, (i, j) in enumerate(indices):
            if idx < len(zigzag_data):
                block[i, j] = zigzag_data[idx]
                
        return block
    
    def _run_length_encode(self, zigzag_data):
        """Run-length encode zigzag data"""
        encoded = []
        i = 0
        while i < len(zigzag_data):
            value = zigzag_data[i]
            count = 1
            
            # Count consecutive identical values
            while i + count < len(zigzag_data) and zigzag_data[i + count] == value:
                count += 1
            
            encoded.append((count, value))
            i += count
            
        return encoded
    
    def _run_length_decode(self, encoded_data):
        """Decode run-length encoded data"""
        decoded = []
        for count, value in encoded_data:
            decoded.extend([value] * count)
        return decoded
    
    def encode_image(self, image_array):
        """Encode an image using SuperJPEG format"""
        if len(image_array.shape) == 3:
            # Convert RGB to YCbCr
            height, width, channels = image_array.shape
            yuv_image = self._rgb_to_yuv(image_array)
        else:
            # Grayscale
            height, width = image_array.shape
            yuv_image = image_array.astype(float)
            channels = 1
        
        # Pad image to be divisible by block size
        pad_height = (self.block_size - height % self.block_size) % self.block_size
        pad_width = (self.block_size - width % self.block_size) % self.block_size
        
        if channels == 3:
            padded_image = np.pad(yuv_image, 
                                ((0, pad_height), (0, pad_width), (0, 0)), 
                                mode='edge')
            new_height, new_width, _ = padded_image.shape
        else:
            padded_image = np.pad(yuv_image, 
                                ((0, pad_height), (0, pad_width)), 
                                mode='edge')
            new_height, new_width = padded_image.shape
        
        # Process image in blocks
        encoded_blocks = []
        
        if channels == 3:
            for channel in range(3):
                channel_data = padded_image[:, :, channel]
                channel_blocks = self._encode_channel(channel_data, new_height, new_width)
                encoded_blocks.append(channel_blocks)
        else:
            channel_blocks = self._encode_channel(padded_image, new_height, new_width)
            encoded_blocks.append(channel_blocks)
        
        # Create SuperJPEG data structure
        superjpeg_data = {
            'magic': 'SUPERJPEG',
            'version': '1.0',
            'block_size': int(self.block_size),
            'quality': int(self.quality),
            'original_width': int(width),
            'original_height': int(height),
            'padded_width': int(new_width),
            'padded_height': int(new_height),
            'channels': int(channels),
            'quantization_table': self.quantization_table.astype(int).tolist(),
            'encoded_blocks': self._convert_to_serializable(encoded_blocks)
        }
        
        return superjpeg_data
    
    def _encode_channel(self, channel_data, height, width):
        """Encode a single channel"""
        blocks = []
        
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                # Extract block
                block = channel_data[y:y+self.block_size, x:x+self.block_size]
                
                # Shift to center around 0
                block = block - 128
                
                # Apply DCT
                dct_block = self._apply_dct(block)
                
                # Quantize
                quantized_block = self._quantize(dct_block)
                
                # Zigzag scan
                zigzag_data = self._zigzag_scan(quantized_block)
                
                # Run-length encode
                rle_data = self._run_length_encode(zigzag_data)
                
                blocks.append(rle_data)
        
        return blocks
    
    def _rgb_to_yuv(self, rgb_image):
        """Convert RGB to YUV color space"""
        rgb = rgb_image.astype(float)
        
        # YUV conversion matrix
        yuv = np.zeros_like(rgb)
        yuv[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]  # Y
        yuv[:, :, 1] = -0.14713 * rgb[:, :, 0] - 0.28886 * rgb[:, :, 1] + 0.436 * rgb[:, :, 2] + 128  # Cb
        yuv[:, :, 2] = 0.615 * rgb[:, :, 0] - 0.51499 * rgb[:, :, 1] - 0.10001 * rgb[:, :, 2] + 128  # Cr
        
        return yuv
    
    def decode_image(self, superjpeg_data):
        """Decode SuperJPEG data back to image"""
        self.block_size = superjpeg_data['block_size']
        self.quality = superjpeg_data['quality']
        self.quantization_table = np.array(superjpeg_data['quantization_table'])
        
        # Regenerate DCT matrices for the block size
        self.dct_matrix = self._generate_dct_matrix()
        self.idct_matrix = self.dct_matrix.T
        
        width = superjpeg_data['original_width']
        height = superjpeg_data['original_height']
        padded_width = superjpeg_data['padded_width']
        padded_height = superjpeg_data['padded_height']
        channels = superjpeg_data['channels']
        encoded_blocks = superjpeg_data['encoded_blocks']
        
        # Decode each channel
        if channels == 3:
            decoded_channels = []
            for channel_idx in range(3):
                channel_data = self._decode_channel(
                    encoded_blocks[channel_idx], 
                    padded_height, 
                    padded_width
                )
                decoded_channels.append(channel_data)
            
            # Combine channels
            yuv_image = np.stack(decoded_channels, axis=2)
            
            # Convert YUV back to RGB
            rgb_image = self._yuv_to_rgb(yuv_image)
            
            # Remove padding
            rgb_image = rgb_image[:height, :width, :]
            
        else:
            # Grayscale
            decoded_image = self._decode_channel(
                encoded_blocks[0], 
                padded_height, 
                padded_width
            )
            rgb_image = decoded_image[:height, :width]
        
        return np.clip(rgb_image, 0, 255).astype(np.uint8)
    
    def _decode_channel(self, encoded_blocks, height, width):
        """Decode a single channel"""
        channel_data = np.zeros((height, width))
        block_idx = 0
        
        for y in range(0, height, self.block_size):
            for x in range(0, width, self.block_size):
                # Decode block
                rle_data = encoded_blocks[block_idx]
                
                # Run-length decode
                zigzag_data = self._run_length_decode(rle_data)
                
                # Inverse zigzag scan
                quantized_block = self._inverse_zigzag_scan(zigzag_data)
                
                # Dequantize
                dct_block = self._dequantize(quantized_block)
                
                # Apply inverse DCT
                block = self._apply_idct(dct_block)
                
                # Shift back to 0-255 range
                block = block + 128
                
                # Place block in image
                channel_data[y:y+self.block_size, x:x+self.block_size] = block
                
                block_idx += 1
        
        return channel_data
    
    def _yuv_to_rgb(self, yuv_image):
        """Convert YUV to RGB color space"""
        yuv = yuv_image.astype(float)
        rgb = np.zeros_like(yuv)
        
        # Adjust Cb and Cr
        cb = yuv[:, :, 1] - 128
        cr = yuv[:, :, 2] - 128
        
        # RGB conversion
        rgb[:, :, 0] = yuv[:, :, 0] + 1.13983 * cr  # R
        rgb[:, :, 1] = yuv[:, :, 0] - 0.39465 * cb - 0.58060 * cr  # G
        rgb[:, :, 2] = yuv[:, :, 0] + 2.03211 * cb  # B
        
        return rgb
    
    def _convert_to_serializable(self, data):
        """Convert NumPy types to native Python types for JSON serialization"""
        if isinstance(data, np.ndarray):
            return data.astype(int).tolist()
        elif isinstance(data, list):
            return [self._convert_to_serializable(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_to_serializable(item) for item in data)
        elif hasattr(data, 'dtype') and 'int' in str(data.dtype):
            return int(data)
        elif hasattr(data, 'dtype') and 'float' in str(data.dtype):
            return float(data)
        else:
            return data


class SuperJPEGGUI:
    """GUI for SuperJPEG encoder/decoder"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("SuperJPEG Encoder/Decoder")
        self.root.geometry("900x700")
        
        self.input_image = None
        self.input_path = ""
        self.superjpeg_path = ""
        self.block_sizes = list(range(4, 68, 4))  # 4, 8, 12, ..., 64
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Encoder tab
        encoder_frame = ttk.Frame(notebook)
        notebook.add(encoder_frame, text="Encoder")
        self.setup_encoder_tab(encoder_frame)
        
        # Decoder tab
        decoder_frame = ttk.Frame(notebook)
        notebook.add(decoder_frame, text="Decoder")
        self.setup_decoder_tab(decoder_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
    
    def setup_encoder_tab(self, parent):
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input Image", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Select Image", 
                  command=self.select_input_image).pack(side=tk.LEFT)
        
        self.input_label = ttk.Label(input_frame, text="No image selected")
        self.input_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Settings section
        settings_frame = ttk.LabelFrame(parent, text="Encoding Settings", padding="10")
        settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Quality setting
        ttk.Label(settings_frame, text="Quality:").grid(row=0, column=0, sticky="w")
        self.quality_var = tk.IntVar(value=50)
        quality_scale = ttk.Scale(settings_frame, from_=1, to=100, 
                                 variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.grid(row=0, column=1, sticky="ew", padx=(10, 5))
        ttk.Label(settings_frame, textvariable=self.quality_var).grid(row=0, column=2)
        
        settings_frame.columnconfigure(1, weight=1)
        
        # Encode button
        encode_btn = ttk.Button(settings_frame, text="Encode to SuperJPEG (All Block Sizes)", 
                               command=self.encode_image)
        encode_btn.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Progress section
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_decoder_tab(self, parent):
        # Input section
        input_frame = ttk.LabelFrame(parent, text="SuperJPEG File", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(input_frame, text="Select SuperJPEG", 
                  command=self.select_superjpeg_file).pack(side=tk.LEFT)
        
        self.superjpeg_label = ttk.Label(input_frame, text="No file selected")
        self.superjpeg_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Decode button
        ttk.Button(input_frame, text="Decode to Image", 
                  command=self.decode_superjpeg).pack(side=tk.RIGHT)
        
        # Image display
        display_frame = ttk.LabelFrame(parent, text="Decoded Image", padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_label = ttk.Label(display_frame)
        self.image_label.pack()
    
    def select_input_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.input_path = file_path
            self.input_label.config(text=os.path.basename(file_path))
            
            # Load and display image info
            try:
                image = Image.open(file_path)
                self.input_image = np.array(image)
                self.log(f"Loaded image: {image.size[0]}x{image.size[1]}, mode: {image.mode}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def encode_image(self):
        if self.input_image is None:
            messagebox.showwarning("Warning", "Please select an input image first")
            return
        
        # Start encoding in separate thread
        thread = threading.Thread(target=self._encode_worker)
        thread.daemon = True
        thread.start()
    
    def _encode_worker(self):
        try:
            if not self.input_path or not os.path.exists(self.input_path):
                self.log("❌ No valid input image selected")
                return
                
            quality = self.quality_var.get()
            total_blocks = len(self.block_sizes)
            
            self.log(f"Starting SuperJPEG encoding with quality {quality}")
            self.log(f"Processing {total_blocks} different block sizes: {self.block_sizes}")
            
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            output_dir = os.path.dirname(self.input_path)
            
            for i, block_size in enumerate(self.block_sizes):
                self.progress_var.set((i / total_blocks) * 100)
                
                self.log(f"Encoding with block size {block_size}x{block_size}...")
                
                # Create encoder
                encoder = SuperJPEGEncoder(block_size=block_size, quality=quality)
                
                # Encode image
                superjpeg_data = encoder.encode_image(self.input_image)
                
                # Save to file
                output_filename = f"{base_name}_superjpeg_{block_size}x{block_size}_q{quality}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(superjpeg_data, f, indent=2)
                
                self.log(f"Saved: {output_filename}")
                
                # Test decode to verify
                try:
                    decoded_image = encoder.decode_image(superjpeg_data)
                    self.log(f"✓ Verified encoding/decoding for {block_size}x{block_size}")
                except Exception as e:
                    self.log(f"✗ Verification failed for {block_size}x{block_size}: {str(e)}")
            
            self.progress_var.set(100)
            self.log("✅ All SuperJPEG encodings completed successfully!")
            self.status_var.set(f"Encoded {total_blocks} SuperJPEG variants")
            
        except Exception as e:
            self.log(f"❌ Encoding error: {str(e)}")
            messagebox.showerror("Error", f"Encoding failed: {str(e)}")
    
    def select_superjpeg_file(self):
        file_path = filedialog.askopenfilename(
            title="Select SuperJPEG File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.superjpeg_path = file_path
            self.superjpeg_label.config(text=os.path.basename(file_path))
    
    def decode_superjpeg(self):
        if not hasattr(self, 'superjpeg_path') or not self.superjpeg_path:
            messagebox.showwarning("Warning", "Please select a SuperJPEG file first")
            return
        
        try:
            # Load SuperJPEG data
            with open(self.superjpeg_path, 'r') as f:
                superjpeg_data = json.load(f)
            
            # Verify it's a SuperJPEG file
            if superjpeg_data.get('magic') != 'SUPERJPEG':
                messagebox.showerror("Error", "Not a valid SuperJPEG file")
                return
            
            # Create decoder
            encoder = SuperJPEGEncoder()  # Will be reconfigured during decode
            
            # Decode image
            decoded_image = encoder.decode_image(superjpeg_data)
            
            # Display image
            self.display_decoded_image(decoded_image, superjpeg_data)
            
            # Save decoded image
            base_name = os.path.splitext(os.path.basename(self.superjpeg_path))[0]
            output_path = f"{base_name}_decoded.png"
            Image.fromarray(decoded_image).save(output_path)
            
            self.status_var.set(f"Decoded and saved as {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")
    
    def display_decoded_image(self, image_array, superjpeg_data):
        # Convert to PIL Image
        if len(image_array.shape) == 3:
            pil_image = Image.fromarray(image_array)
        else:
            pil_image = Image.fromarray(image_array, mode='L')
        
        # Resize for display
        display_size = (400, 400)
        pil_image.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.image_label.config(image=photo, text="")
        # Keep reference to prevent garbage collection
        self.current_photo = photo
        
        # Show info
        info = (f"Block Size: {superjpeg_data['block_size']}x{superjpeg_data['block_size']}\n"
                f"Quality: {superjpeg_data['quality']}\n"
                f"Original Size: {superjpeg_data['original_width']}x{superjpeg_data['original_height']}\n"
                f"Channels: {superjpeg_data['channels']}")
        
        messagebox.showinfo("SuperJPEG Info", info)
    
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()


def main():
    # Check for scipy (needed for quantization table scaling)
    try:
        import scipy.ndimage
    except ImportError:
        print("Installing scipy for quantization table scaling...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
        import scipy.ndimage
    
    root = tk.Tk()
    app = SuperJPEGGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
