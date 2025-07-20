"""
Slitscanner - Video to Image Processing Application

This application creates artistic images by extracting vertical pixel columns
from video frames and combining them horizontally into a single image.
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os


class SlitscannerApp:
    def __init__(self):
        # Configure CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Slitscanner - Video to Image Converter")
        self.root.geometry("900x700")
        
        # Variables
        self.video_path = None
        self.output_image = None
        self.processing = False
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main title
        title_label = ctk.CTkLabel(
            self.root, 
            text="🎬 Slitscanner Video Processor", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = ctk.CTkFrame(self.root)
        file_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(file_frame, text="Video File:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        select_frame = ctk.CTkFrame(file_frame)
        select_frame.pack(pady=5, padx=20, fill="x")
        
        self.file_label = ctk.CTkLabel(select_frame, text="No file selected", wraplength=400)
        self.file_label.pack(side="left", padx=10, pady=10)
        
        self.select_button = ctk.CTkButton(
            select_frame, 
            text="Select Video", 
            command=self.select_video_file,
            width=120
        )
        self.select_button.pack(side="right", padx=10, pady=10)
        
        # Processing options frame
        options_frame = ctk.CTkFrame(self.root)
        options_frame.pack(pady=10, padx=20, fill="x")
        
        ctk.CTkLabel(options_frame, text="Processing Options:", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
        
        # Frame sampling option
        sampling_frame = ctk.CTkFrame(options_frame)
        sampling_frame.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkLabel(sampling_frame, text="Frame Sampling:").pack(side="left", padx=10, pady=10)
        
        self.sampling_var = ctk.StringVar(value="every_frame")
        
        sampling_options = [
            ("Every Frame", "every_frame"),
            ("Every 2 Frames", "every_2"),
            ("Every N Frames", "every_n")
        ]
        
        for text, value in sampling_options:
            radio = ctk.CTkRadioButton(
                sampling_frame, 
                text=text, 
                variable=self.sampling_var, 
                value=value,
                command=self.on_sampling_change
            )
            radio.pack(side="left", padx=10, pady=10)
        
        # N frames input (initially hidden)
        self.n_frame = ctk.CTkFrame(options_frame)
        
        ctk.CTkLabel(self.n_frame, text="Every N frames (3-1000):").pack(side="left", padx=10, pady=10)
        self.n_entry = ctk.CTkEntry(self.n_frame, width=100, placeholder_text="Enter N (3-1000)")
        self.n_entry.pack(side="left", padx=10, pady=10)
        self.n_entry.insert(0, "3")
        
        # Add validation feedback label
        self.n_validation_label = ctk.CTkLabel(self.n_frame, text="", text_color="orange")
        self.n_validation_label.pack(side="left", padx=10, pady=10)
        
        # Bind validation to entry changes
        self.n_entry.bind("<KeyRelease>", self.validate_n_input)
        
        # Process button
        self.process_button = ctk.CTkButton(
            self.root,
            text="🎯 Process Video",
            command=self.start_processing,
            height=40,
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.process_button.pack(pady=20)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.root, width=400)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(self.root, text="Ready to process video")
        self.progress_label.pack(pady=5)
        
        # Image display frame
        self.image_frame = ctk.CTkScrollableFrame(self.root, height=200)
        self.image_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="Processed image will appear here")
        self.image_label.pack(pady=20)
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.root,
            text="💾 Save Image",
            command=self.save_image,
            state="disabled"
        )
        self.save_button.pack(pady=10)
        
    def on_sampling_change(self):
        """Show/hide N frames input based on selection"""
        if self.sampling_var.get() == "every_n":
            self.n_frame.pack(pady=5, padx=20, fill="x")
            self.validate_n_input()  # Validate current value when shown
        else:
            self.n_frame.pack_forget()
    
    def validate_n_input(self, event=None):
        """Validate the N frames input and provide feedback"""
        try:
            value = int(self.n_entry.get())
            if 3 <= value <= 1000:
                self.n_validation_label.configure(text="✓ Valid", text_color="green")
                return True
            else:
                self.n_validation_label.configure(text="Must be 3-1000", text_color="orange")
                return False
        except ValueError:
            if self.n_entry.get().strip() == "":
                self.n_validation_label.configure(text="Required", text_color="orange")
            else:
                self.n_validation_label.configure(text="Invalid number", text_color="red")
            return False
    
    def select_video_file(self):
        """Open file dialog to select video file"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )
        
        if filename:
            self.video_path = filename
            self.file_label.configure(text=os.path.basename(filename))
            
    def get_frame_step(self):
        """Get the frame step based on user selection"""
        sampling = self.sampling_var.get()
        
        if sampling == "every_frame":
            return 1
        elif sampling == "every_2":
            return 2
        elif sampling == "every_n":
            try:
                value = int(self.n_entry.get())
                # Enforce the 3-1000 range
                if 3 <= value <= 1000:
                    return value
                else:
                    # Show error and return default
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Invalid Input", 
                        f"N frames value must be between 3 and 1000. Using default value of 3."
                    ))
                    return 3
            except ValueError:
                # Show error and return default
                self.root.after(0, lambda: messagebox.showwarning(
                    "Invalid Input", 
                    f"Please enter a valid number between 3 and 1000. Using default value of 3."
                ))
                return 3
        return 1
    
    def start_processing(self):
        """Start video processing in a separate thread"""
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first!")
            return
            
        if self.processing:
            return
        
        # Validate N frames input if "every_n" is selected
        if self.sampling_var.get() == "every_n":
            if not self.validate_n_input():
                messagebox.showerror("Invalid Input", 
                                   "Please enter a valid number between 3 and 1000 for frame sampling.")
                return
            
        # Start processing in background thread
        self.processing = True
        self.process_button.configure(state="disabled", text="Processing...")
        
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()
    
    def process_video(self):
        """Process the video and create slitscanned image"""
        try:
            # Open video file
            cap = cv2.VideoCapture(self.video_path)
            
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate center column
            center_x = frame_width // 2
            
            # Get frame step
            frame_step = self.get_frame_step()
            
            # Calculate how many frames we'll actually process
            frames_to_process = total_frames // frame_step
            max_width = 3000
            
            if frames_to_process > max_width:
                frames_to_process = max_width
                
            self.root.after(0, lambda: self.progress_label.configure(
                text=f"Processing {frames_to_process} frames..."
            ))
            
            # Create output image array
            slitscanned_image = np.zeros((frame_height, frames_to_process, 3), dtype=np.uint8)
            
            frame_count = 0
            processed_count = 0
            
            while cap.isOpened() and processed_count < frames_to_process:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Check if we should process this frame
                if frame_count % frame_step == 0:
                    # Extract center column
                    center_column = frame[:, center_x, :]
                    
                    # Add to slitscanned image
                    slitscanned_image[:, processed_count, :] = center_column
                    
                    processed_count += 1
                    
                    # Update progress
                    progress = processed_count / frames_to_process
                    self.root.after(0, lambda p=progress: self.progress_bar.set(p))
                    self.root.after(0, lambda c=processed_count, t=frames_to_process: 
                                  self.progress_label.configure(
                                      text=f"Processed {c}/{t} frames ({round(c/t*100)}%)"
                                  ))
                
                frame_count += 1
            
            cap.release()
            
            # Convert BGR to RGB
            slitscanned_image = cv2.cvtColor(slitscanned_image, cv2.COLOR_BGR2RGB)
            
            # Store the result
            self.output_image = Image.fromarray(slitscanned_image)
            
            # Display the image
            self.root.after(0, self.display_image)
            
            self.root.after(0, lambda: self.progress_label.configure(
                text=f"✅ Complete! Created {processed_count}x{frame_height} pixel image"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.progress_label.configure(text="❌ Processing failed"))
        
        finally:
            self.processing = False
            self.root.after(0, lambda: self.process_button.configure(state="normal", text="🎯 Process Video"))
    
    def display_image(self):
        """Display the processed image in the GUI"""
        if self.output_image:
            # Calculate display size (scale down if too large)
            display_width = min(800, self.output_image.width)
            display_height = int(self.output_image.height * (display_width / self.output_image.width))
            
            # Resize for display
            display_image = self.output_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to CTkImage for better compatibility
            ctk_image = ctk.CTkImage(light_image=display_image, dark_image=display_image, 
                                   size=(display_width, display_height))
            
            # Update label
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image  # Keep a reference
            
            # Enable save button
            self.save_button.configure(state="normal")
    
    def save_image(self):
        """Save the processed image"""
        if not self.output_image:
            messagebox.showerror("Error", "No image to save!")
            return
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.asksaveasfilename(
            title="Save Slitscanned Image",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filename:
            try:
                self.output_image.save(filename)
                messagebox.showinfo("Success", f"Image saved as {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = SlitscannerApp()
    app.run()
