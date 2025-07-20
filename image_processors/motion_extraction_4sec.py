import subprocess
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the root Tkinter window
Tk().withdraw()

# Step 1: File selection dialog for input file
# Allows the user to select an input video file using a GUI dialog
input_file = askopenfilename(title="Select Input Video File", filetypes=[("Video Files", "*.mp4")])
if not input_file:
    print("No file selected. Exiting.")
    exit(1)

# Generate temporary and output file paths
# temp_file: Resized version of the input file
# output_file: Final montage output file
temp_file = "temp_resized.mp4"
output_file = os.path.splitext(input_file)[0] + "_motion.mp4"

# Step 2: Reduce input resolution to a maximum of 480x360
# Ensures the resolution is reduced while maintaining the aspect ratio
resize_command = [
    "ffmpeg",
    "-i", input_file,
    "-loglevel", "error",  # Suppress unnecessary output
    "-vf", "scale=480:-2",  # Scale width to 480 and adjust height to maintain aspect ratio
    "-c:v", "libx264", # Use H.264 codec for video
    "-c:a", "aac", # Use AAC codec for audio
    "-y", temp_file # Output to a temporary file
]

# Run the resize command
print("Reducing resolution to 480x360...")
try:
    subprocess.run(resize_command, check=True)
    print(f"Resolution reduced. Temporary file saved to {temp_file}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while resizing video: {e}")
    exit(1)

# Step 3: Prompt user for time shift in seconds
# The user specifies how much to shift the video in seconds
try:
    time_shift = float(input("Enter the time shift in seconds (e.g., 4.0): "))
    if time_shift <= 0:
        raise ValueError("Time shift must be a positive number.")
except ValueError as e:
    print(f"Invalid input for time shift: {e}")
    exit(1)

# Step 4: Get the duration of the input file
# Uses ffprobe to retrieve the total duration of the input video
try:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    input_duration = float(result.stdout.strip())
    if input_duration <= time_shift:
        raise ValueError("Time shift must be less than the duration of the input video.")
except (subprocess.CalledProcessError, ValueError) as e:
    print(f"Error occurred while retrieving input file duration: {e}")
    exit(1)

# Calculate the duration of the output montage
# The output duration is the input duration minus the time shift
output_duration = input_duration - time_shift

# Step 5: Create the alpha-blended overlay file
# This step creates a video with an inverted and alpha-blended overlay
alpha_blended_temp_file = "temp_alpha_blended.mp4"
alpha_blend_command = [
    "ffmpeg",
    "-i", temp_file,  # Use the resized temp file
    "-loglevel", "error",  # Suppress unnecessary output
    "-filter_complex",
    f"[0:v]split=2[orig][inv]; "
    f"[inv]lutrgb=r=negval:g=negval:b=negval,eq=contrast=1.0:brightness=0.0:saturation=1.0,format=yuva444p, "
    f"colorchannelmixer=aa=0.5,setpts=PTS-{time_shift}/TB[inv_shifted]; "
    f"[orig][inv_shifted]overlay=format=auto:shortest=1[blended]",
    "-map", "[blended]",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-shortest",
    "-y", alpha_blended_temp_file
]

# Run the alpha blending command
try:
    subprocess.run(alpha_blend_command, check=True)
    print(f"Alpha-blended video created. Temporary file saved to {alpha_blended_temp_file}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while creating alpha-blended video: {e}")
    exit(1)

# Step 6: Enhance the contrast of the alpha-blended video
# Applies a contrast enhancement to the alpha-blended video
contrast_enhanced_alpha_temp_file = "temp_alpha_blended_contrast.mp4"  # Temporary file for contrast enhancement
contrast_enhance_command = [
    "ffmpeg",
    "-i", alpha_blended_temp_file,
    "-loglevel", "error",  # Suppress unnecessary output
    "-vf", "eq=contrast=3.0",  # Apply contrast enhancement
    "-c:v", "libx264",
    "-c:a", "aac",
    "-y", contrast_enhanced_alpha_temp_file
]

# Run the contrast enhancement command
try:
    subprocess.run(contrast_enhance_command, check=True)
    print(f"Contrast enhanced for alpha-blended video. Temporary file created: {contrast_enhanced_alpha_temp_file}")
    # Replace the original alpha-blended file with the contrast-enhanced version
    os.replace(contrast_enhanced_alpha_temp_file, alpha_blended_temp_file)
    print(f"Original alpha-blended file replaced with contrast-enhanced version: {alpha_blended_temp_file}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while enhancing contrast for alpha-blended video: {e}")
    exit(1)

# Step 7: Create the montage
# Combines the original resized video (LHS) and the contrast-enhanced alpha-blended video (RHS)
ffmpeg_command = [
    "ffmpeg",
    "-i", temp_file,  # Original temp file
    "-i", alpha_blended_temp_file,  # Alpha-blended temp file (now contrast-enhanced)
    "-loglevel", "error",  # Suppress unnecessary output
    "-filter_complex",
    f"[0:v][1:v]hstack=inputs=2[stacked]; "
    f"[stacked]trim=duration={output_duration},setpts=PTS-STARTPTS[v]",
    "-map", "[v]",
    "-map", "0:a?",
    "-c:v", "libx264",
    "-c:a", "aac",
    "-shortest",
    "-y", output_file
]

# Run the montage command
try:
    subprocess.run(ffmpeg_command, check=True)
    print(f"Video montage created. Output saved to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while creating video montage: {e}")
    exit(1)