from PIL import Image

# Open the image file
img = Image.open(r'C:\MyPythonCoding\MyCoding\images_general\symbols.png') # r is for raw string

#C:\MyPythonCoding\MyCoding\images_general\symbols.png

# Get the size of the image
width, height = img.size

# Define the number of splits
print("The image is", width, "pixels wide and", height, "pixels high.")
splits = input("Enter the number of splits (e.g., '4,3' for 4 horizontal and 3 vertical): ")
h_splits, v_splits = map(int, splits.split(','))

# Calculate the size of each piece
h_size = width // h_splits
v_size = height // v_splits

# Loop through the image
for i in range(h_splits):
    for j in range(v_splits):
        # Define the box to crop
        box = (i*h_size, j*v_size, (i+1)*h_size, (j+1)*v_size)
        # Crop the image
        cropped_img = img.crop(box)
        # Save the cropped image
        cropped_img.save(f'image_processors/crops/cropped_{i}_{j}.png')