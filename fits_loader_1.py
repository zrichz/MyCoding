import tkinter as tk
from tkinter import filedialog
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

# Create a Tkinter root window (it will not be shown)
root = tk.Tk()
root.withdraw()

# Open a file dialog to choose the FITS file
file_path = filedialog.askopenfilename(filetypes=[("FITS files", "*.fits")])
#file_path = "/home/rich/Documents/astro/myDwarf_Fits_Files/DWARF_RAW_M31_EXP_6_GAIN_80_2024-01-06-20-58-33-796/0012.fits"

# Open the selected FITS file
if file_path:
    hdul = fits.open(file_path)
    # Print info of the FITS file
    hdul.info()
    # Access the primary HDU (Header/Data Unit)
    primary_hdu = hdul[0]
    print(primary_hdu.header)  # Print the header of the primary HDU
    data = primary_hdu.data  # Access the data of the primary HDU
    
    print(f"Data shape: {data.shape}")

    # Debayer the data (GRBG). do this by separating the data into the three color channels
    blue = data[1::2, 1::2]
    red = data[0::2, 0::2]
    green = data[1::2, 0::2] + data[0::2, 1::2]

    print(f" blue Data shape after debayer: {blue.shape}")
    print(f"  red Data shape after debayer: {red.shape}")
    print(f"green Data shape after debayer: {green.shape}")

    print("\nnow apply bicubic interpolation to restore each channel to 4K...")
    #the three color channels are now half the size of the original data
    #we need to expand them to the original size of the data, using bicubic interpolation
    blue = resize(blue, (data.shape[0], data.shape[1]), order=3, mode='reflect', anti_aliasing=False)
    red = resize(red, (data.shape[0], data.shape[1]), order=3, mode='reflect', anti_aliasing=False)
    green = resize(green, (data.shape[0], data.shape[1]), order=3, mode='reflect', anti_aliasing=False)
    
    print(f" blue Data shape after expansion: {blue.shape}")
    print(f"  red Data shape after expansion: {red.shape}")
    print(f"green Data shape after expansion: {green.shape}")

    # Print statistics before normalization
    print("\nBefore normalization")

    #show min,max and mean values for each channel, to 2 decimal places
    print(f"  red channel min: {np.min(red):.2f}    max: {np.max(red):.2f}    mean: {np.mean(red):.2f}")
    print(f"green channel min: {np.min(green):.2f}    max: {np.max(green):.2f}    mean: {np.mean(green):.2f}")
    print(f" blue channel min: {np.min(blue):.2f}    max: {np.max(blue):.2f}    mean: {np.mean(blue):.2f}")
    
    #use log transformation to enhance the image
    blue = np.log(blue + 1)
    red = np.log(red + 1)
    green = np.log(green + 1)

    # Normalize the color channels to the range [0, 255]
    blue = (blue - np.min(blue)) / (np.max(blue) - np.min(blue)) * 255
    red = (red - np.min(red)) / (np.max(red) - np.min(red)) * 255
    green = (green - np.min(green)) / (np.max(green) - np.min(green)) * 255

    print("\nAfter normalization")
    print(f" blue channel min: {np.min(blue)}")
    print(f" blue channel max: {np.max(blue)}")
    print(f"  red channel min: {np.min(red)}")
    print(f"  red channel max: {np.max(red)}")
    print(f"green channel min: {np.min(green)}")
    print(f"green channel max: {np.max(green)}")
    print(f" blue channel mean: {np.mean(blue)}")
    print(f"  red channel mean: {np.mean(red)}")
    print(f"green channel mean: {np.mean(green)}")

    
    # Create a color image with three channels
    color = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    color[:, :, 0] = red
    color[:, :, 1] = green
    color[:, :, 2] = blue

    print(f"\ncolor Data shape after debayer: {color.shape}")

    # Display the debayered color image
    plt.imshow(color)
    plt.title('Debayered Color Image')
    plt.show()
    
    # Close the FITS file
    hdul.close()
else:
    print("No file selected")