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
    
    print(f"original raw data shape: {data.shape}")

    ''' Debayer the data (GBRG). do this by separating the data into the three color channels
            |-0-1---2-3---4-5---6-7--
            |_column____________________
    row 0   | G B   G B   G B   G B
        1   | R G   R G   R G   R G
            |________________________
        2   | G B   G B   G B   G B
        3   | R G   R G   R G   R G
            |________________________
    '''
# note GBRG is apparently the bayer pattern for the Dwarf2, and is also what is in the FITS header

    red = data[1::2, 0::2]     # Red: start at row 1, col 0
    green = (data[0::2, 0::2] + data[1::2, 1::2]) / 2  # Green: positions (0,0) and (1,1) (average the two)
    blue = data[0::2, 1::2]    # Blue: start at row 0, col 1
    
    assert blue.shape == red.shape == green.shape, "The shapes of the extracted color channels do not match!"
    print(f"Extraction of RBG channels complete. All channels have the same shape: {blue.shape}")

    print("\napply interpolation to restore each channel to 4K...")
    # the three color channels are now half the size of the original data
    # we can expand them to the original size of the data, using bicubic interpolation
    # a note on "order" parameter: 
    # order=0: Nearest-neighbor
    # order=1: Bi-linear : Uses weighted avg of 4 nearest pixels
    # order=2: Bi-quadratic : Uses weighted avg of 9 nearest pixels
    # orders=3,4,5: Use weighted average of 16,25,36 nearest pixels respectively
    # mode='reflect' : Reflects the image at the boundaries
    
    blue = resize(blue, (data.shape[0], data.shape[1]), order=1, mode='reflect', anti_aliasing=False)
    red = resize(red, (data.shape[0], data.shape[1]), order=1, mode='reflect', anti_aliasing=False)
    green = resize(green, (data.shape[0], data.shape[1]), order=1, mode='reflect', anti_aliasing=False)
    
    assert blue.shape == red.shape == green.shape, "The shapes of the color channels do not match after expansion!"
    print(f"All channels have the same shape after expansion: {blue.shape}")

    # Print statistics before normalization
    print("\nBefore normalization")

    #show min,max and mean values for each channel, to n decimal places
    print(f"  red channel min: {np.min(red):.3f}    max: {np.max(red):.3f}    mean: {np.mean(red):.3f}")
    print(f"green channel min: {np.min(green):.3f}    max: {np.max(green):.3f}    mean: {np.mean(green):.3f}")
    print(f" blue channel min: {np.min(blue):.3f}    max: {np.max(blue):.3f}    mean: {np.mean(blue):.3f}")

    #use log transformation to enhance the image
    blue = np.log(blue + 1)
    red = np.log(red + 1)
    green = np.log(green + 1)
    
    # normalize to 0-255 range
    red = (red - np.min(red)) / (np.max(red) - np.min(red)) * 255
    green = (green - np.min(green)) / (np.max(green) - np.min(green)) * 255
    blue = (blue - np.min(blue)) / (np.max(blue) - np.min(blue)) * 255

    # Verify results
    print("\nstats after log transform and subsequent normalization")
    print(f"  RED min: {np.min(red):.3f}    max: {np.max(red):.3f}    mean: {np.mean(red):.3f}")
    print(f"GREEN channel min: {np.min(green):.3f}    max: {np.max(green):.3f}    mean: {np.mean(green):.3f}")
    print(f" BLUE channel min: {np.min(blue):.3f}    max: {np.max(blue):.3f}    mean: {np.mean(blue):.3f}")

    # apply a curve adjustment to the green channel
    # this is a simple gamma correction
    gamma = 1.5
    green = ((green/255) ** gamma) * 255 # apply gamma correction. CARE: need to normalize to 0-1 range first, then multiply by 255
    print(f"GREEN channel following gamma correction: min: {np.min(green):.3f}    max: {np.max(green):.3f}    mean: {np.mean(green):.3f}")
        
    # Create a color 4K image with three channels
    color = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8) # Initialize with zeros
    color[:, :, 0] = red
    color[:, :, 1] = green
    color[:, :, 2] = blue

    print(f"\ncolor Data shape after debayer: {color.shape}")

    # # Display the debayered color image
    # plt.imshow(color)
    # #plt.axis('off')  # Turn off axis
    # plt.title('Debayered Color Image')
    # plt.show()


    #now convert the RGB image to grayscale, using the formula: Y = 0.2126 R + 0.7152 G + 0.0722 B
    #note that the above formula is for the sRGB color space, which is what we are using here
    #we will use the same formula to convert the color image to grayscale
    #first, normalize the color channels to 0-1 range
    red = red / 255
    green = green / 255
    blue = blue / 255

    #apply the formula to convert to grayscale
    gray = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    gray = gray * 255 #convert back to 0-255 range

    # Display the grayscale image
    #set vmin and vmax to ensure that the image is displayed correctly
    vmin = np.min(gray)
    vmax = np.max(gray)/5
    
    plt.imshow(gray, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title('Grayscale Image')
    
    plt.show() # Display the grayscale image    





    # Close the FITS file
    hdul.close()
else:
    print("No file selected")