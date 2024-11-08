'''display and crop images from a specified directory. 

uses `PIL` (Python Imaging Library) and `matplotlib` to handle images and plotting.

defines a function `onclick(event)` that handles mouse clicks.
This is used with `matplotlib`'s event handling system - When a mouse click occurs, the function records xy coords of the click.
Co-ords are stored in a global list `coords`. Once 2 clicks have been recorded, the function closes the plot.

The main part of the script is the `display_images(directory)` function.
This takes a directory path as an argument and processes all `.jpg` images in that directory.
It opens and displays each image. The user can then click on the image to select a rectangle to crop.

The function then connects the `onclick` function to the mouse click event using `fig.canvas.mpl_connect('button_press_event', onclick)`.
 - every time a mouse click occurs in the plot, the `onclick` function will be called.

After displaying the plot, the function checks if two coordinates have been selected. 
If they have, it crops the image to the rectangle defined by the two coordinates using `image.crop()`.
The cropped image is then saved in a subdirectory named `cropped` within the original directory.

The coords are then reset to an empty list, ready for the next image.
This process repeats for all `.jpg` images in the specified directory.'''

import os
from PIL import Image
import matplotlib.pyplot as plt

# Function to handle mouse click events
def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    global coords
    coords.append((ix, iy))
    if len(coords) == 2:
        plt.close()


def display_images(directory):
    global coords
    coords = []
    cropped_dir = f'{directory}/cropped'
    os.makedirs(cropped_dir, exist_ok=True)
        
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith('.jpg'):
                # Open the image
                image = Image.open(entry.path)
                    
                # Create a figure and subplot
                fig = plt.figure()
                ax = fig.add_subplot(111)
                    
                ax.imshow(image)
                
                # Connect the mouse click event to the onclick function
                cid = fig.canvas.mpl_connect('button_press_event', onclick)
                
                plt.show()
                
                # If two coordinates are selected, crop the image and save it
                if len(coords) == 2:
                    cropped_image = image.crop((coords[0][0], coords[0][1], coords[1][0], coords[1][1]))
                    cropped_image.save(f'{cropped_dir}/{entry.name}')
                    
                    # Reset the coordinates
                    coords = []

# Example usage
#display_images('C:/Users/richm/Pictures')
                    
def main():
    directory = input("Enter the directory path (defualts to C:/Users/richm/Pictures): ")
    if not directory:
        directory = 'C:/Users/richm/Pictures'
    display_images(directory)

if __name__ == '__main__':
    main()