# vanilla autoencoder, ,all images in directory. adapted from Mike X Cohen code with copilot

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path
#-----------------------------------------------------------------------------------------

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image"""
    image = Image.open(image_path)
    image_resized = image.resize((64, 64))
    image_grey = image_resized.convert('L')
    image_flat = np.array(image_grey).flatten()
    return image_flat / np.max(image_flat)

def load_image_directory(directory_path):
    """Load all images from directory"""
    # Supported image formats
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    # List to store processed images
    processed_images = []
    
    # Iterate through directory
    for file in Path(directory_path).glob('*'):
        if file.suffix.lower() in valid_extensions:
            try:
                processed = load_and_preprocess_image(file)
                processed_images.append(processed)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Convert to tensor
    if processed_images:
        return torch.tensor(np.stack(processed_images)).float()
    else:
        raise ValueError("No valid images found in directory")

# Replace single image loading with directory loading
directory_path = 'C:\\MyPythonCoding\\MyDeepLearningCoding\\images'

dataT = load_image_directory(directory_path)
print(f"Loaded {len(dataT)} images, tensor shape: {dataT.shape}")

# Add these debug prints after loading images
print(f"Individual image shape before flatten: (64, 64)")
print(f"Individual image shape after flatten: (4096,)")
print(f"Batch shape: {dataT.shape}")

print('checkpoint 1')

print(dataT.shape)
print('checkpoint 2')

# create a class for the model
def createTheMNISTAE():

    class aenet(nn.Module):
        def __init__(self):
            super().__init__()

            self.input = nn.Linear(4096, 128)

            self.enc = nn.Linear(128, 32)

            self.lat = nn.Linear(32, 128)

            self.dec = nn.Linear(128, 4096)

        # forward pass
        def forward(self, x):
            # Handle both single images and batches
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = F.relu(self.input(x))
            x = F.relu(self.enc(x))
            x = F.relu(self.lat(x))
            return torch.sigmoid(self.dec(x))

    # create the model instance
    net = aenet()

    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=.0001)

    return net, lossfun, optimizer

# Modify training loop to handle batches
def function2trainTheModel():
    numepochs = 1500
    net, lossfun, optimizer = createTheMNISTAE()
    losses = []
    
    for epochi in range(numepochs):
        total_loss = 0
        
        # Process all images
        yHat = net(dataT)
        loss = lossfun(yHat, dataT)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epochi % 15 == 0:
            print(f'{epochi} of {numepochs+1} : {loss.item()}')
    
    return losses, net

# TRAIN model
#############
losses,net = function2trainTheModel()
print(f'Final loss: {losses[-1]:.4f}')

# visualize the losses
plt.plot(losses,'.-')
plt.xlabel('Epochs')
plt.ylabel('Model loss')
#add a title
plt.title('Model training loss')
plt.show()

# create a true copy of the data, so we have the same shape and type, then replace data with just random noise
X = copy.deepcopy( dataT )
X = 1*torch.randn(X.shape) # random noise

# run (noise) through the (TRAINED) model
###########################################    
deOccluded = net(X)

def plot_denoised(deOccluded, num_images=8):
    # Create figure with single column of denoised images
    fig, axs = plt.subplots(num_images, 1, figsize=(4, 4*num_images))
    
    # Handle case where only one image is shown
    if num_images == 1:
        axs = [axs]
    
    # Plot denoised images
    for i in range(min(num_images, len(deOccluded))):
        denoised = deOccluded[i].view(64, 64).detach()
        axs[i].imshow(denoised, cmap='gray')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    plt.suptitle('Denoised Images')
    plt.tight_layout()
    plt.show()

# Replace existing visualization code with:
plot_denoised(deOccluded)

print('checkpoint 3')
