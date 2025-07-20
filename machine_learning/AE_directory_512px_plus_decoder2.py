# vanilla autoencoder, single image dataset(!) adapted from Mike X Cohen code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# global variables
pix_per_side = 64 # image size (increase for more detail)
latent_dim = 32   # number of latent dimensions (increase for more capacity)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:',device)

# Define the transformation (remove grayscale, use color)
transform = transforms.Compose([
  transforms.Resize((pix_per_side, pix_per_side)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize RGB image
])

# Load the images from a directory
dataset = ImageFolder(r'D:\2024_My_AI_art\512x512_5step_FigureStudy', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
print('dataloader created')

# create a class for the model (use convolutional layers)
def createTheConvAE():
    class ConvAE(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1),  # (B,32,32,32)
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), # (B,64,16,16)
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1), # (B,128,8,8)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(128 * (pix_per_side // 8) * (pix_per_side // 8), latent_dim)
            )
            # Decoder
            self.decoder_fc = nn.Linear(latent_dim, 128 * (pix_per_side // 8) * (pix_per_side // 8))
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (128, pix_per_side // 8, pix_per_side // 8)),
                nn.ConvTranspose2d(128, 64, 4, 2, 1), # (B,64,16,16)
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (B,32,32,32)
                nn.ReLU(),
                nn.ConvTranspose2d(32, 3, 4, 2, 1),   # (B,3,64,64)
                nn.Sigmoid()
            )

        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder_fc(z)
            x_hat = self.decoder(x_hat)
            return x_hat

    net = ConvAE().to(device)
    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    return net, lossfun, optimizer

def function2trainTheModel():
    numepochs = 50  # increase epochs for better results
    net, lossfun, optimizer = createTheConvAE()
    losses = []
    for epochi in range(numepochs):
        net.train()
        for images, _ in dataloader:
            images = images.to(device)
            yHat = net(images)
            loss = lossfun(yHat, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epochi} of {numepochs}, loss: {loss.item():.3f}')
        if (epochi + 1) % 10 == 0:
            avg_loss = np.mean(losses[-len(dataloader)*10:])
            print(f'Average loss for epochs {epochi-9} to {epochi}: {avg_loss:.3f}\n')
    return losses, net

# TRAIN model
#############
if __name__ == "__main__":
    losses, net = function2trainTheModel()
    print(f'Final loss: {losses[-1]:.4f}')

    # visualize the losses
    plt.plot(losses, '.-', label='Loss')
    # calculate and plot the EMA of losses
    ema_losses = []
    alpha = 0.1  # smoothing factor
    ema = losses[0]
    for loss in losses:
        ema = alpha * loss + (1 - alpha) * ema
        ema_losses.append(ema)
    plt.plot(ema_losses, 'r-', label='EMA of Losses')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Model loss')
    plt.show()

    # Visualize reconstructions of real images
    net.eval()
    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)
    with torch.no_grad():
        reconstructed = net(real_images)
    n_show = min(6, real_images.size(0))
    fig, axs = plt.subplots(2, n_show, figsize=(2*n_show, 4))
    for i in range(n_show):
        axs[0, i].imshow((real_images[i].permute(1,2,0).cpu().numpy() * 0.5) + 0.5)
        axs[0, i].set_title("Original")
        axs[0, i].axis('off')
        axs[1, i].imshow((reconstructed[i].permute(1,2,0).cpu().numpy() * 0.5) + 0.5)
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()

    # Visualize reconstructions from random noise
    net.eval()
    n_random = 5
    # Generate random latent vectors
    random_latents = torch.randn(n_random, latent_dim, device=device)
    # Pass through decoder only
    with torch.no_grad():
        random_decoded = net.decoder(net.decoder_fc(random_latents))
    fig, axs = plt.subplots(1, n_random, figsize=(2*n_random, 2))
    for i in range(n_random):
        axs[i].imshow((random_decoded[i].permute(1,2,0).cpu().numpy() * 0.5) + 0.5)
        axs[i].set_title("Random Recon")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()