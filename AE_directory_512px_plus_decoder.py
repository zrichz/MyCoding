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
pix_per_side = 64 # image size


# Define the transformation
transform = transforms.Compose([
  transforms.Resize((pix_per_side, pix_per_side)),
  transforms.Grayscale(),  # Convert to grayscale
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale image
])

# Load the images from a directory
dataset = ImageFolder('C:/MyPythonCoding/MyDeepLearningCoding/images_512x512', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

print('dataloader created')

# create a class for the model
def createTheMNISTAE():

  class aenet(nn.Module):
    def __init__(self):
      super().__init__()

      self.input = nn.Linear(pix_per_side*pix_per_side,600)

      self.enc = nn.Linear(600,16)

      self.lat = nn.Linear(16,600)

      self.dec = nn.Linear(600,pix_per_side*pix_per_side)

    # forward pass
    def forward(self,x):
      x = F.relu( self.input(x) )
      x = F.relu( self.enc(x) )
      x = F.relu( self.lat(x) )
      y = torch.sigmoid( self.dec(x) )
      return y

  # create the model instance
  net = aenet()

  lossfun = nn.MSELoss()
  optimizer = torch.optim.Adam(net.parameters(),lr=.001)

  return net,lossfun,optimizer

def function2trainTheModel():
  numepochs = 250

  net,lossfun,optimizer = createTheMNISTAE()       # create a new model
  
  losses = []     # initialize losses
  for epochi in range(numepochs):
    if epochi == 0: print('training has started')
    for images, _ in dataloader:
      images = images.view(images.size(0), -1)  # Flatten the images
      yHat = net(images)              # forward pass
      loss = lossfun(yHat,images)     # loss        

      optimizer.zero_grad()           # backprop
      loss.backward()
      optimizer.step()

      # losses in this epoch
      losses.append( loss.item() )
    print (epochi,' of ',numepochs,'loss: ',loss.item())

  return losses,net

# TRAIN model
#############
losses,net = function2trainTheModel()
print(f'Final loss: {losses[-1]:.4f}')

# visualize the losses
plt.plot(losses,'.-')
plt.xlabel('Epochs')
plt.ylabel('Model loss')
plt.show()

# create a decoder model, that just decodes the latent representation
def fnDECODER():
  class mydecoder(nn.Module):
    def __init__(self):
      super().__init__()

      self.lat = nn.Linear(16,600)          # latent layer

      self.dec = nn.Linear(600,pix_per_side*pix_per_side)  # decoder layer

    # forward pass
    def forward(self,x):
      x = F.relu( self.lat(x) )
      y = torch.sigmoid( self.dec(x) )
      return y

  # create the model instance
  decnet = mydecoder()

  # copy the weights from the trained 'net' model to 'decnet'...
  # for latent layer...
  decnet.lat.weight.data = net.lat.weight.data
  decnet.lat.bias.data = net.lat.bias.data
  # and decoder layer...
  decnet.dec.weight.data = net.dec.weight.data
  decnet.dec.bias.data = net.dec.bias.data

  return decnet

decnet = fnDECODER()  # call the function and create the model


remade1 = decnet(torch.rand(16))  # pass random noise to the decoder
remade2 = decnet(torch.rand(16))
remade3 = decnet(torch.rand(16))
remade4 = decnet(torch.rand(16))
remade5 = decnet(torch.rand(16))
remade6 = decnet(torch.rand(16))

# show images
fig, axs = plt.subplots(1, 6, figsize=(12, 4)) # 1 row, 6 columns
axs[0].imshow(remade1.view(pix_per_side, pix_per_side).detach())
axs[1].imshow(remade2.view(pix_per_side, pix_per_side).detach())
axs[2].imshow(remade3.view(pix_per_side, pix_per_side).detach())
axs[3].imshow(remade4.view(pix_per_side, pix_per_side).detach())
axs[4].imshow(remade5.view(pix_per_side, pix_per_side).detach())
axs[5].imshow(remade6.view(pix_per_side, pix_per_side).detach())

plt.show()

print('checkpoint 4')