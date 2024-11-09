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
pix_per_side = 32 # image size
latent_dim = 2   # latent dimension
numepochs = 501   # number of epochs

#--------------------------------------------------------------------------------
# Calculate mean and standard deviation of images in a directory
def calculate_mean_std(directory):
  transform = transforms.Compose([
    transforms.Resize((pix_per_side, pix_per_side)),
    transforms.ToTensor()
  ])
  dataset = ImageFolder(directory, transform=transform)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
  
  mean = torch.zeros(3)
  std = torch.zeros(3)
  num_samples = 0
  
  for image, _ in dataloader:
    mean += torch.mean(image, dim=(2, 3)).squeeze()
    std += torch.std(image, dim=(2, 3)).squeeze()
    num_samples += 1
  
  mean /= num_samples
  std /= num_samples
  
  return mean, std

# Usage example
directory = 'C:/MyPythonCoding/MyDeepLearningCoding/images_512x512'
calcmean, calcstd = calculate_mean_std(directory)
print('Mean:', calcmean)
print('Standard Deviation:', calcstd)

# Define the transformation
transform = transforms.Compose([
  transforms.Resize((pix_per_side, pix_per_side)),
  transforms.ToTensor(),
  transforms.Normalize(mean=calcmean, std=calcstd)  # Normalize RGB image using calculated estimates of mean and standard deviation
])

# Create dataset by loading the images from a directory
dataset = ImageFolder('C:/MyPythonCoding/MyDeepLearningCoding/images_512x512', transform=transform)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

print('dataloader created')

# Inspect dataloader contents, first batch only, then break
for images, _ in dataloader:
  print('dataloader first batch shape: ',images.shape)
  break

# create a class for the model
def createTheMNISTAE():
  class aenet(nn.Module):
    def __init__(self):
      super().__init__()

      self.input = nn.Linear(3*pix_per_side*pix_per_side,400)

      self.enc1 = nn.Linear(400,10)
      self.enc2 = nn.Linear(10,latent_dim)

      self.lat = nn.Linear(latent_dim,10)
      
      self.dec1 = nn.Linear(10,400)
      self.dec2 = nn.Linear(400,3*pix_per_side*pix_per_side)

    # forward pass
    def forward(self,x):
      x = F.relu( self.input(x) )
      x = F.relu( self.enc1(x) )
      x = F.relu( self.enc2(x) )
      x = F.relu( self.lat(x) )
      x = F.relu( self.dec1(x) )
      y = torch.sigmoid( self.dec2(x) )
      return y

  # create the model instance
  net = aenet()

  lossfun = nn.MSELoss()
  optimizer = torch.optim.Adam(net.parameters(),lr=.001)

  return net,lossfun,optimizer

def function2trainTheModel():
  net,lossfun,optimizer = createTheMNISTAE()       # create a new model
  
  lossEpoch = []
  for epochi in range(numepochs):
    lossBatch = []     # initialize losses
    if epochi == 0: print('training has started')
    for images, _ in dataloader:
      images = images.view(images.size(0), -1)  # Flatten the images
      yHat = net(images)              # forward pass
      loss = lossfun(yHat,images)     # loss        

      optimizer.zero_grad()           # backprop
      loss.backward()
      optimizer.step()

      lossBatch.append( loss.item() )  # losses per batch
    lossEpoch.append( np.mean(lossBatch) )
    if epochi%100==0: print (epochi,' of ',numepochs,'loss (x100): ',100 * round(lossEpoch[-1],5))

  return lossEpoch,net

# TRAIN model
#############
losses,net = function2trainTheModel()
print(f'Final loss: {losses[-1]:.5f}')

# visualize the losses
lastlosses = losses[-200:]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
# Plot 1: Model loss
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Model loss (last n epochs)')
ax1.plot(lastlosses, '.-')

# Plot 2: Log Model loss
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Log Model loss (last 100 epochs)')
ax2.plot(lastlosses, '.-')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.tight_layout()
plt.show()

# create a decoder model, that just decodes the latent representation
def fnDECODER():
  class mydecoder(nn.Module):
    def __init__(self):
      super().__init__()

      self.lat = nn.Linear(latent_dim,10)          # latent layer
      
      self.dec1 = nn.Linear(10,400)  # decoder layer
      self.out = nn.Linear(400,3*pix_per_side*pix_per_side)  # decoder layer

    # forward pass
    def forward(self,x):
      x = F.relu( self.lat(x) )
      x = F.relu( self.dec1(x) )
      y = torch.sigmoid( self.out(x) )
      return y

  # create the model instance
  decnet = mydecoder()

  # copy the weights from the trained 'net' model to 'decnet'...
  # for latent layer...
  decnet.lat.weight.data = net.lat.weight.data
  decnet.lat.bias.data = net.lat.bias.data
  # and decoder layers...
  decnet.dec1.weight.data = net.dec1.weight.data
  decnet.dec1.bias.data = net.dec1.bias.data

  decnet.out.weight.data = net.dec2.weight.data
  decnet.out.bias.data = net.dec2.bias.data

  return decnet

decnet = fnDECODER()  # call the function and create the model

#================================================================================================
def fn2DnumberSpace(x1,y1,x2,y2,steps):
  ''' create a 2D grid of values between x1,y1 and x2,y2 in n steps
   returns a tensor of size n*n x 2'''

  a= np.linspace(x1,x2,steps) #one axis of the 2D latent space
  b= np.linspace(y1,y2,steps) #the other axis of the 2D latent space
  c=[]
  for i in range(steps*steps):
    c.append((a[i%steps],b[int(i/steps)])) # create a vector of n*n latent value pairs
  #print(c)
  
  c = torch.tensor(c).float()  # convert c to tensor
  
  #check: create a rounded version of c
  d = np.round(c,1)
  print('\n(rounded) latent interpolated array: \n',d[:4],d[-4:]) # print the first and last few values
  return c  
#================================================================================================

print('checkpoint 4')

# pass the first batch of images through the encoder (2 images)
import matplotlib.pyplot as plt

images, _ = next(iter(dataloader)) # get the first batch
images = images.view(images.size(0), -1)  # Flatten the images
# pass through the encoder section of the model
latent = net.enc2(net.enc1(net.input(images)))
print('\nlatent values after passing images through trained model: \n',latent)

#convert to numpy
latent = latent.detach().numpy()

c = fn2DnumberSpace(latent[0,0],latent[0,1],latent[1,0],latent[1,1],10)  # create a 2D latent space

#plot a grid of images, but using *actual latent values* as limits
plt.figure(figsize=(13,13),)
for r in range(10*10):
  # pass values at relevant indices of c to the decoder
  remade = decnet(c[r])
  # at this point, 'remade' is a tensor, but we need to convert it to an image
  # remade is a tensor of size 1 x something, so we need to reshape it to 3 x pixelsize x pixelsize
  if r == 0: print('\nremade.shape before un-normalising: ',remade.shape)  
  # first, un-normalize the image
  #remade = remade * 0.25 + 0.25 #hack to un-normalize the image

  if r==0:print('\nremade.shape: ',remade.shape)  
  
  plt.subplot(10,10,r+1)
  plt.imshow(remade.view(3,pix_per_side,pix_per_side).permute(1,2,0).detach()) # permute to get the right dimensions
  mytitle = str(np.round(c[r][0].detach().numpy(),1))+' '+str(np.round(c[r][1].detach().numpy(),1)) 
  #plt.title(mytitle,fontsize=6)
  
  # remove axes
  plt.axis('off')
plt.tight_layout()
plt.show()

print('checkpoint 5')
