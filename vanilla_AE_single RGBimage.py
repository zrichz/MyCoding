# vanilla autoencoder, single image dataset(!) adapted from Mike X Cohen code

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from PIL import Image
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------

# load image from the workspace directory using PIL
image = Image.open('c:/MyPythonCoding/MyDeepLearningCoding/1024x1024RGB_image_sample.png')

# rescale image to 64x64 with PIL
image_resized = image.resize((64, 64))

# display the image
plt.imshow(image_resized)
plt.show()

print('checkpoint 1')

# flatten to 1D array
image_flattened = np.array(image_resized).flatten()

# normalize the data
image_flattened = image_flattened / np.max(image_flattened)

# convert to tensor and create batch
dataT = torch.tensor(image_flattened).float().unsqueeze(0)

print(dataT.shape)
print('checkpoint 2')

# create a class for the model
def createTheMNISTAE():

  class aenet(nn.Module):
    def __init__(self):
      super().__init__()

      self.input = nn.Linear(4096*3,26)

      self.enc = nn.Linear(26,1)

      self.lat = nn.Linear(1,26)

      self.dec = nn.Linear(26,4096*3)

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
  numepochs = 400

  net,lossfun,optimizer = createTheMNISTAE()       # create a new model

  
  losses = []     # initialize losses

  for epochi in range(numepochs):
    yHat = net(dataT)              # forward pass : pass the single image through the model
    loss = lossfun(yHat,dataT)     # loss         : compare the output to the input

    optimizer.zero_grad()          # backprop
    loss.backward()
    optimizer.step()

    # losses in this epoch
    losses.append( loss.item() )
    print (epochi,' of ',numepochs)
  
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

# create a true copy of the data
X = copy.deepcopy( dataT )

# add noise
img = X.view(64,64,3)     # reshape the image

for lines in range(0,100):
  # occlude random rows or columns
  startloc = np.random.choice(range(2,62))
  if np.random.choice(range(0,1000))%2==0: # even -> horizontal occlusion
    img[startloc:startloc+1,:,:] = np.random.uniform(0, 1)
  else:      # odd -> vertical occlusion
    img[:,startloc:startloc+1,:] = np.random.uniform(0, 1)

# run X through the (TRAINED) model
###################################    
deOccluded = net(X)


# show images
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(dataT.view(64, 64, 3).detach())
axs[1].imshow(X.view(64, 64, 3).detach())
axs[2].imshow(deOccluded.view(64, 64, 3).detach())
axs[0].set_xticks([]), axs[0].set_yticks([])
axs[0].set_title('Original')
axs[1].set_title('Noisy')
axs[2].set_title('De-noised')
axs[1].set_xticks([]), axs[1].set_yticks([])
axs[2].set_xticks([]), axs[2].set_yticks([])
plt.show()

print('checkpoint 3')