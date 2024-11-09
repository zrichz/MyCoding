"""tqdm_test_MNIST.ipynb
### MNIST with PyTorch and tqdm
###code adapted from DUDL L121 - MNIST basic vanilla, with train sample sizes configurable
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# import MNIST (comes with colab, but this is in my workspace)
MNISTdata10k = np.loadtxt(open('mnist_10k.csv','rb'),delimiter=',')
print('shape of MNIST data: ',MNISTdata10k.shape) # 10k samples, 785 columns (1st=number ID, rest=pixels)

def fnCreateDataLoaders():
  labels = MNISTdata10k[:,0]       # extract labels (number IDs) and remove from data
  data   = MNISTdata10k[:,1:]

  dataNorm = data / np.max(data)  # normalize the data to a range of [0 1]

  dataT   = torch.tensor( dataNorm ).float()     # convert to tensors
  labelsT = torch.tensor( labels ).long()

  # sklearn SPLIT...
  train_data,test_data, train_labels,test_labels = train_test_split(dataT, labelsT, train_size=.9)

  # convert into PyTorch Datasets, DataLoaders...
  train_data = TensorDataset(train_data,train_labels)
  test_data  = TensorDataset(test_data,test_labels)

  train_loader = DataLoader(train_data,batch_size=20,shuffle=True,drop_last=True)
  test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0]) # all in one batch

  return train_loader,test_loader


"""Create the DL model CLASS wrapped in a function..."""
def fnCreateModel(finalHidden=12):
  class mnistNet(nn.Module):
    def __init__(self):
      super().__init__()
      self.finalHidden = finalHidden
      self.input = nn.Linear(784,34) ### input layer
      self.fc1 = nn.Linear(34,32)    ### hidden layer
      self.fc2 = nn.Linear(32,finalHidden)    ### final hidden layer
      self.output = nn.Linear(finalHidden,10) ### output layer

    # forward pass
    def forward(self,x):
      x = F.relu( self.input(x) )
      x = F.relu( self.fc1(x) )
      x = F.relu( self.fc2(x) )
      return self.output(x)

  net = mnistNet() # create an INSTANCE of this CLASS

  lossfun = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters(),lr=.002)

  return net,lossfun,optimizer

"""function that trains the model..."""
def fnTrainModel(train_loader, test_loader, numepochs=10, fh=12): # default to 10 epochs
  net,lossfun,optimizer = fnCreateModel(fh)    # call fn to create a new instance of our model

  losses    = torch.zeros(numepochs)         # initialize losses
  trainAcc  = []
  testAcc   = []

  for epochi in range(numepochs):
    batchAcc  = []
    batchLoss = []
    for X,y in train_loader:
      yHat = net(X)
      loss = lossfun(yHat,y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      batchLoss.append(loss.item())

      matches = torch.argmax(yHat,axis=1) == y     # booleans (false/true)
      matchesNumeric = matches.float()             # convert to numbers (0/1)
      accuracyPct = 100*torch.mean(matchesNumeric) # average and x100
      batchAcc.append( accuracyPct )               # add to list of accuracies
    trainAcc.append( np.mean(batchAcc) )
    losses[epochi] = np.mean(batchLoss)

    # calc accuracy on TEST set
    X,y = next(iter(test_loader)) # extract X,y from test dataloader
    with torch.no_grad():
      yHat = net(X)
    testAcc.append( 100*torch.mean((torch.argmax(yHat,axis=1)==y).float()) )
    if epochi <5 or epochi%5 == 0: # print first 5 epochs, then every 5th
      print(f'epoch {epochi}, loss={losses[epochi]:.4f}, trainAcc={trainAcc[-1]:.2f}, testAcc={testAcc[-1]:.2f}')
  return trainAcc,testAcc,losses,net

# main code...
def main():
  # create data loaders...
  train_loader,test_loader = fnCreateDataLoaders()
  # train the model...
  for finalHidden in [9,10,11]:
    tick = time.time()
    print(f'\n\nfinalHidden={finalHidden}')
    trainAcc, testAcc, losses, net = fnTrainModel(train_loader, test_loader, 12, finalHidden) # train for n epochs, with 12 hidden units
    tock = time.time()
    print(f'\n\nfinalHidden={finalHidden}, elapsed time: {tock-tick:.2f} sec')
    #store results for plotting
    if finalHidden == 9:
      trainAcc_A,testAcc_A,losses_A = trainAcc,testAcc,losses
    elif finalHidden == 10:
      trainAcc_B,testAcc_B,losses_B = trainAcc,testAcc,losses
    elif finalHidden == 11:
      trainAcc_C,testAcc_C,losses_C = trainAcc,testAcc,losses
  # plot results...
  fig,ax = plt.subplots(1,2,figsize=(12,5))
  ax[0].plot(trainAcc_A,'s-',label='train (9)',markersize=3)
  ax[0].plot(testAcc_A,'s--',label='test (9)',markersize=3,alpha=.5)
  ax[0].plot(trainAcc_B,'s-',label='train (10)',markersize=3)
  ax[0].plot(testAcc_B,'s--',label='test (10)',markersize=3,alpha=.5)
  ax[0].plot(trainAcc_C,'s-',label='train (11)',markersize=3)
  ax[0].plot(testAcc_C,'s--',label='test (11)',markersize=3,alpha=.5)
  ax[0].set_xlabel('epoch')
  ax[0].set_ylabel('accuracy')
  ax[0].set_ylim([70,100])
  ax[0].legend()

  ax[1].plot(losses_A,'s-',markersize=3)
  ax[1].plot(losses_B,'s-',markersize=3)
  ax[1].plot(losses_C,'s-',markersize=3)
  ax[1].set_xlabel('epoch')
  ax[1].set_ylabel('loss')
  plt.tight_layout()
  plt.show()

if __name__ == "__main__": # only run this if this script is called as the main script
  main()