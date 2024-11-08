# Implementing an Autoencoder in PyTorch
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# We set the batch size, the number of training epochs, and the learning rate.
batch_size = 512
epochs = 6
learning_rate = 1e-3

# We load our MNIST dataset using the `torchvision` package. 
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
# ## Autoencoder
# An autoencoder is a type of neural network that finds the function mapping the features x to itself. This objective is known as reconstruction, and an autoencoder accomplishes this through the following process: (1) an encoder learns the data representation in lower-dimension space, i.e. extracting the most salient features of the data, and (2) a decoder learns to reconstruct the original data based on the learned representation by the encoder.
# We define our autoencoder class with fully connected layers for both its encoder and decoder components.
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

# Before using our defined autoencoder class, we have the following things to do:
#     1. We configure which device we want to run on.
#     2. We instantiate an `AE` object.
#     3. We define our optimizer.
#     4. We define our reconstruction loss.

device = torch.device("cpu") # stick to cpu for now

# create a model from `AE` autoencoder class
model = AE(input_shape=784).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lossfn = nn.MSELoss()

for epoch in range(epochs):
    loss = 0
    for data, _ in train_loader:
        data = data.view(-1, 784).to(device) # flatten the image data to 1D tensor
        
        yHat = model(data) # forward pass
        
        train_loss = lossfn(yHat, data) # compute the loss
        
        train_loss.backward() # backprop
        optimizer.step()
        optimizer.zero_grad()

        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.4f}".format(epoch + 1, epochs, loss))

# Let's extract some test examples to reconstruct using our trained autoencoder.
test_dataset = torchvision.datasets.MNIST( root="~/torch_datasets", train=False, transform=transform, download=True)

test_loader = torch.utils.data.DataLoader(  test_dataset, batch_size=10, shuffle=False)

test_examples = None

with torch.no_grad():
    for data in test_loader:
        data = data[0]
        test_examples = data.view(-1, 784)
        reconstruction = model(test_examples)
        break

# ## Visualize Results
# reconstruct some test images using our trained autoencoder.

number = 10
plt.figure(figsize=(20, 4))
for index in range(number):
    # display original
    ax = plt.subplot(2, number, index + 1)
    plt.imshow(test_examples[index].numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, number, index + 1 + number)
    plt.imshow(reconstruction[index].numpy().reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()