# Mnist 28x28 images of handwritten digits 0-9
# 60,000 training images and 10,000 test images
# 10 classes (0-9)

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn

# Load the data
train_data = torchvision.datasets.MNIST(root='/data', train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.MNIST(root='/data', train=False, download=True, transform=torchvision.transforms.ToTensor())

# Create the data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define the model
model = nn.Sequential(nn.Linear(784, 500),
                        nn.ReLU(),
                        nn.Linear(500, 100),
                        nn.ReLU(),
                        nn.Linear(100, 10),
                        nn.Softmax(dim=1))

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Train the model
epochs = 5
for e in range(epochs):
    
    running_loss = 0
    for images, labels in train_loader:
        
        # Flatten the images
        images = images.view(images.shape[0], -1)
        
        # Clear the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(images)
        
        # Calculate the loss
        loss = loss_fn(output, labels)
        
        # Backward pass
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        print(f"Training loss: {running_loss/len(train_loader)}")
        
# Test the model
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

# Turn off gradients
with torch.no_grad():
        
        model.eval()
        
        total_correct = 0
        for images, labels in test_loader:
            
            # Flatten the images
            images = images.view(images.shape[0], -1)
            
            # Forward pass
            output = model(images)
            
            # Get the predictions
            ps = torch.exp(output)
            
            # Get the top prediction
            top_p, top_class = ps.topk(1, dim=1)
            
            # Get the number of correct predictions
            equals = top_class == labels.view(*top_class.shape)
            
            # Sum the correct predictions
            total_correct += torch.sum(equals.type(torch.FloatTensor))
            
        else:
            print(f"Test accuracy: {100*total_correct.item()/len(test_loader.dataset)}%")


# Save the model
torch.save(model.state_dict(), 'C:/Users/katti/Desktop/Python-Projects/pytorch_tutorials/draw_digit/mnist_model.pth')

# thx copliot :)