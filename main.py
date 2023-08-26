import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn as nn
from torch.optim import SGD as sgd
import pandas as pd
import torchviz

# dataset = pd.read_csv('mnist_train.csv')
# x = dataset.iloc[:, 1:].values # all columns except the first one
# y = dataset.iloc[:, 0].values # first column is the label

x = torch.tensor([[6,2],[5,2],[1,3],[7,6]]).float()
y = torch.tensor([1,5,2,5]).float()

# print(x.shape)
# print(y.shape)

class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(2,80) # bias=False
        self.Matrix2 = nn.Linear(80,80) # bias=False
        self.Matrix3 = nn.Linear(80,1) # bias=False
        self.R = nn.ReLU()
    def forward(self,x):
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()
    

# Test model before training
f = MyNeuralNet()
yhat = f(x)

L = nn.MSELoss()
loss = L(yhat,y)

print("Current output: ")
print(yhat)
print("Excepted output: ")
print(y)
print("Currect loss: ")
print(loss)

def train_model(x,y,f, n_epochs=50, lr=0.001):
    opt = sgd(f.parameters(), lr=lr) # Sofisticated Gradient Descent
    L = nn.MSELoss()

    # Train model
    losses = []
    for _ in range(n_epochs):
        opt.zero_grad() # flush previous epoch's gradient
        loss_value = L(f(x), y) #compute loss
        loss_value.backward() # compute gradient
        opt.step() # Perform iteration using gradient above
        losses.append(loss_value.item())
    return f, losses

# Train model
f, losses = train_model(x,y,f, n_epochs=10000, lr=0.001)

# Test model after training
yhat = f(x)

L = nn.MSELoss()
loss = L(yhat,y)

print("New output: ")
print(yhat)
print("Excepted output: ")
print(y)
print("New loss: ")
print(loss)

# Visualize the neural network
dummy_input = torch.tensor([[6.0, 2.0]])
torchviz.make_dot(f(dummy_input), params=dict(f.named_parameters())).render("nn", format="png")

# Plot loss
plt.plot(losses)
plt.ylabel('Loss $L(y,\hat{y};a)$')
plt.xlabel('Epochs')
plt.savefig('plt.png')
plt.show()


