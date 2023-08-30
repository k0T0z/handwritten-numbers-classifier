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

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CTDataset(Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(filepath)

        self.x = self.dataset.iloc[:, 1:].values  # all columns except the first one
        self.y = self.dataset.iloc[:, 0].values  # first column is the label

        # Reshape features into images (assuming images are 28x28 pixels)
        self.x = self.x.reshape(-1, 28, 28)

        # Convert to PyTorch tensors
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

        # Normalize image pixels
        self.x = self.x / 255.0

        # Encoding labels into one-hot vectors
        self.y = F.one_hot(self.y, num_classes=10).to(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds = CTDataset("mnist_train.csv")
test_ds = CTDataset("mnist_test.csv")

# print(len(train_ds))
# print(len(test_ds))

# Process data in batches
train_dl = DataLoader(train_ds, batch_size=5)
test_dl = DataLoader(test_ds, batch_size=5)


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2, 100)  # bias=False
        self.Matrix2 = nn.Linear(100, 50)  # bias=False
        self.Matrix3 = nn.Linear(50, 10)  # bias=False
        self.R = nn.ReLU()

    def forward(self, x):
        # Converting images from matrices to vectors (AKA flattening)
        x = x.view(-1, 28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze()


f = MyNeuralNet()


# Train model function
def train_model(dl, f, n_epochs=20, lr=0.01):
    # Optimization
    opt = sgd(f.parameters(), lr=lr)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()
            # Store training data for plotting
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)


epoch_data, loss_data = train_model(train_dl, f)

# Visualize the neural network
dummy_input = train_ds[0][0].unsqueeze(0)
torchviz.make_dot(f(dummy_input), params=dict(f.named_parameters())).render(
    "nn", format="png"
)

plt.plot(epoch_data, loss_data)
plt.xlabel("Epoch Number")
plt.ylabel("Cross Entropy")
plt.title("Cross Entropy (per batch)")
plt.savefig("plt.png")
plt.show()

# TODO: This does not work!
# temp_global_loss_data = np.array(global_loss_data).reshape(20,-1).mean(axis=1)
# temp_global_time_stamps = np.array(global_time_stamps).reshape(20,-1).mean(axis=1)

# plt.plot(temp_global_time_stamps, temp_global_loss_data, marker='o', label='Loss')
# plt.xlabel("Time (s)")
# plt.ylabel("MSE")
# plt.title("MSE (per batch)")
# plt.legend()
# plt.savefig("plt.png")

# test_x, test_y = test_ds[:]
# yhats = f(test_x).argmax(axis=1)

# fig, ax = plt.subplots(10,4,figsize=(10,15))
# for i in range(40):
#     plt.subplot(10,4,i+1)
#     plt.imshow(test_x[i])
#     plt.title(f'Predicted Digit: {yhats[i]}')
# fig.tight_layout()
# plt.show()
