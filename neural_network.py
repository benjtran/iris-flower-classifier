"""
============================================================================
   Iris Flower Classifier - neural_network.py
   Author: Benjamin Tran
   Description: A simple neural network using PyTorch to classify Iris flowers
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd

from sklearn.model_selection import train_test_split

# Create a model class that inherits from nn.Module
class Model(nn.Module):
    def __init__(self, input_features=4, h1=8, h2=9, output_features=3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(41)

# Instantiate the model
model = Model()

# Load dataset
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Map species labels to integers
my_df['species'] = my_df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Check for unmapped labels (optional)
if my_df['species'].isnull().sum() > 0:
    raise ValueError("There are unmapped labels in 'species' column!")

# Prepare features and labels
X = my_df.drop('species', axis=1).values
Y = my_df['species'].values

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
Y_train = torch.LongTensor(Y_train)
Y_test = torch.LongTensor(Y_test)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
losses = []

model.train()  # set model to training mode

for i in range(epochs):
    Y_pred = model(X_train)     # forward pass
    loss = criterion(Y_pred, Y_train)  # compute loss
    losses.append(loss.item())  # track loss

    if i % 10 == 0:
        print(f'Epoch {i} Loss: {loss.item()}')

    optimizer.zero_grad()       # zero the gradients
    loss.backward()             # backpropagation
    optimizer.step()            # update weights

# Turn off backpropagation for evaluation
with torch.no_grad():
    Y_eval = model(X_test)  # forward pass on test set
    loss = criterion(Y_eval, Y_test)  # compute loss on test set
    print(f'Test Loss: {loss.item()}')

correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        Y_val = model.forward(data)

        # Print the predicted and actual labels
        print(f'{i + 1}.) {str(Y_val)} \t {Y_test[i]}')

        # Correct or not
        if Y_val.argmax() == Y_test[i]:
            correct += 1

print(f'We got {correct} out of {len(X_test)} correct!')  