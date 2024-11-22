import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load the model
MODEL_PATH = 'model/CNN_RegDrop.pt'

class CNN_RegDrop(nn.Module):
    def __init__(self):
        super(CNN_RegDrop, self).__init__()

        # Input: (batch_size, 1, 128, 431)

        # BatchNormalization for input
        self.batch_norm1 = nn.BatchNorm2d(1)

        # First Conv2D layer with reduced filters (16 filters, kernel size 7x7)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm2 = nn.BatchNorm2d(16)  # BatchNormalization after conv1

        # Second Conv2D layer with reduced filters (32 filters, kernel size 3x3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.batch_norm3 = nn.BatchNorm2d(32)  # BatchNormalization after conv2

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layer with L2 regularization and Dropout
        # L2 regularization in PyTorch is done using weight decay in the optimizer (not here)
        self.fc1 = nn.Linear(97440, 64)
        self.dropout = nn.Dropout(0.5)  # 50% dropout
        self.batch_norm4 = nn.BatchNorm1d(64)  # BatchNormalization for fully connected layer

        # Output layer with softmax for classification (6 classes)
        self.fc2 = nn.Linear(64, 6)

    def forward(self, x):
        # Input shape: (batch_size, 1, 128, 431)

        # First conv block: Conv2d -> ReLU -> MaxPool2d -> BatchNorm
        x = self.batch_norm1(x)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batch_norm2(x)

        # Second conv block: Conv2d -> ReLU -> MaxPool2d -> BatchNorm
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.batch_norm3(x)

        # Flatten the output
        x = self.flatten(x)

        # Fully connected layer with dropout and batch norm
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.batch_norm4(x)

        # Output layer with softmax
        x = F.softmax(self.fc2(x), dim=1)  # Softmax for multi-class classification

        return x


