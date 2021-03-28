import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.convolution_layer1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.activation_function1 = nn.ReLU()

        # Pooling 1
        self.pooling1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.convolution_layer2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.activation_function2 = nn.ReLU()

        # Pooling 2
        self.pooling2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1
        self.fully_connected_layer1 = nn.Linear(32 * 5 * 5, 10)

    def forward(self, x):
        # Layer 1
        output = self.convolution_layer1(x)
        output = self.activation_function1(output)
        output = self.pooling1(output)

        # Layer 2
        output = self.convolution_layer2(output)
        output = self.activation_function2(output)
        output = self.pooling2(output)

        # Flatten
        output = output.view(output.size(0), -1)

        # Layer 3
        output = self.fully_connected_layer1(output)

        return output
