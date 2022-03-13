"""
Artificial Neural Network
Artificial Neural Network (ANN), a straightforward neural network architecture, was inspired by the biological neural network. 
ANN has three layers (1) Input layer; (2) Hidden layer; (3) Output layer, the formula of ANN is y = f(Wx + b) where f is the activation function.
X is the input data, W is the weight, b is the bias, and y is the output.
"""
# Create and training an artificail neural network (ANN) using pytorch from scratch for XOR problem.

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ANN(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int) -> None:
        """
        Initialize the neural network.
        :param input_size: number of input neurons
        :param output_size: number of output neurons
        :param hidden_size: number of hidden neurons
        """
        super(ANN, self).__init__()
        self.W_1 = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_1 = nn.Parameter(torch.randn(hidden_size))
        self.W_2 = nn.Parameter(torch.randn(hidden_size, output_size))
        self.b_2 = nn.Parameter(torch.randn(output_size))
        self.activation = torch.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.
        :param x: input data
        :return: output of the neural network
        """
        x = torch.mm(x, self.W_1) + self.b_1
        x = self.activation(x)
        x = torch.mm(x, self.W_2) + self.b_2
        return x

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 1000,
        lr: float = 0.05,
        criterion: torch.nn = nn.MSELoss(),
    ) -> None:
        """
        Train the neural network.
        :param x: input data
        :param y: target data
        :param epochs: number of epochs
        :param lr: learning rate
        :param criterion: loss function
        """
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the neural network.
        :param x: input data
        :return: output of the neural network
        """
        return self.forward(x)


if __name__ == "__main__":
    x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)
    model = ANN(input_size=2, output_size=1, hidden_size=32)
    model.train(x, y, epochs=1000, lr=0.05)
    print(model.predict(torch.tensor([[0, 0]], dtype=torch.float)))
    print(model.predict(torch.tensor([[0, 1]], dtype=torch.float)))
    print(model.predict(torch.tensor([[1, 0]], dtype=torch.float)))
    print(model.predict(torch.tensor([[1, 1]], dtype=torch.float)))
