import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from ann import ANN


# Read the data from the CSV file
red_data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

white_data_path = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


class WineDataReader():
    def __init__(self, red_data_path: str, white_data_path: str) -> None:
        self.red_data_path = red_data_path
        self.white_data_path = white_data_path

    def _read_data(self) -> tuple:
        """
        Read the data from the CSV file
        :param red_data_path: path to the red data
        :param white_data_path: path to the white data
        :return: red and white data
        """
        red = pd.read_csv(self.red_data_path, sep=";")
        white = pd.read_csv(self.white_data_path, sep=";")
        return red, white
    
    def _create_label(self, red: pd.DataFrame, white: pd.DataFrame) -> tuple:
        """
        Create label for 2 dataset with red = 1 and white = 0 and merge them.
        :param red: red data
        :param white: white data
        :return: red and white data
        """
        red["label"] = 1
        white["label"] = 0
        dataset = pd.concat([red, white], ignore_index=True)
        return dataset

    def _split_data(self, dataset: pd.DataFrame) -> tuple:
        """
        Split the dataset into training, develop and test set with 80%/10%/10% ratio.
        :param dataset: dataset
        :return: training, develop and test set
        """
        train, dev, test = np.split(
            dataset.sample(frac=1), [int(0.8 * len(dataset)), int(0.9 * len(dataset))]
        )
        return train, dev, test

    def create_dataset(self) -> tuple:
        """
        Create the input and target tensors for the neural network.
        :return: input and target tensors
        """
        red, white = self._read_data()
        dataset = self._create_label(red, white)
        train, dev, test = self._split_data(dataset)
        x_train, y_train = (
            torch.from_numpy(train.drop(["label"], axis=1).values).float(),
            torch.from_numpy(train["label"].values.reshape(-1, 1)).float(),
        )
        x_val, y_val = (
            torch.from_numpy(dev.drop(["label"], axis=1).values).float(),
            torch.from_numpy(dev["label"].values.reshape(-1, 1)).float(),
        )
        x_test, y_test = (
            torch.from_numpy(test.drop(["label"], axis=1).values).float(),
            torch.from_numpy(test["label"].values.reshape(-1, 1)).float(),
        )
        return x_train, y_train, x_val, y_val, x_test, y_test

# Create the ANN architecture with validation set.
class ANNClassifier(ANN):
    """
    ANN Classifier with validation set.
    """

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int = 64
    ) -> None:
        """
        Initialize the neural network.
        :param input_size: input size
        :param output_size: output size
        :param hidden_size: hidden size
        """
        super().__init__(input_size, output_size, hidden_size)

    # Override the train method to use validation set.
    def train(
        self,
        training_set=(x_train, y_train),
        validation_set=(x_val, y_val),
        batch_size: int = 32,
        epochs: int = 1000,
        lr: float = 0.05,
        criterion: torch.nn = nn.CrossEntropyLoss(),
    ) -> None:
        """
        Train the neural network and validate the model with the validation set.
        :param training_set: training set
        :param validation_set: validation set
        :param batch_size: batch size
        :param epochs: number of epochs
        :param lr: learning rate
        :param criterion: loss function
        """
        optimizer = optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            for i in range(0, len(training_set[0]), batch_size):
                # Calculate the training loss and validation loss.
                y_train_pred = self.forward(training_set[0][i : i + batch_size])
                train_loss = criterion(
                    y_train_pred, training_set[1][i : i + batch_size]
                )
                # Update the weights and biases.
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                # Print the training loss and validation loss each 100 epochs.
            if epoch % 100 == 0:
                y_val_pred = self.forward(validation_set[0])
                val_loss = criterion(y_val_pred, validation_set[1])
                print(
                    f"Epoch: {epoch}, Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}"
                )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the label of the test set.
        """
        return super().forward(x)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Evaluate the model with the test set.
        :param x: test set
        :param y: test label
        :return: accuracy
        """
        y_pred = self.predict(x)
        y_pred = (y_pred > 0.5).float()
        return (y_pred == y).sum().item() / len(y)


if __name__ == "__main__":
    # Calculate the accuracy of the model on the test set.
    x_train, y_train, x_val, y_val, x_test, y_test = WineDataReader(red_data_path, white_data_path).create_dataset()
    model = ANNClassifier(input_size=12, output_size=1, hidden_size=64)
    model.train(
        training_set=(x_train, y_train),
        validation_set=(x_val, y_val),
        batch_size=128,
        epochs=1000,
        lr=0.05,
        criterion=nn.MSELoss(),
    )
    y_pred = model.predict(x_test)
    # Convert the prediction to 0 and 1.
    y_pred = (y_pred > 0.5).float()
    print(f"Test accuracy: {(y_pred == y_test).sum().item() / len(y_test)}")
