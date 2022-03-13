import numpy as np
import pandas as pd
import torch
from .ann import ANN


# Read the data from the CSV file

red = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    sep=";",
)
white = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    sep=";",
)

# Create label for 2 dataset with red = 1 and white = 0 and merge them.
red["label"] = 1
white["label"] = 0
dataset = pd.concat([red, white], ignore_index=True)

# Split the dataset into training, develop and test set with 80%/10%/10% ratio.

train, dev, test = np.split(
    dataset.sample(frac=1), [int(0.8 * len(dataset)), int(0.9 * len(dataset))]
)
print(len(train), len(dev), len(test))
