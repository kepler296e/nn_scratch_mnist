import pandas as pd
import numpy as np
import tensorflow as tf
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(pytorch=False):
    mnist = pd.read_csv("mnist_train.csv", header=None).to_numpy()

    # First column is the label, the rest are 28x28=784 pixels
    X, y = mnist[:, 1:], mnist[:, 0]

    # Normalize dividing by max
    X = X / 255

    # Split into train and validation
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    tf.keras.utils.set_random_seed(42)

    if pytorch:
        return (
            torch.tensor(X_train, dtype=torch.float32, device=DEVICE),
            torch.tensor(y_train, dtype=torch.long, device=DEVICE),
            torch.tensor(X_val, dtype=torch.float32, device=DEVICE),
            torch.tensor(y_val, dtype=torch.long, device=DEVICE),
        )
    else:
        return X_train, y_train, X_val, y_val
