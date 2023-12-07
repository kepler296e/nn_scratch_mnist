import pandas as pd
import numpy as np

mnist = pd.read_csv("mnist_train.csv").to_numpy()

X, y = mnist[:, 1:], mnist[:, 0]

X = X / 255

train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

np.random.seed(42)
