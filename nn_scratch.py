import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # load data
    data = pd.read_csv("mnist_train.csv", header=None, nrows=5000).to_numpy()

    X = data[:, 1:] / 255  # normalize
    y = data[:, 0]

    # split data
    train = int(len(X) * 0.8)
    val = int(len(X) * 0.1)
    X_train, y_train = X[:train], y[:train]
    X_val, y_val = X[train : train + val], y[train : train + val]
    X_test, y_test = X[train + val :], y[train + val :]

    np.random.seed(42)  # make it reproducible

    # build model
    batch_size = 1
    layers = [X.shape[1], 25, 15, 10]  # 784, 25, 15, 10
    model = NN(layers, batch_size)

    # train
    model.fit(
        X_train,
        y_train,
        epochs=30,
        learning_rate=0.01,
    )

    # evaluate
    NN.batch_size = 1
    y_pred = model.predict(X_val)
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", np.mean(y_pred == y_val))

    save_model(model, "models/scratch_model.npy")


class NN:
    def __init__(self, layers, batch_size):
        self.layers = layers
        self.batch_size = batch_size
        self.layer_dims = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.W = [np.random.randn(batch_size, a, b) * np.sqrt(2 / a) for a, b in self.layer_dims]  # (784, 25), (25, 15), (15, 10)
        self.B = [np.zeros((batch_size, b)) for _, b in self.layer_dims]
        self.Z = [np.zeros((batch_size, c)) for c in layers]
        self.dl_dz = self.Z.copy()
        self.dl_dw = self.W.copy()
        self.dl_db = self.B.copy()

    def fit(self, X_train, y_train, epochs, learning_rate):
        X_train = X_train.reshape(-1, self.batch_size, X_train.shape[1])  # reshape to (batch_size, features)

        start_time = time.time()
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(len(X_train)):
                # forward pass
                y_pred = self.forward_propagation(X_train[i])  # (32, 10)

                y_true = np.zeros((self.batch_size, self.layers[-1]))  # (32, 10)
                y_true[np.arange(self.batch_size), y_train[i]] = 1  # (32, 10) one hot encode
                batch_loss = cross_entropy(y_true, y_pred)  # (32,) => mean)

                epoch_loss += batch_loss

                # backprop
                self.dl_dw, self.dl_db = self.backward_propagation(y_true, y_pred)

                # update
                self.update_parameters(learning_rate)

            print("Epoch:", epoch, "Loss:", epoch_loss)
            loss_history.append(epoch_loss)

        # plot_loss(loss_history)
        # plot_loss_derivative(loss_history)
        print("Time:", time.time() - start_time)

    def forward_propagation(self, X):
        # input layer
        self.Z[0] = X  # (32, 784)

        # hidden layers
        for i in range(len(self.layers) - 1):  # 2
            self.Z[i + 1] = relu(np.einsum("bi,bij->bj", self.Z[i], self.W[i]) + self.B[i])  # einsum does @ but over the whole batch
            # i = 0 => (32, 25) = relu((32, 784) @ (32, 784, 25) + (32, 25))
            # i = 1 => (32, 15) = relu((32, 25) @ (32, 25, 15) + (32, 15))

        # ouput layer
        self.Z[-1] = np.einsum("bi,bij->bj", self.Z[-2], self.W[-1]) + self.B[-1]  # linear activation
        # (32, 10) = (32, 15) @ (32, 15, 10) + (32, 10)
        y_pred = softmax(self.Z[-1])

        return y_pred

    def backward_propagation(self, y_true, y_pred):
        # y_true and y_pred shapes are (32, 10)

        # Output layer
        self.dl_dz[-1] = y_pred - y_true  # (32, 10)
        self.dl_dw[-1] = np.einsum("bi,bj->bji", self.dl_dz[-1], self.Z[-2])  # (32, 10) @ (32, 15) => (32, 15, 10)
        self.dl_db[-1] = self.dl_dz[-1]  # (32, 10) = (32, 10)

        # Hidden layers
        for i in range(-2, -len(self.layers), -1):  # goes through -2, -3
            self.dl_dz[i] = np.einsum("bi,bji->bj", self.dl_dz[i + 1], self.W[i + 1]) * relu_derivative(self.Z[i])
            self.dl_dw[i] = np.einsum("bi,bj->bji", self.dl_dz[i], self.Z[i - 1])
            self.dl_db[i] = self.dl_dz[i]
            # i = -2 =>
            # (32, 15) = ((32, 10) @ (32, 15, 10)) * (32, 15)
            # (32, 15) @ (32, 25) => (32, 25, 15)
            # (32, 15) = (32, 15)

        return self.dl_dw, self.dl_db

    def update_parameters(self, learning_rate):
        for i in range(len(self.layers) - 2):
            self.W[i] -= learning_rate * self.dl_dw[i] / self.batch_size
            self.B[i] -= learning_rate * self.dl_db[i] / self.batch_size

    def predict(self, X):
        return self.forward_propagation(X)


def cross_entropy(y_true, y_pred):
    # y_true and y_pred shapes are (32, 10)
    # the sum must be done over each row
    return -np.sum(y_true * np.log(y_pred + 1e-8))  # 1e-8 to avoid log(0)


def softmax(X):
    X = X - np.max(X, axis=1, keepdims=True)  # normalize
    exps = np.exp(X)  # (32, 10)
    return exps / np.sum(exps, axis=1, keepdims=True)  # (32, 10) / (32, 1)


def relu(X):
    # X shape (32, n)
    return np.where(X > 0, X, 0)


def relu_derivative(X):
    # X shape (32, n)
    return np.where(X > 0, 1, 0)


def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_loss_derivative(loss_history):
    plt.plot(np.gradient(loss_history))
    plt.xlabel("Epoch")
    plt.ylabel("Loss Derivative")
    plt.show()


def save_model(model, filepath):
    model_data = {"layers": model.layers, "W": model.W, "B": model.B}
    np.save(filepath, model_data)


def load_model(filepath):
    model_data = np.load(filepath, allow_pickle=True).item()
    model = NN(model_data["layers"], 1)
    model.W = model_data["W"]
    model.B = model_data["B"]
    return model


if __name__ == "__main__":
    main()
