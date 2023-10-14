import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # Training data
    train = pd.read_csv("mnist_train.csv", header=None)[:5000]
    X_train, y_train = train.iloc[:, 1:].values / 255, train.iloc[:, 0].values

    # Test data
    test = pd.read_csv("mnist_test.csv", header=None)[:500]
    X_test, y_test = test.iloc[:, 1:].values / 255, test.iloc[:, 0].values

    # Build model
    layers = [784, 25, 15, 10]
    model = NN(layers)

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=30,
        learning_rate=0.01,
        batch_size=1,
    )

    # Test model
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", np.mean(y_pred == y_test))

    save_model(model, "models/scratch_model.npy")


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.layer_dims = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.W = [np.random.randn(b, a) * np.sqrt(2 / a) for a, b in self.layer_dims]
        self.B = [np.zeros(b) for _, b in self.layer_dims]
        self.Z = [np.zeros(c) for c in layers]
        self.dl_dz = self.Z.copy()
        self.dl_dw = self.W.copy()
        self.dl_db = self.B.copy()

    def fit(self, X_train, y_train, epochs, learning_rate, batch_size):
        start = time.time()
        loss_history = []
        for epoch in range(epochs):
            epoch_loss = 0
            for x_index in range(len(X_train)):
                """FORWARD"""
                y_pred = self.predict(X_train[x_index])
                y_true = np.zeros(self.layers[-1])
                y_true[y_train[x_index]] = 1

                loss = cross_entropy(y_true, y_pred)
                epoch_loss += loss

                """BACKPROP"""
                # Output layer
                self.dl_dz[-1] = y_pred - y_true
                self.dl_dw[-1] = np.outer(self.dl_dz[-1], self.Z[-2])
                self.dl_db[-1] = self.dl_dz[-1]

                # Hidden layers
                for i in range(-2, -len(self.layers), -1):
                    self.dl_dz[i] = np.dot(self.W[i + 1].T, self.dl_dz[i + 1]) * relu_derivative(self.Z[i])
                    self.dl_dw[i] = np.outer(self.dl_dz[i], self.Z[i - 1])
                    self.dl_db[i] = self.dl_dz[i]

                # Update weights and biases
                for i in range(len(self.layers) - 2):
                    self.W[i] -= learning_rate * self.dl_dw[i] / batch_size
                    self.B[i] -= learning_rate * self.dl_db[i] / batch_size

            print("Epoch:", epoch, "Loss:", epoch_loss)
            loss_history.append(epoch_loss)

        plot_loss(loss_history)
        plot_loss_derivative(loss_history)
        print("Time:", time.time() - start)

    def predict(self, X):
        # Input layer
        self.Z[0] = X

        # Hidden layers
        for i in range(len(self.layers) - 2):
            self.Z[i + 1] = relu(np.dot(self.Z[i], self.W[i].T) + self.B[i])

        # Output layer
        self.Z[-1] = np.dot(self.Z[-2], self.W[-1].T) + self.B[-1]
        y_pred = softmax(self.Z[-1])

        return y_pred


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def relu(x):
    return np.where(x > 0, x, 0)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


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
    model = NN(model_data["layers"])
    model.W = model_data["W"]
    model.B = model_data["B"]
    return model


if __name__ == "__main__":
    main()
