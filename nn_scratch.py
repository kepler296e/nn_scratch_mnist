import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # Load data
    data = pd.read_csv("mnist.csv", header=None, nrows=10000).to_numpy()

    # The first column is the label and the rest are 28x28=784 pixels
    X = data[:, 1:] / 255  # / max to normalize
    y = data[:, 0]

    # Split data into train, validation, and test (80%, 10%, 10%)
    train = int(len(X) * 0.8)
    val = int(len(X) * 0.1)
    X_train, y_train = X[:train], y[:train]
    X_val, y_val = X[train : train + val], y[train : train + val]
    X_test, y_test = X[train + val :], y[train + val :]

    np.random.seed(42)  # for reproducibility

    # Build model
    layers = [784, 50, 25, 10]
    model = NN(layers)

    print("Parameters:", model.count_params())

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=10,
        learning_rate=0.01,
        batch_size=64,
    )

    # Evaluate
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_val, y_val, "Validation")

    # save_model(model, "models/scratch_model.npy")


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.layer_dims = [(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.W = [np.random.randn(a, b) * np.sqrt(2 / a) for a, b in self.layer_dims]
        self.B = [np.zeros(b) for _, b in self.layer_dims]
        self.dl_dw = [np.zeros_like(w) for w in self.W]
        self.dl_db = [np.zeros_like(b) for b in self.B]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs, learning_rate, batch_size):
        start_time = time.time()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0

            remainder = len(X_train) % batch_size
            if remainder != 0:
                # truncate remainder to fit data on batches
                X_train, y_train = X_train[:-remainder], y_train[:-remainder]

            for i in range(0, len(X_train) - remainder, batch_size):
                X_batch = X_train[i : i + batch_size]  # (64, 784)
                y_batch = y_train[i : i + batch_size]  # (64,)

                # Forward pass
                Z, y_pred = self.predict(X_batch)
                # Z = [(64, 784), (64, 25), (64, 15), (64, 10)]
                # y_pred = (64, 10) probs for each batch X

                # Loss
                y_true = np.zeros((batch_size, self.layers[-1]))  # (64, 10)
                y_true[np.arange(batch_size), y_batch] = 1  # label one hot encoded
                epoch_loss += cross_entropy(y_true, y_pred) / batch_size

                # Backprop
                self.dl_dw, self.dl_db = self.backprop(Z, y_true, y_pred)

                # Update params
                self.update_params(learning_rate)

            print("Epoch:", epoch, "Loss:", epoch_loss)
            loss_history.append(epoch_loss)

        plot_loss(loss_history)
        plot_loss_derivative(loss_history)
        print("Time:", time.time() - start_time)

    def predict(self, X):
        batch_size = X.shape[0]
        Z = [np.zeros((batch_size, c)) for c in self.layers]

        # Input layer
        Z[0] = X

        # Hidden layers
        for i in range(len(self.layers) - 2):  # 0, 1
            Z[i + 1] = relu(Z[i] @ self.W[i] + self.B[i])

        # Output layer
        Z[3] = Z[2] @ self.W[2] + self.B[2]

        return Z, softmax(Z[3])

    def backprop(self, Z, y_true, y_pred):
        dZ = [np.zeros_like(z) for z in Z]

        # Remember that:
        # layers = [784, 50, 25, 10]
        # batch_size = 64, so:
        # Z (activations) = [(64, 784), (64, 50), (64, 25), (64, 10)]
        # Weights = [(784, 50), (50, 25), (25, 10)]
        # Biases = [(50,), (25,), (10,)]

        # dZ is the partial derivative of the loss function with respect to the activations

        # Output layer
        dZ[-1] = y_pred - y_true  # (64, 10)
        self.dl_dw[-1] = np.einsum("bi,bj->ji", dZ[-1], Z[-2])  # (64, 10) @ (64, 25) => (25, 10)
        self.dl_db[-1] = np.sum(dZ[-1], axis=0)  # (64, 10) => (10,)

        """
        dZ-1 = loss/softmax * softmax/Z-1
        dW-1 = dZ-1 * Z-2
        dB-1 = dZ-1

        dZ-2 = dZ-1 * W-1 * dRelu(Z-2)
        dW-2 = dZ-2 * Z-3
        dB-2 = dZ-2
        """

        # Hidden layers
        for i in range(-2, -len(self.layers), -1):  # -2, -3
            dZ[i] = np.einsum("ij,bj->bi", self.W[i + 1], dZ[i + 1]) * relu_derivative(Z[i])
            self.dl_dw[i] = np.einsum("bi,bj->ji", dZ[i], Z[i - 1])
            self.dl_db[i] = np.sum(dZ[i], axis=0)

        # When i = -2:
        # (25, 10) @ (64, 10) => (64, 25)
        # (64, 25) @ (64, 50) => (50, 25)
        # (64, 25) => (25,)

        return self.dl_dw, self.dl_db

    def update_params(self, learning_rate):
        for i in range(len(self.layers) - 1):  # 0, 1, 2
            self.W[i] -= learning_rate * self.dl_dw[i]
            self.B[i] -= learning_rate * self.dl_db[i]

    def count_params(self):
        return sum(np.prod(w.shape) for w in self.W) + sum(np.prod(b.shape) for b in self.B)


def relu(X):
    return np.where(X > 0, X, 0)


def relu_derivative(X):
    return np.where(X > 0, 1, 0)


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))  # 1e-8 to avoid ln(0)


def softmax(logits):
    exp_logits = np.exp(logits)  # (64, 10)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (64, 10) / (64, 1)


def evaluate(model: NN, X: np.ndarray, y: np.ndarray, data):
    _, y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)  # (64, 10) => (64,)
    accuracy = np.sum(y_pred == y) / len(y)
    print(data, "Accuracy:", accuracy)


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
