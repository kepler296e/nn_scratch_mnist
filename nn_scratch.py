import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def main():
    # Load data
    data = pd.read_csv("mnist.csv", header=None).to_numpy()

    # The first column is the label and the rest are 28x28=784 pixels
    X = data[:, 1:] / 255  # / max to normalize
    y = data[:, 0]

    # Split train and validation
    train = int(len(X) * 0.8)
    val = int(len(X) * 0.2)
    X_train, y_train = X[:train], y[:train]
    X_val, y_val = X[train : train + val], y[train : train + val]

    np.random.seed(42)

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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs, learning_rate, batch_size):
        start_time = time.time()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0

            remainder = len(X_train) % batch_size
            if remainder != 0:
                # Add extra samples to make it divisible by batch_size
                X_train = np.concatenate((X_train, X_train[: batch_size - remainder]))
                y_train = np.concatenate((y_train, y_train[: batch_size - remainder]))

            for i in range(0, len(X_train) - remainder, batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Forward pass
                Z, y_pred = self.predict(X_batch)

                # Loss
                y_true = np.zeros((batch_size, self.layers[-1]))
                y_true[np.arange(batch_size), y_batch] = 1  # label one hot encoded
                epoch_loss += cross_entropy(y_true, y_pred) / batch_size

                # Backprop
                dW, dB = self.backprop(Z, y_true, y_pred)

                # Update params
                self.update_params(learning_rate, dW, dB)

            print("Epoch:", epoch, "Loss:", epoch_loss)
            loss_history.append(epoch_loss)

        # plot_loss(loss_history)
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
        dW = [np.zeros_like(w) for w in self.W]
        dB = [np.zeros_like(b) for b in self.B]

        # Supposing batch_size=64 and layers=[784, 50, 25, 10], then:
        # Z = [(64, 784), (64, 50), (64, 25), (64, 10)]
        # W = [(784, 50), (50, 25), (25, 10)]
        # B = [(50,), (25,), (10,)]

        """
        ∂Loss/∂Z-1 = ∂Loss/∂Softmax * ∂Softmax/∂Z-1
                   = (Softmax - Y) * Softmax * (1 - Softmax) where Softmax = y_pred
        ∂Loss/∂W-1 = ∂Loss/∂Z-1 * ∂Z-1/∂W-1
                   = ∂Loss/∂Z-1 * Z-2
        ∂Loss/B-1  = ∂Loss/∂Z-1 * ∂Z-1/∂B-1
                   = ∂Loss/∂Z-1 * 1 (cos the bias is just added)
        """

        # Output layer
        dZ[-1] = (y_pred - y_true) * y_pred * (1 - y_pred)  # (64, 10)^3 => (64, 10)
        dW[-1] = (dZ[-1].T @ Z[-2]).T  # ((64, 10).T @ (64, 25)).T => (25, 10)
        dB[-1] = np.sum(dZ[-1], axis=0)  # (64, 10) => (10,)

        # Hidden layers
        for i in range(-2, -len(self.layers), -1):  # -2, -3
            dZ[i] = (self.W[i + 1] @ dZ[i + 1].T).T * relu_derivative(Z[i])  # ((25, 10) @ (64, 10).T).T => (64, 25)
            dW[i] = (dZ[i].T @ Z[i - 1]).T  # ((64, 25).T @ (64, 50)).T => (50, 25)
            dB[i] = np.sum(dZ[i], axis=0)  # (64, 25) => (25,)

        return dW, dB

    def update_params(self, learning_rate, dW, dB):
        for i in range(len(self.layers) - 1):  # 0, 1, 2
            self.W[i] -= learning_rate * dW[i]
            self.B[i] -= learning_rate * dB[i]

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
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.sum(y_pred == y) / len(y)
    print(data, "Accuracy:", accuracy)


def plot_loss(loss_history):
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
