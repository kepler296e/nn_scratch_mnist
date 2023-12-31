import numpy as np
import pandas as pd
import nn_data
import matplotlib.pyplot as plt
import time


def main():
    # Load data
    X_train, y_train, X_val, y_val = nn_data.load_data()

    # Build model
    model = NN(layers=[28 * 28, 50, 25, 10])
    print(model.count_params(), "parameters")

    # Train
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=10,
        eval_every=1,
        lr=0.01,
        batch_size=128,
    )

    model.evaluate(X_val, y_val)
    model.save("models/scratch.npy")


class NN:
    def __init__(self, layers):
        self.layers = layers
        self.W = [np.random.randn(i, j) * np.sqrt(2 / i) for i, j in zip(layers[:-1], layers[1:])]  # xavier init
        self.B = [np.zeros(i) for i in layers[1:]]

    def fit(self, X_train, y_train, X_val, y_val, epochs, eval_every, lr, batch_size):
        start_train_time = time.time()
        lossi = []

        for epoch in range(epochs):
            train_loss = self.train(X_train, y_train, batch_size, lr)

            if (epoch + 1) % eval_every == 0:
                val_loss = self.eval(X_val, y_val, batch_size)
                print(f"Epoch {epoch + 1}/{epochs}, Train loss {train_loss:.4f}, Val loss {val_loss:.4f}")

            lossi.append(train_loss)

        print("Train time", time.time() - start_train_time)
        # plot_loss(lossi)

    def train(self, X_train, y_train, batch_size, lr):
        train_loss = 0
        for i in range(0, len(X_train) - len(X_train) % batch_size, batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            # Forward pass
            Z, y_pred = self.forward(X_batch)

            # Loss
            y_true = np.zeros((batch_size, self.layers[-1]))
            y_true[np.arange(batch_size), y_batch] = 1  # label one hot encoded
            train_loss += cross_entropy(y_true, y_pred) / batch_size

            # Backprop
            dW, dB = self.backprop(Z, y_true, y_pred)

            # Update params
            self.update_params(lr, dW, dB)

        return train_loss / (len(X_train) / batch_size)

    def eval(self, X_val, y_val, batch_size):
        val_loss = 0
        for i in range(0, len(X_val) - len(X_val) % batch_size, batch_size):
            X_batch = X_val[i : i + batch_size]
            y_batch = y_val[i : i + batch_size]

            _, y_pred = self.forward(X_batch)

            y_true = np.zeros((batch_size, self.layers[-1]))
            y_true[np.arange(batch_size), y_batch] = 1
            val_loss += cross_entropy(y_true, y_pred) / batch_size

        return val_loss / (len(X_val) / batch_size)

    def forward(self, X):
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

    def update_params(self, lr, dW, dB):
        for i in range(len(self.layers) - 1):  # 0, 1, 2
            self.W[i] -= lr * dW[i]
            self.B[i] -= lr * dB[i]

    def count_params(self):
        return sum(np.prod(w.shape) for w in self.W) + sum(np.prod(b.shape) for b in self.B)

    def evaluate(self, X, y):
        _, y_pred = self.forward(X)
        y_pred = np.argmax(y_pred, axis=1)

        # Accuracy
        print("Accuracy", np.mean(y_pred == y))

        # Confusion matrix
        print(pd.crosstab(y, y_pred, rownames=["True"], colnames=["Pred"]))

    def save(self, filepath):
        model_data = {"layers": self.layers, "W": self.W, "B": self.B}
        np.save(filepath, model_data)


def relu(X):
    return np.where(X > 0, X, 0)


def relu_derivative(X):
    return np.where(X > 0, 1, 0)


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8))  # 1e-8 to avoid ln(0)


def softmax(logits):
    logits -= np.max(logits, axis=1, keepdims=True)  # (64, 10) - (64, 1)
    exp_logits = np.exp(logits)  # (64, 10)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (64, 10) / (64, 1)


def plot_loss(lossi):
    plt.plot(lossi)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def load_model(filepath):
    model_data = np.load(filepath, allow_pickle=True).item()
    model = NN(model_data["layers"])
    model.W = model_data["W"]
    model.B = model_data["B"]
    return model


if __name__ == "__main__":
    main()
