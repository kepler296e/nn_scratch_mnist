import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import time


def main():
    # Training data
    train = pd.read_csv("mnist_test.csv", header=None)[:700]
    X_train, y_train = train.iloc[:, 1:] / 255, train.iloc[:, 0]

    # Test data
    test = pd.read_csv("mnist_test.csv", header=None)[700:1000]
    X_test, y_test = test.iloc[:, 1:] / 255, test.iloc[:, 0]

    # Build model
    model = Sequential(
        [
            tf.keras.Input(shape=(784,)),
            Dense(units=75, activation="relu"),
            Dense(units=25, activation="relu"),
            Dense(units=15, activation="relu"),
            Dense(units=10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # Train model
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
    )
    plot_loss(history)
    print("Time:", time.time() - start)

    # Test model
    y_pred = tf.nn.softmax(model.predict(X_test))
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", np.mean(y_pred == y_test))

    model.save("model.keras")


def plot_loss(history):
    plt.plot(history.history["loss"])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()
