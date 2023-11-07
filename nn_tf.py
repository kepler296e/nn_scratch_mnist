import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def main():
    # load data
    data = pd.read_csv("mnist.csv", header=None, nrows=10000).to_numpy()

    X = data[:, 1:] / 255  # normalize
    y = data[:, 0]

    # split data
    train = int(len(X) * 0.8)
    val = int(len(X) * 0.1)
    X_train, y_train = X[:train], y[:train]
    X_val, y_val = X[train : train + val], y[train : train + val]
    X_test, y_test = X[train + val :], y[train + val :]

    # build model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input((784,)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # train
    start = time.time()
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
    )
    print("Time:", time.time() - start)

    # evaluate
    evaluate(model, X_train, y_train, "Train")
    evaluate(model, X_val, y_val, "Validation")

    # model.save("models/model.keras")


def evaluate(model, X, y, data):
    y_pred = tf.nn.softmax(model.predict(X))
    y_pred = np.argmax(y_pred, axis=1)
    print(data, "accuracy:", np.mean(y_pred == y))


if __name__ == "__main__":
    main()
