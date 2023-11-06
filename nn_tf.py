import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def main():
    # load data
    data = pd.read_csv("mnist.csv", header=None, nrows=5000).to_numpy()

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
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(15, activation="relu"),
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
        batch_size=4,
    )
    print("Time:", time.time() - start)

    # evaluate
    y_pred = tf.nn.softmax(model.predict(X_val))
    y_pred = np.argmax(y_pred, axis=1)
    print("Accuracy:", np.mean(y_pred == y_val))

    # model.save("models/model.keras")


if __name__ == "__main__":
    main()
