import numpy as np
import pandas as pd
import tensorflow as tf
import time


def main():
    # Load data
    data = pd.read_csv("mnist.csv", header=None).to_numpy()

    # The first column is the label and the rest are 28x28=784 pixels
    X = data[:, 1:] / 255  # divide by max=255 to normalize
    y = data[:, 0]

    # Split train and validation
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]

    # Build model
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input((784,)),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(25, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

    # Train
    start_time = time.time()
    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=128,
    )
    print("Time:", time.time() - start_time)

    # Evaluate
    evaluate(model, X_val, y_val)

    model.save("models/tf.keras")


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    # Accuracy
    print("Accuracy", np.mean(y_pred == y))

    # Confusion matrix
    # print("Confusion matrix")
    # print(pd.crosstab(y, y_pred, rownames=["True"], colnames=["Pred"]))

    # val loss
    print("Val loss", model.evaluate(X, y))


if __name__ == "__main__":
    main()
