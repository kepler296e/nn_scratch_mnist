import tensorflow as tf
import nn_data
import time

# Load data
X_train, y_train, X_val, y_val = nn_data.load_data()

# Build model
model = tf.keras.models.Sequential(
    [
        tf.keras.Input((28 * 28,)),
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
start_train_time = time.time()
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
)
print("Train time", time.time() - start_train_time)

model.save("models/tf.keras")
