import tensorflow as tf
import numpy as np

# Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
    ])

# Compile the model
model.compile(optimizer="sgd", loss="mean_squared_error")

# Define training data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction
print("500 training Prediction of y=2x-1. Test x = 10")
print(model.predict(np.array([10.0])))
