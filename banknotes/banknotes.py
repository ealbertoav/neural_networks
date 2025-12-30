import csv

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Read data in from a file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

# Convert to NumPy arrays for Keras compatibility
X_training = np.array(X_training)
X_testing = np.array(X_testing)
y_training = np.array(y_training)
y_testing = np.array(y_testing)

# Create a neural network
model = tf.keras.models.Sequential([
    # Define input shape using Input layer
    tf.keras.layers.Input(shape=(4,)),
    # Add a hidden layer with 8 units, with ReLU activation
    tf.keras.layers.Dense(8, activation="relu"),
    # Add an output layer with 1 unit, with sigmoid activation
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.fit(X_training, y_training, epochs=20)

# Evaluate how well the model performs
model.evaluate(X_testing, y_testing, verbose=2)
