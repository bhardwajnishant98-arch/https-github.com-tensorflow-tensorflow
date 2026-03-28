"""
Nishant's Neural Net (N3)
Trains a small neural network on the MNIST dataset.
"""

import tensorflow as tf

print("=" * 40)
print("  Nishant's Neural Net (N3)")
print("=" * 40)
print(f"TensorFlow version: {tf.__version__}")

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple sequential model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model for a single epoch
print("Training model...")
model.fit(x_train, y_train, epochs=1)

# Evaluate the model
print("\nEvaluating model...")
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test loss:     {loss:.4f}")
