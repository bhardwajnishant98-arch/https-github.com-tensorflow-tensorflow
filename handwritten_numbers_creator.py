"""
Handwritten Numbers Creator

Generates sets of test images from the MNIST handwritten digit dataset,
asks the user which number they wish to test, and runs the trained model
to predict the digit.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

TEST_IMAGES_DIR = "test_images"
MODEL_PATH = "mnist_model.keras"
SAMPLES_PER_DIGIT = 5


def create_test_images():
    """Extract sample images from the MNIST dataset and save them as PNGs."""
    print("Creating test images from the MNIST dataset...")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

    for digit in range(10):
        digit_dir = os.path.join(TEST_IMAGES_DIR, str(digit))
        os.makedirs(digit_dir, exist_ok=True)

        indices = np.where(y_test == digit)[0][:SAMPLES_PER_DIGIT]
        for i, idx in enumerate(indices):
            img_array = x_test[idx]
            img = Image.fromarray(img_array, mode="L")
            filepath = os.path.join(digit_dir, f"sample_{i}.png")
            img.save(filepath)

    total = SAMPLES_PER_DIGIT * 10
    print(f"Created {total} test images in '{TEST_IMAGES_DIR}/' directory.")
    print("Each digit (0-9) has its own folder with sample images.\n")


def load_and_preprocess_image(image_path):
    """Load a PNG image and preprocess it for the MNIST model."""
    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def predict_digit(model, image_path):
    """Run prediction on a single image and return results."""
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    probabilities = tf.nn.softmax(prediction[0]).numpy()
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])
    return predicted_digit, confidence


def main():
    """Main interactive loop for the Handwritten Numbers Creator."""
    print("=" * 50)
    print("   Handwritten Numbers Creator")
    print("=" * 50)
    print()

    # Step 1: Generate test images
    create_test_images()

    # Step 2: Load the trained model
    if not os.path.exists(MODEL_PATH):
        print(f"Trained model not found at '{MODEL_PATH}'.")
        print("Please run 'python train.py' first to train and save the model.")
        sys.exit(1)

    print(f"Loading trained model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!\n")

    # Step 3: Interactive prediction loop
    while True:
        number = input(
            "What number do you wish to input on the trained model? (0-9, or 'q' to quit): "
        ).strip()

        if number.lower() == "q":
            print("Goodbye!")
            break

        if not number.isdigit() or int(number) not in range(10):
            print("Please enter a single digit between 0 and 9.\n")
            continue

        digit = int(number)
        digit_dir = os.path.join(TEST_IMAGES_DIR, str(digit))

        if not os.path.isdir(digit_dir):
            print(f"Test images directory '{digit_dir}' not found.")
            print("Regenerating test images...\n")
            create_test_images()

        samples = sorted(
            f
            for f in os.listdir(digit_dir)
            if f.endswith(".png")
        )

        print(f"\nTesting {len(samples)} sample image(s) of digit '{digit}':\n")
        for sample in samples:
            image_path = os.path.join(digit_dir, sample)
            predicted_digit, confidence = predict_digit(model, image_path)
            status = "CORRECT" if predicted_digit == digit else "WRONG"
            print(
                f"  {sample}: predicted={predicted_digit}  "
                f"confidence={confidence:.2%}  [{status}]"
            )
        print()


if __name__ == "__main__":
    main()
