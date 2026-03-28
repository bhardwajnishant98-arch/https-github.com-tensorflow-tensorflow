"""
Gradio UI for visualising MNIST model inference.

Trains the same model used in train.py, then lets users draw a digit
on a canvas or upload an image and see the model's prediction along
with a confidence bar chart for all 10 digit classes.
"""

import numpy as np
from PIL import Image as PILImage
import tensorflow as tf
import gradio as gr


# ── Model ────────────────────────────────────────────────────────────────────

def build_and_train_model():
    """Build and train the MNIST model (mirrors train.py)."""
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=1, verbose=2)
    return model


print("Training model – this may take a moment …")
model = build_and_train_model()
print("Model ready.\n")


# ── Inference ────────────────────────────────────────────────────────────────

CLASS_NAMES = [str(i) for i in range(10)]


def predict(image):
    """Return the predicted label and per-class confidences."""
    if image is None:
        return "No image provided", {c: 0.0 for c in CLASS_NAMES}

    # Gradio Sketchpad/Image gives an RGB or RGBA numpy array.
    # Convert to grayscale, resize to 28×28, and normalise.
    img = PILImage.fromarray(image.astype("uint8"))
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28), PILImage.Resampling.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0

    # MNIST digits are white-on-black; if the canvas background is white
    # (mean pixel value > 0.5), invert so the digit becomes white-on-black.
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    arr = arr.reshape(1, 28, 28)

    logits = model.predict(arr, verbose=0)
    probabilities = tf.nn.softmax(logits[0]).numpy()

    predicted_label = CLASS_NAMES[int(np.argmax(probabilities))]
    confidences = {c: float(probabilities[i]) for i, c in enumerate(CLASS_NAMES)}
    return predicted_label, confidences


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MNIST Inference Visualiser") as demo:
    gr.Markdown(
        "## MNIST Inference Visualiser\n"
        "Draw a digit (0-9) on the canvas **or** upload an image, "
        "then click **Predict** to see the model's output."
    )

    with gr.Row():
        input_image = gr.Image(
            label="Draw or upload a digit",
            type="numpy",
            image_mode="RGB",
        )
        with gr.Column():
            predicted_label = gr.Label(label="Predicted digit")
            confidence_chart = gr.Label(label="Class confidences", num_top_classes=10)

    predict_btn = gr.Button("Predict", variant="primary")
    predict_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[predicted_label, confidence_chart],
    )

    gr.Markdown(
        "---\n"
        "*Model: simple dense network trained on MNIST for 1 epoch "
        "(same architecture as `train.py`).*"
    )

if __name__ == "__main__":
    demo.launch()
