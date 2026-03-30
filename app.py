"""
Gradio UI for Nishant's Neural Net (N3) — MNIST digit recognition.

Loads a pre-trained model from disk when available (fast startup) or
trains the same architecture as train.py for 5 epochs and saves it for
next time.  Users can draw a digit (0-9) on a canvas or click a test
sample and see:
  * the model's prediction & confidence,
  * a signal-flow visualisation using *actual learned weights*,
  * a gradient-descent analysis (saliency map + loss landscape).
"""

import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import tensorflow as tf
import gradio as gr


# -- Model -----------------------------------------------------------------

def load_or_train_model():
    """Load a saved model from disk, or train a new one and save it."""
    model_path = "mnist_model.keras"

    if os.path.exists(model_path):
        print(f"Found saved model at '{model_path}' -- loading it ...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded.\n")
        return model

    print("No saved model found. Training from scratch (5 epochs) ...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ])

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(x_train, y_train, epochs=5, verbose=2)
    model.save(model_path)
    print(f"\nModel saved to '{model_path}'.\n")
    return model


model = load_or_train_model()

# Activation model -- exposes every layer's output for the signal-flow viz.
activation_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[layer.output for layer in model.layers],
)

# Extract learned weight matrices for the signal-flow diagram.
_dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
W1, _b1 = _dense_layers[0].get_weights()   # (784, 128)
W2, _b2 = _dense_layers[1].get_weights()   # (128, 10)


# -- Sample MNIST test images ----------------------------------------------

def _prepare_samples():
    """Pick a diverse selection of MNIST test images (2 per digit)."""
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    sample_dir = os.path.join(tempfile.gettempdir(), "n3_samples")
    os.makedirs(sample_dir, exist_ok=True)

    gallery_items = []
    sample_rgb_arrays = []

    for digit in range(10):
        indices = np.where(y_test == digit)[0]
        for pick in [0, len(indices) // 3]:
            idx = indices[pick]
            img_28 = x_test[idx]

            pil_img = PILImage.fromarray(img_28, mode="L")
            pil_big = pil_img.resize((112, 112), PILImage.Resampling.NEAREST)
            path = os.path.join(sample_dir, f"sample_{digit}_{pick}.png")
            pil_big.save(path)
            gallery_items.append((path, f"Digit {digit}"))

            rgb = np.stack([img_28] * 3, axis=-1)
            sample_rgb_arrays.append(rgb)

    return gallery_items, sample_rgb_arrays


_gallery_items, _sample_rgb = _prepare_samples()


# -- Visualisation helpers --------------------------------------------------

CLASS_NAMES = [str(i) for i in range(10)]


def _draw_network_signal(activations, probabilities):
    """Signal-flow diagram using actual weight x activation products.

    Deep-learning theory recap:
      Forward pass through neuron j in a Dense layer:
        pre_j = sum_i  W[i,j] * a_i  +  b_j
        a_j   = ReLU(pre_j)
      The contribution of source neuron i to target j is W[i,j] * a_i.
      Positive contributions are excitatory (push up); negative are
      inhibitory (push down).  The winner pathway is highlighted in amber.
    """
    bg = "#0f172a"
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    # Activations by layer type
    flatten_acts = None
    dense_acts = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Flatten):
            flatten_acts = activations[i].flatten()
        elif isinstance(layer, tf.keras.layers.Dense) and dense_acts is None:
            dense_acts = activations[i].flatten()

    winner = int(np.argmax(probabilities))

    # Sample neurons for display
    n_flat, n_dense, n_out = 20, 16, 10
    idx_flat  = np.linspace(0, len(flatten_acts) - 1, n_flat,  dtype=int)
    idx_dense = np.linspace(0, len(dense_acts) - 1,   n_dense, dtype=int)

    flat_sample  = flatten_acts[idx_flat]
    dense_sample = dense_acts[idx_dense]

    # Actual weight contributions
    W1_sub = W1[np.ix_(idx_flat, idx_dense)]          # (20, 16)
    W2_sub = W2[idx_dense, :]                          # (16, 10)

    contrib1 = flat_sample[:, None] * W1_sub           # (20, 16)
    contrib2 = dense_sample[:, None] * W2_sub          # (16, 10)

    c1_abs = np.abs(contrib1)
    c2_abs = np.abs(contrib2)
    c1_max = c1_abs.max() + 1e-8
    c2_max = c2_abs.max() + 1e-8

    # Positions
    xs = [0.12, 0.48, 0.84]

    def _ys(n):
        return np.linspace(0.08, 0.88, n)

    ys_flat  = _ys(n_flat)
    ys_dense = _ys(n_dense)
    ys_out   = _ys(n_out)

    # Connections: input -> hidden (weight x activation)
    for i in range(n_flat):
        for j in range(n_dense):
            strength = c1_abs[i, j] / c1_max
            if strength < 0.05:
                continue
            color = "#38bdf8" if contrib1[i, j] >= 0 else "#f87171"
            ax.plot(
                [xs[0], xs[1]], [ys_flat[i], ys_dense[j]],
                color=color,
                linewidth=0.3 + strength * 1.5,
                alpha=float(np.clip(strength * 0.55, 0, 1)),
                zorder=1,
            )

    # Connections: hidden -> output (highlight winner pathway)
    for j in range(n_dense):
        for k in range(n_out):
            strength = c2_abs[j, k] / c2_max
            is_winner = (k == winner)

            if not is_winner and strength < 0.08:
                continue

            if is_winner:
                color = "#fbbf24" if contrib2[j, k] >= 0 else "#fb923c"
                alpha = float(np.clip(strength * 0.85 + 0.12, 0, 1))
                lw = 0.6 + strength * 3.5
            else:
                color = "#38bdf8" if contrib2[j, k] >= 0 else "#f87171"
                alpha = float(np.clip(strength * 0.25, 0, 1))
                lw = 0.2 + strength * 0.8

            ax.plot(
                [xs[1], xs[2]], [ys_dense[j], ys_out[k]],
                color=color, linewidth=lw, alpha=alpha,
                zorder=2 if is_winner else 1,
            )

    # Neuron circles
    r = 0.018

    flat_norm = flat_sample / (flat_sample.max() + 1e-8)
    for y, v in zip(ys_flat, flat_norm):
        ax.add_patch(plt.Circle(
            (xs[0], y), r,
            color=plt.cm.Blues(float(v) * 0.8 + 0.2),
            ec="#475569", linewidth=0.3, zorder=3))

    dense_norm = dense_sample / (dense_sample.max() + 1e-8)
    for y, v in zip(ys_dense, dense_norm):
        ax.add_patch(plt.Circle(
            (xs[1], y), r,
            color=plt.cm.cool(float(v)),
            ec="#475569", linewidth=0.3, zorder=3))

    # Output neurons -- winner gets a bright glow
    for k, (y, prob) in enumerate(zip(ys_out, probabilities)):
        if k == winner:
            ax.add_patch(plt.Circle((xs[2], y), r * 2.8,
                                    color="#fbbf24", alpha=0.15, zorder=2))
            ax.add_patch(plt.Circle((xs[2], y), r * 2.0,
                                    color="#fbbf24", alpha=0.30, zorder=2))
            ax.add_patch(plt.Circle((xs[2], y), r * 1.3,
                                    color="#fbbf24", ec="#fbbf24",
                                    linewidth=1.5, zorder=4))
        else:
            brightness = float(prob) * 0.5 + 0.15
            ax.add_patch(plt.Circle(
                (xs[2], y), r,
                color=plt.cm.Greys(brightness),
                ec="#475569", linewidth=0.3, zorder=3))

    # Labels
    for x, label in zip(
        xs,
        ["Input\n(784->20 shown)", "Hidden\n(128->16 shown)", "Output\n(10 classes)"],
    ):
        ax.text(x, 0.96, label, ha="center", va="top",
                fontsize=7, color="#94a3b8", fontweight="bold")

    for k, (y, name) in enumerate(zip(ys_out, CLASS_NAMES)):
        is_w = (k == winner)
        ax.text(
            xs[2] + 0.05, y,
            f"{name}  ({probabilities[k]*100:.1f}%)",
            ha="left", va="center", fontsize=7,
            color="#fbbf24" if is_w else "#64748b",
            fontweight="bold" if is_w else "normal",
        )

    # Legend
    ax.plot([], [], color="#38bdf8", lw=2, label="Excitatory (+)")
    ax.plot([], [], color="#f87171", lw=2, label="Inhibitory (-)")
    ax.plot([], [], color="#fbbf24", lw=3, label=f"Winner path -> {winner}")
    ax.legend(loc="lower center", fontsize=6, ncol=3,
              facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#cbd5e1", framealpha=0.9)

    fig.tight_layout(pad=0.3)
    return fig


def _draw_gradient_analysis(arr_batch, predicted_class):
    """Gradient-descent analysis for this specific input.

    Saliency map -- |dL/dx|.  Pixels with large gradient magnitude sit
    on a steep slope of the loss surface.

    Loss landscape slice -- perturb the input along the gradient direction:
    x' = x + eps * (grad / ||grad||).  Positive eps goes uphill (loss
    increases); negative eps goes downhill (gradient descent direction).
    """
    x = tf.constant(arr_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x)
        logits = model(x, training=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant([predicted_class]), logits=logits,
        )

    grad = tape.gradient(loss, x)

    saliency = tf.abs(grad).numpy().squeeze()

    grad_flat = tf.reshape(grad, [-1]).numpy()
    grad_norm_val = np.linalg.norm(grad_flat) + 1e-8
    grad_dir = grad_flat / grad_norm_val

    x_flat = arr_batch.flatten()
    current_loss = float(loss.numpy().item())

    epsilons = np.linspace(-2.0, 2.0, 35)
    losses = []
    for eps in epsilons:
        x_pert = np.clip(x_flat + eps * grad_dir, 0, 1).reshape(1, 28, 28)
        logits_p = model(tf.constant(x_pert, dtype=tf.float32), training=False)
        loss_p = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant([predicted_class]), logits=logits_p,
        )
        losses.append(float(loss_p.numpy().item()))

    bg  = "#0f172a"
    txt = "#cbd5e1"
    sub = "#94a3b8"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 4), facecolor=bg,
        gridspec_kw={"width_ratios": [1, 1.4]},
    )

    # Left -- saliency heatmap
    ax1.set_facecolor(bg)
    im = ax1.imshow(saliency, cmap="inferno", interpolation="bilinear")
    ax1.set_title("Saliency Map  |dL/dpixel|", color=txt, fontsize=9, pad=8)
    ax1.set_xlabel("Brighter = pixel matters more", color=sub, fontsize=7)
    ax1.tick_params(colors=sub, labelsize=5)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=sub, labelsize=5)

    # Right -- loss landscape slice
    ax2.set_facecolor(bg)
    ax2.plot(epsilons, losses, color="#2dd4bf", linewidth=2.5, zorder=2)
    ax2.fill_between(epsilons, losses, alpha=0.12, color="#2dd4bf")
    ax2.scatter([0], [current_loss], color="#fbbf24", s=120, zorder=5,
                edgecolors="white", linewidths=1.5)

    loss_range = max(losses) - min(losses) + 1e-8
    ax2.annotate(
        f"Current loss = {current_loss:.3f}",
        xy=(0, current_loss),
        xytext=(0.5, current_loss + loss_range * 0.25),
        fontsize=7, color="#fbbf24",
        arrowprops=dict(arrowstyle="->", color="#fbbf24", lw=1.2),
    )

    # Arrow showing gradient-descent direction (toward -eps, lower loss)
    descent_eps = -0.6
    descent_idx = int(np.argmin(np.abs(np.array(epsilons) - descent_eps)))
    descent_loss = losses[descent_idx]
    ax2.annotate(
        "", xy=(descent_eps, descent_loss), xytext=(0, current_loss),
        arrowprops=dict(arrowstyle="-|>", color="#f87171", lw=2.2,
                        mutation_scale=15),
        zorder=4,
    )
    ax2.text(descent_eps - 0.15, descent_loss, "grad descent",
             fontsize=7, color="#f87171", style="italic", ha="right")

    ax2.set_title("Loss Landscape (along grad direction)", color=txt,
                  fontsize=9, pad=8)
    ax2.set_xlabel("eps (perturbation magnitude)", color=sub, fontsize=8)
    ax2.set_ylabel("Cross-Entropy Loss", color=sub, fontsize=8)
    ax2.tick_params(colors=sub, labelsize=7)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#334155")

    fig.suptitle(
        f"Gradient Analysis -- predicted class '{predicted_class}'",
        color=txt, fontsize=10, y=0.02, va="bottom",
    )
    fig.tight_layout(pad=0.5, rect=[0, 0.05, 1, 1])
    return fig


# -- Image extraction helper -----------------------------------------------

def _extract_image(value):
    """Robustly extract a numpy array from Sketchpad / Image / dict."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, dict):
        composite = value.get("composite")
        if composite is not None:
            if isinstance(composite, np.ndarray):
                return composite
            if isinstance(composite, PILImage.Image):
                return np.array(composite)
            if isinstance(composite, str) and os.path.exists(composite):
                return np.array(PILImage.open(composite))
        layers = value.get("layers", [])
        for layer_data in reversed(layers):
            if isinstance(layer_data, np.ndarray):
                return layer_data
            if isinstance(layer_data, dict):
                d = layer_data.get("data")
                if isinstance(d, np.ndarray):
                    return d
    if isinstance(value, str) and os.path.exists(value):
        return np.array(PILImage.open(value))
    return None


# -- Core prediction -------------------------------------------------------

def _predict_core(image):
    """Core prediction.  Accepts any numpy image; returns 4 outputs."""
    if image is None:
        e1 = plt.figure(facecolor="#0f172a"); plt.close(e1)
        e2 = plt.figure(facecolor="#0f172a"); plt.close(e2)
        return "Draw or select an image", {c: 0.0 for c in CLASS_NAMES}, e1, e2

    img = PILImage.fromarray(image.astype("uint8"))
    img = img.convert("L")
    img = img.resize((28, 28), PILImage.Resampling.LANCZOS)

    arr = np.array(img, dtype="float32") / 255.0

    if arr.mean() > 0.5:
        arr = 1.0 - arr

    if arr.max() < 0.08:
        e1 = plt.figure(facecolor="#0f172a"); plt.close(e1)
        e2 = plt.figure(facecolor="#0f172a"); plt.close(e2)
        return "No digit detected", {c: 0.0 for c in CLASS_NAMES}, e1, e2

    arr_batch = arr.reshape(1, 28, 28)

    all_acts = activation_model.predict(arr_batch, verbose=0)
    logits = all_acts[-1]
    probabilities = tf.nn.softmax(logits[0]).numpy()

    predicted_class = int(np.argmax(probabilities))
    predicted_label = CLASS_NAMES[predicted_class]
    confidences = {c: float(probabilities[i]) for i, c in enumerate(CLASS_NAMES)}

    signal_fig   = _draw_network_signal(all_acts, probabilities)
    gradient_fig = _draw_gradient_analysis(arr_batch, predicted_class)

    return predicted_label, confidences, signal_fig, gradient_fig


def predict_from_sketch(sketch_value):
    """Handle Sketchpad input."""
    image = _extract_image(sketch_value)
    return _predict_core(image)


def predict_from_sample(evt: gr.SelectData):
    """Handle gallery sample click -- instantly predict."""
    idx = evt.index
    if 0 <= idx < len(_sample_rgb):
        return _predict_core(_sample_rgb[idx])
    return _predict_core(None)


# -- UI --------------------------------------------------------------------

with gr.Blocks(title="Nishant's Neural Net (N3) -- MNIST") as demo:
    gr.Markdown(
        "## Nishant's Neural Net (N3) -- MNIST Digit Recogniser\n"
        "**Draw** a digit (0-9) on the canvas with your cursor, "
        "**or** click any test sample below."
    )

    with gr.Row():
        with gr.Column(scale=2):
            sketchpad = gr.Sketchpad(
                label="Draw a digit with your cursor",
                brush=gr.Brush(
                    default_size=20,
                    colors=["#000000"],
                    default_color="#000000",
                    color_mode="fixed",
                ),
                canvas_size=(280, 280),
                image_mode="RGB",
                type="numpy",
            )
            predict_btn = gr.Button("Predict", variant="primary", size="lg")

        with gr.Column(scale=1):
            predicted_label  = gr.Label(label="Predicted digit")
            confidence_chart = gr.Label(
                label="Class confidences", num_top_classes=10,
            )

    gr.Markdown("### Test Samples -- click any digit to instantly predict")
    sample_gallery = gr.Gallery(
        value=_gallery_items,
        label="MNIST Test Samples (2 per digit, mixed difficulty)",
        columns=10,
        rows=2,
        height="auto",
    )

    with gr.Row():
        signal_plot   = gr.Plot(
            label="Neural Network Signal Flow  (weight x activation)",
        )
        gradient_plot = gr.Plot(
            label="Gradient Descent Analysis",
        )

    _outputs = [predicted_label, confidence_chart, signal_plot, gradient_plot]

    predict_btn.click(
        fn=predict_from_sketch, inputs=sketchpad, outputs=_outputs,
    )
    sample_gallery.select(
        fn=predict_from_sample, inputs=None, outputs=_outputs,
    )

    with gr.Accordion("How does this work?", open=False):
        gr.Markdown(
            "### Signal-Flow Diagram\n"
            "Each connection line represents **W[i,j] x activation_i** -- "
            "the *actual learned weight* multiplied by the neuron's current "
            "activation.\n\n"
            "| Colour | Meaning |\n"
            "|--------|---------|\n"
            "| **Cyan** | Excitatory (+) -- pushes the output *up* |\n"
            "| **Red** | Inhibitory (-) -- pushes the output *down* |\n"
            "| **Amber (thick)** | Winner pathway -- connections feeding "
            "the predicted digit |\n\n"
            "Thicker, brighter lines = larger |W*a| = stronger signal.  "
            "The winning output neuron gets an amber glow so you can "
            "instantly see *which* digit won and *why*.\n\n"
            "### Gradient Descent Analysis\n"
            "- **Saliency Map** -- |dLoss/dpixel| for every input pixel.  "
            "Bright spots are where the loss surface is steepest: tiny "
            "changes there would most alter the prediction.  Each input "
            "creates a unique saliency pattern.\n"
            "- **Loss Landscape** -- a 1-D slice of the loss surface along "
            "the gradient direction.  The *yellow dot* is where the model "
            "currently sits; the *red arrow* shows which way gradient "
            "descent would step to reduce the loss.  Different inputs "
            "produce different landscape shapes because the gradient "
            "direction changes with the input.\n\n"
            "| Term | Meaning |\n"
            "|------|---------|\n"
            "| **Neuron** | A unit that sums weighted inputs and applies "
            "an activation function. |\n"
            "| **Weight (W)** | A learned scalar that scales a connection "
            "between two neurons. |\n"
            "| **ReLU** | max(0, x) -- passes positive signals, blocks "
            "negatives. |\n"
            "| **Softmax** | Converts raw scores to probabilities summing "
            "to 100%. |\n"
            "| **Gradient** | Direction of steepest increase in loss; "
            "descent goes opposite. |\n"
            "| **Saliency** | Input-pixel sensitivity -- which pixels the "
            "model looks at most. |\n"
        )

    gr.Markdown(
        "---\n"
        "*Model: Dense(784->128->10) trained on MNIST for 5 epochs "
        "(same architecture as train.py).*"
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
    )
