# TensorFlow Training (This project is **Nishant's Neural Net (N3)**.)

A simple TensorFlow training example that trains a neural network on the MNIST dataset.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Run Training Locally

```bash
python train.py
```

### Run the Inference Visualisation UI

Launch the Gradio web UI to draw or upload a digit and see the model's
prediction along with per-class confidence scores:

```bash
python app.py
```

The app will train the model on startup and then open a local web page
(by default at `http://127.0.0.1:7860`) where you can:

1. **Draw** a digit (0-9) on the canvas or **upload** an image.
2. Click **Predict** to see the predicted label and a confidence bar chart.

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow (`.github/workflows/tensorflow_training.yml`) that automatically:

1. Sets up Python 3.10
2. Installs TensorFlow
3. Runs the training script (`train.py`)

The workflow is triggered on:
- Pushes to `main` or `copilot/**` branches
- Pull requests targeting `main`
- Manual dispatch via the **Actions** tab (`workflow_dispatch`)
