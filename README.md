# TensorFlow Training

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

This trains the model and saves it to `mnist_model.keras`.

### Handwritten Numbers Creator

After training the model, run the **Handwritten Numbers Creator** to generate
test images and interactively test the model:

```bash
python handwritten_numbers_creator.py
```

The script will:

1. Create a `test_images/` directory with 5 sample images for each digit (0–9)
   extracted from the MNIST test dataset.
2. Ask you which number you wish to input on the trained model.
3. Run the model on the corresponding test images and display predictions with
   confidence scores.

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow (`.github/workflows/tensorflow_training.yml`) that automatically:

1. Sets up Python 3.10
2. Installs TensorFlow
3. Runs the training script (`train.py`)

The workflow is triggered on:
- Pushes to `main` or `copilot/**` branches
- Pull requests targeting `main`
- Manual dispatch via the **Actions** tab (`workflow_dispatch`)
