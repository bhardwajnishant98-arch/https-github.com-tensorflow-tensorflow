# TensorFlow Training

A simple TensorFlow training example that trains a neural network on the MNIST dataset.

This project is **Nishant's Neural Net (N3)**.

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

> Note: You can choose **any number** as input (where supported by the training script / configuration).

## GitHub Actions Workflow

The repository includes a GitHub Actions workflow (`.github/workflows/tensorflow_training.yml`) that automatically:

1. Sets up Python 3.10
2. Installs TensorFlow
3. Runs the training script (`train.py`)

The workflow is triggered on:
- Pushes to `main` or `copilot/**` branches
- Pull requests targeting `main`
- Manual dispatch via the **Actions** tab (`workflow_dispatch`)
