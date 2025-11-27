# CIFAR-10 VGG Image Classification with Adversarial Attacks

A PyTorch-based image classification project that trains VGG-style CNN models on CIFAR-10 dataset and evaluates their robustness against FGSM adversarial attacks.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements three VGG-style CNN architectures (shallow, original, and deep) to classify images from the CIFAR-10 dataset. It evaluates model performance on clean test data and assesses robustness against FGSM (Fast Gradient Sign Method) adversarial attacks.

The workflow is organized into four independent steps:
1. **Data Preparation** - Extract and prepare CIFAR-10 subset
2. **Training** - Train VGG models with different configurations
3. **Testing** - Evaluate trained models on test set
4. **Adversarial Attack** - Test model robustness with FGSM

## Features

- 🧠 Three VGG variants: Shallow (5 conv), Original (8 conv), Deep (13 conv)
- 🎯 Configurable kernel sizes (3x3, 5x5, 7x7, etc.)
- 📊 Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- 🎨 Confusion matrix visualizations
- ⚔️ FGSM adversarial attack evaluation
- 📈 Adversarial robustness curves
- 🖼️ Adversarial example visualizations
- 🔧 Modular design with independent scripts

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- numpy
- scikit-learn
- matplotlib

## Platform Notes

**Mac/Linux users:** Use `python3` and `pip3` commands
**Windows users:** Use `python` and `pip` commands

Throughout this guide:
- 🍎 Mac/Linux commands are shown first
- 🪟 Windows commands are shown below (when different)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Soen-proj
```

### 2. Create a virtual environment (recommended)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

**Mac/Linux:**
```bash
pip3 install torch torchvision numpy scikit-learn matplotlib
```

**Windows:**
```cmd
pip install torch torchvision numpy scikit-learn matplotlib
```

Or use requirements.txt:

**Mac/Linux:**
```bash
pip3 install -r requirements.txt
```

**Windows:**
```cmd
pip install -r requirements.txt
```

## Quick Start

Run the complete workflow with default settings:

**Mac/Linux:**
```bash
# Step 0: Prepare dataset (first time only)
python3 step0_project_setup.py

# Step 1: Train a model
python3 step1_train.py --depth original --kernel 3 --epochs 10

# Step 2: Test the model
python3 step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Step 3: Attack the model
python3 step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
```

**Windows:**
```cmd
# Step 0: Prepare dataset (first time only)
python step0_project_setup.py

# Step 1: Train a model
python step1_train.py --depth original --kernel 3 --epochs 10

# Step 2: Test the model
python step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Step 3: Attack the model
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
```

## Detailed Usage

### Step 0: Data Preparation

**Purpose**: Download CIFAR-10 and extract a subset for training and testing.

**Mac/Linux:**
```bash
python3 step0_project_setup.py
```

**Windows:**
```cmd
python step0_project_setup.py
```

**What it does**:
- Downloads CIFAR-10 dataset (170 MB) to `./data/`
- Selects 500 images per class for training (5,000 total)
- Selects 100 images per class for testing (1,000 total)
- Exports images as PNG files to `training_images/` and `test_images/`

**Note**: Only needs to be run once. Subsequent runs will reuse existing data.

---

### Step 1: Training

**Purpose**: Train a VGG model and save the best checkpoint.

**Mac/Linux:**
```bash
python3 step1_train.py [OPTIONS]
```

**Windows:**
```cmd
python step1_train.py [OPTIONS]
```

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--depth` | `original` | Architecture: `shallow`, `original`, or `deep` |
| `--kernel` | `3` | Convolution kernel size (3, 5, 7, etc.) |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `0.01` | Learning rate |
| `--batch` | `64` | Batch size |

**Examples:**

**Mac/Linux:**
```bash
# Train original VGG with kernel size 3 (default 10 epochs)
python3 step1_train.py --depth original --kernel 3

# Train shallow model with larger kernels
python3 step1_train.py --depth shallow --kernel 5

# Train deep architecture with more epochs for better convergence
python3 step1_train.py --depth deep --kernel 3 --epochs 20
```

**Windows:**
```cmd
# Train original VGG with kernel size 3 (default 10 epochs)
python step1_train.py --depth original --kernel 3

# Train shallow model with larger kernels
python step1_train.py --depth shallow --kernel 5

# Train deep architecture with more epochs for better convergence
python step1_train.py --depth deep --kernel 3 --epochs 20

# Train deep model with more epochs
python step1_train.py --depth deep --epochs 60 --lr 0.001
```

**Outputs**:
- `outputs/vgg_{depth}_k{kernel}.pt` - Model checkpoint
- `outputs/vgg_{depth}_k{kernel}_meta.txt` - Training metadata

---

### Step 2: Testing

**Purpose**: Evaluate a trained model on the test set.

**Mac/Linux:**
```bash
python3 step2_test.py --model <path-to-model> [OPTIONS]
```

**Windows:**
```cmd
python step2_test.py --model <path-to-model> [OPTIONS]
```

**Required**:
- `--model` - Path to trained model (e.g., `outputs/vgg_original_k3.pt`)

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--depth` | `original` | Must match training architecture |
| `--kernel` | `3` | Must match training kernel size |
| `--batch` | `64` | Batch size for evaluation |

**Examples:**

**Mac/Linux:**
```bash
# Test the trained original model
python3 step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Test a shallow model with kernel size 5
python3 step2_test.py --model outputs/vgg_shallow_k5.pt --depth shallow --kernel 5
```

**Windows:**
```cmd
# Test the trained original model
python step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Test a shallow model with kernel size 5
python step2_test.py --model outputs/vgg_shallow_k5.pt --depth shallow --kernel 5
```

**Outputs**:
- `outputs/classification_report_{depth}_k{kernel}.txt` - Detailed metrics
- `outputs/confusion_matrix_{depth}_k{kernel}.png` - Confusion matrix plot

---

### Step 3: Adversarial Attack

**Purpose**: Evaluate model robustness against FGSM attacks.

**Mac/Linux:**
```bash
python3 step3_attack.py --model <path-to-model> [OPTIONS]
```

**Windows:**
```cmd
python step3_attack.py --model <path-to-model> [OPTIONS]
```

**Required**:
- `--model` - Path to trained model

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--depth` | `original` | Must match training architecture |
| `--kernel` | `3` | Must match training kernel size |
| `--batch` | `64` | Batch size for evaluation |
| `--epsilons` | `0.0,0.01,0.03,0.05,0.1` | Comma-separated epsilon values |
| `--seed` | `None` | Random seed for reproducibility |
| `--shuffle` | `False` | Shuffle test data for different examples |

**Examples:**

**Mac/Linux:**
```bash
# Standard attack (random examples each run)
python3 step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Reproducible results with seed
python3 step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42

# Shuffle data to see different examples each run
python3 step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --shuffle

# Custom epsilon values
python3 step3_attack.py --model outputs/vgg_deep_k3.pt --depth deep --kernel 3 \
  --epsilons "0.0,0.005,0.01,0.02,0.05,0.1"
```

**Windows:**
```cmd
# Standard attack (random examples each run)
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3

# Reproducible results with seed
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --seed 42

# Shuffle data to see different examples each run
python step3_attack.py --model outputs/vgg_original_k3.pt --depth original --kernel 3 --shuffle

# Custom epsilon values (note: use ^ for line continuation in cmd)
python step3_attack.py --model outputs/vgg_deep_k3.pt --depth deep --kernel 3 --epsilons "0.0,0.005,0.01,0.02,0.05,0.1"
```

**Getting Different Results Each Run**:
- By default (no `--seed`), visualizations show random examples each run
- Use `--shuffle` to randomize the order of test data
- Use `--seed <number>` for reproducible results

**Outputs**:
- `outputs/attack_results_{depth}_k{kernel}.txt` - Attack summary
- `outputs/fgsm_curve_{depth}_k{kernel}.png` - Accuracy vs epsilon plot
- `outputs/attack_examples_{depth}_k{kernel}_eps*.png` - Visual examples

---

## Project Structure

```
Soen-proj/
├── README.md                  # This file
├── step0_project_setup.py     # Data preparation
├── step1_train.py            # Model training
├── step2_test.py             # Model testing
├── step3_attack.py           # Adversarial attacks
├── data/                     # CIFAR-10 dataset (auto-downloaded)
├── training_images/          # Training subset (5,000 images)
├── test_images/             # Test subset (1,000 images)
├── outputs/                 # Models and results
│   ├── *.pt                # Model checkpoints
│   ├── *.txt               # Reports and metadata
│   └── *.png               # Plots and visualizations
└── selected_indices.json    # Dataset subset metadata
```

## Model Architectures

### VGG_Shallow (5 convolutional layers)
- **Structure**: 3 → 64 → 128 → 256
- **Classifier**: 4096 → 2048 → 10
- **Best for**: Faster training, less parameters

### VGG11 - Original (8 convolutional layers)
- **Structure**: 3 → 64 → 128 → 256 → 512 → 512
- **Classifier**: 4096 → 4096 → 10
- **Best for**: Balanced performance

### VGG_Deep (13 convolutional layers)
- **Structure**: 3 → 64 → 128 → 256 → 512 → 512 (doubled layers)
- **Classifier**: 4096 → 4096 → 10
- **Best for**: Maximum capacity, potential overfitting

**All models include**:
- Batch Normalization
- ReLU activations
- Max Pooling
- Dropout (0.5) in classifier
- SGD optimizer with momentum (0.9)

## Results

After running the complete workflow, you'll get:

### Training Results
- Training progress printed to console
- Best model checkpoint saved
- Training metadata (accuracy, parameters, hyperparameters)

### Testing Results
- Overall test accuracy
- Per-class precision, recall, F1-score
- Confusion matrix visualization

### Attack Results
- Accuracy degradation across epsilon values
- Adversarial robustness curve
- Visual comparison of original vs adversarial images
- Perturbation visualizations

## Troubleshooting

### Issue: "No module named 'torch'"
**Solution**: Install PyTorch

**Mac/Linux:**
```bash
pip3 install torch torchvision
```

**Windows:**
```cmd
pip install torch torchvision
```

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size

**Mac/Linux:**
```bash
python3 step1_train.py --batch 32  # or even 16
```

**Windows:**
```cmd
python step1_train.py --batch 32
```

### Issue: "Model file not found"
**Solution**: Make sure you've run step1 first and use the correct path

**Mac/Linux:**
```bash
# Check what models exist
ls outputs/*.pt
# Use the correct path
python3 step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
```

**Windows:**
```cmd
# Check what models exist
dir outputs\*.pt
# Use the correct path
python step2_test.py --model outputs/vgg_original_k3.pt --depth original --kernel 3
```

### Issue: Low accuracy during training
**Possible solutions**:
- Increase epochs: `--epochs 60`
- Adjust learning rate: `--lr 0.001`
- Try different architecture: `--depth deep`

### Issue: Dataset download fails
**Solution**:
- Check internet connection
- Manually download CIFAR-10 from https://www.cs.toronto.edu/~kriz/cifar.html
- Extract to `./data/cifar-10-batches-py/`

## CIFAR-10 Classes

The model classifies images into 10 categories:
1. Airplane ✈️
2. Automobile 🚗
3. Bird 🐦
4. Cat 🐱
5. Deer 🦌
6. Dog 🐕
7. Frog 🐸
8. Horse 🐴
9. Ship 🚢
10. Truck 🚚

## Hardware Requirements

- **Minimum**: CPU, 4GB RAM
- **Recommended**: GPU (CUDA/MPS), 8GB+ RAM
- **Disk Space**: ~500MB (dataset + models + outputs)

## Performance Tips

- Use GPU if available (automatically detected)
- Start with shallow architecture for quick experiments
- Use larger batch sizes on GPUs (128 or 256)
- Monitor training - stop if loss plateaus early

## License

[Add your license here]

## Contact

[Add your contact information here]

## Acknowledgments

- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009
- VGG architecture: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), Simonyan & Zisserman, 2014
- FGSM attack: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572), Goodfellow et al., 2014
