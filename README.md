# Skin Cancer Classification using ML

## Project Overview

This project implements a machine learning approach to classify skin lesion images into different diagnostic categories. It utilizes both a custom CNN architecture and a pre-trained model (ResNet18) to compare their performance on skin lesion classification tasks.

## Dataset

This project uses the HAM10000 dataset ("Human Against Machine with 10000 training images") which consists of dermatoscopic images of common pigmented skin lesions.

- **Source**: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data)
- **Classes**: 7 different diagnostic categories:
  - Actinic keratoses (akiec)
  - Basal cell carcinoma (bcc)
  - Benign keratosis-like lesions (bkl)
  - Dermatofibroma (df)
  - Melanoma (mel)
  - Melanocytic nevi (nv)
  - Vascular lesions (vasc)

## Features

- Data preprocessing with balanced class sampling for handling imbalanced data
- Custom CNN architecture with configurable parameters
- Pre-trained model implementation using ResNet18
- Model comparison and performance visualization
- MLflow integration for experiment tracking
- Model export to ONNX or PyTorch format for deployment
- YAML-based configuration for easy parameter adjustments

## Setup and Installation

1. Clone the repository:
   ```
   git clone git@github.com:saurluca/skin-cancer-classification.git
   cd ml-skin-cancer
   ```

2. Install the required dependencies using uv:
[uv docs](https://docs.astral.sh/uv/getting-started/installation/)
   ```
   uv sync
   ```


3. Download the HAM10000 dataset from Kaggle and place it in the following structure:
   ```
   data/
   ├── HAM10000_metadata.csv
   └── images/
       ├── [image-id].jpg
       └── ...
   ```

## Usage

Run the main script to train and evaluate the models:

```
uv run main.py
```

The script will:
1. Load and preprocess the image data
2. Train both a custom CNN and a pre-trained model
3. Compare their performance
4. Save the models in either PyTorch or ONNX format (configurable)
5. Log experiments to MLflow

## Configuration

Model and training parameters can be configured in `config.yaml`. Key parameters include:
- Batch size
- Image dimensions
- Network architecture (number of layers, features)
- Learning rate and other optimization parameters
- Dataset paths
- Model save format (pth or onnx)

Example configuration:
```yaml
# Model parameters
conv_channels: [32, 64, 128, 256]
fc_features: [512, 256, 128]

# Training parameters
learning_rate: 5.0e-4
epochs: 10

# Model saving format (options: "pth" or "onnx")
model_save_format: "onnx"
```

## Results

The training process produces:
- Trained model weights saved in the `models/` directory
- Visualization of training/testing loss and accuracy
- Model exports for deployment (PyTorch or ONNX format)
- MLflow experiment logs for tracking performance

## Project Structure

- `main.py` - Main script to run the training and evaluation pipeline
- `config.yaml` - Configuration parameters in YAML format
- `config_loader.py` - Utility for loading YAML configuration
- `models.py` - Model architecture definitions and setup
- `preprocessing.py` - Data loading, preprocessing, and dataset management functions
- `training.py` - Training and evaluation functions
- `utils.py` - General utility functions
- `visualization.py` - Functions for plotting and visualizing results
- `experiment.py` - MLflow experiment tracking and model export

## Roadmap

- [ ] Implement k-fold crossvalidation