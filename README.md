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
- Model export to ONNX format for deployment

## Setup and Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd ml-skin-cancer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download the HAM10000 dataset from Kaggle and place it in the following structure:
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
python main.py
```

The script will:
1. Load and preprocess the image data
2. Train both a custom CNN and a pre-trained model
3. Compare their performance
4. Save the models in both PyTorch and ONNX formats
5. Log experiments to MLflow

## Configuration

Model and training parameters can be configured in `config.py`. Key parameters include:
- Batch size
- Image dimensions
- Network architecture (number of layers, features)
- Learning rate and other optimization parameters
- Dataset paths

## Results

The training process produces:
- Trained model weights saved in the `models/` directory
- Visualization of training/testing loss and accuracy
- ONNX model exports for deployment
- MLflow experiment logs for tracking performance

## Project Structure

- `main.py` - Main script to run the training and evaluation pipeline
- `config.py` - Configuration parameters
- `models.py` - Model architecture definitions
- `preprocessing.py` - Data loading and preprocessing functions
- `training.py` - Training and evaluation functions
- `utils.py` - Utility functions for model saving and MLflow integration
- `visualization.py` - Functions for plotting and visualizing results
