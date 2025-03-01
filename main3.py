import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch
import torch.onnx
import datetime
from pathlib import Path


@dataclass
class ModelConfig:
    # Data parameters
    batch_size: int = 32
    image_size: tuple = (64, 48)  # image is rectangular with og size 600x450
    train_split: float = 0.8

    # Model parameters
    conv_channels: list = (32, 64, 128)
    fc_features: list = (512, 256)
    dropout_rates_conv: list = (0.1, 0.2, 0.3)
    dropout_rates_fc: list = (0.5, 0.4)

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2

    # Paths
    model_save_path: str = "skin_lesion_model.pth"
    csv_path: str = "HAM10000_metadata.csv"
    base_path: str = "/home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/"
    folders: list = None

    # MLflow parameters
    experiment_name: str = "skin-lesion-classification"
    run_name: str = "cnn-model"

    def __post_init__(self):
        if self.folders is None:
            self.folders = [
                "ham10000_images_part_1",
                "ham10000_images_part_2",
                "HAM10000_images_part_1",
                "HAM10000_images_part_2",
            ]


# Initialize configuration
config = ModelConfig()

# Set up MLflow experiment
mlflow.set_experiment(config.experiment_name)

np.random.seed(42)
torch.manual_seed(42)

# Check if the CSV file exists
if not os.path.exists(config.csv_path):
    print(f"Error: CSV file not found at {config.csv_path}")
    exit(1)

df = pd.read_csv(config.csv_path)

# Create a mapping from diagnosis to numerical label
if "dx" in df.columns:
    unique_diagnoses = df["dx"].unique()
    diagnosis_to_idx = {
        diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)
    }
    idx_to_diagnosis = {idx: diagnosis for diagnosis, idx in diagnosis_to_idx.items()}
    print(f"Diagnosis classes: {diagnosis_to_idx}")


# Function to load images from multiple folders
def load_images_from_folders(image_ids, base_path, folders):
    images = []
    found_image_ids = []
    missing_image_ids = []

    for image_id in image_ids:
        found = False
        # Try each folder until the image is found
        for folder in folders:
            image_path = os.path.join(base_path, folder, f"{image_id}.jpg")
            if os.path.exists(image_path):
                image = Image.open(image_path)
                images.append(image)
                found_image_ids.append(image_id)
                found = True
                break

        if not found:
            missing_image_ids.append(image_id)

    # Print information about missing and found images
    print(f"Total images in CSV: {len(image_ids)}")
    print(f"Found: {len(found_image_ids)} images")
    print(f"Missing: {len(missing_image_ids)} images")

    if found_image_ids:
        print(f"First few found: {found_image_ids[:5]}")

    if missing_image_ids:
        print(f"First few missing: {missing_image_ids[:5]}")

    return images, found_image_ids


# Try to load the images from all folders
print(f"Attempting to load images from multiple folders in {config.base_path}")
images, found_image_ids = load_images_from_folders(
    df["image_id"].tolist(), config.base_path, config.folders
)

print(f"Successfully loaded {len(images)} images")

# Filter the dataframe to include only the images we found
filtered_df = df[df["image_id"].isin(found_image_ids)]
print(f"Filtered dataframe contains {len(filtered_df)} rows")


# Create a custom dataset class
class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define image transformations
transform = transforms.Compose(
    [
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Add data augmentation for minority classes
transform_minority = transforms.Compose(
    [
        transforms.Resize((config.image_size[0] + 4, config.image_size[1] + 4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create image paths and labels
image_paths = []
for img_id in filtered_df["image_id"]:
    # Try to find the image in each folder
    found = False
    for folder in config.folders:
        path = os.path.join(config.base_path, folder, f"{img_id}.jpg")
        if os.path.exists(path):
            image_paths.append(path)
            found = True
            break
    if not found:
        print(f"Warning: Could not find image {img_id} in any folder")

labels = [diagnosis_to_idx[diagnosis] for diagnosis in filtered_df["dx"]]

# Ensure we have the same number of paths and labels
if len(image_paths) != len(labels):
    print(
        f"Warning: Number of image paths ({len(image_paths)}) doesn't match number of labels ({len(labels)})"
    )
    # Keep only the labels for which we have images
    labels = labels[: len(image_paths)]


# Create a class-balanced sampler to handle imbalanced classes
def create_class_balanced_sampler(dataset):
    # Get all labels from the dataset
    all_labels = [dataset[i][1] for i in range(len(dataset))]

    # Count occurrences of each class
    class_counts = {}
    for label in all_labels:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

    # Calculate weights for each sample
    weights = [1.0 / class_counts[all_labels[i]] for i in range(len(all_labels))]
    weights = torch.DoubleTensor(weights)

    # Create a sampler that samples with replacement according to the weights
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, len(weights), replacement=True
    )

    return sampler


# Create the dataset
dataset = SkinLesionDataset(image_paths, labels, transform=transform)

# Split into train and test sets
train_size = int(config.train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

print("Creating balanced sampler for training set")
# Create a balanced sampler for the training set
train_sampler = create_class_balanced_sampler(train_dataset)

print("Creating data loaders")
# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=train_sampler,  # Use the balanced sampler instead of shuffle=True
)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)


# Enhanced CNN model with configurable architecture
class CNNModel(nn.Module):
    def __init__(self, config, num_classes=None):
        super().__init__()
        if num_classes is None:
            num_classes = len(diagnosis_to_idx)

        # Build convolutional layers dynamically
        conv_layers = []
        in_channels = 3  # RGB images

        for i, out_channels in enumerate(config.conv_channels):
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(config.dropout_rates_conv[i]),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of flattened features after convolutions
        # For 28x28 input with 3 MaxPool2d layers: 28 -> 14 -> 7 -> 3
        conv_output_size = (
            config.conv_channels[-1]
            * (config.image_size[0] // (2 ** len(config.conv_channels)))
            * (config.image_size[1] // (2 ** len(config.conv_channels)))
        )

        # Build fully connected layers dynamically
        fc_layers = []
        in_features = conv_output_size

        for i, out_features in enumerate(config.fc_features):
            fc_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rates_fc[i]),
                ]
            )
            in_features = out_features

        # Add final classification layer
        fc_layers.append(nn.Linear(config.fc_features[-1], num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device", "\n")

model = CNNModel(config).to(device)
print(model)

print("Calculating class weights")
# Calculate class weights based on class frequencies
class_counts = filtered_df["dx"].value_counts()
total_samples = len(filtered_df)
class_weights = torch.tensor(
    [
        (total_samples / class_counts[idx_to_diagnosis[i]])
        for i in range(len(diagnosis_to_idx))
    ],
    dtype=torch.float,
).to(device)

print("Creating loss function")
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

print("Creating optimizer")
optimizer = optim.Adam(
    model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config.scheduler_factor,
    patience=config.scheduler_patience,
)

print("Training function")


# Training function
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0

    # Create progress bar for training batches
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.epochs} [Train]",
        leave=False,
        unit="batch",
    )

    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Update progress bar with current loss and accuracy
        batch_loss = loss.item()
        batch_acc = (pred.argmax(1) == y).type(torch.float).mean().item()
        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}", accuracy=f"{100 * batch_acc:.2f}%"
        )

    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / size
    print(
        f"Epoch {epoch + 1}: Avg loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_acc:.2f}%"
    )

    # Log metrics to MLflow
    mlflow.log_metrics(
        {"train_loss": epoch_loss, "train_accuracy": epoch_acc}, step=epoch
    )

    # Update the learning rate scheduler
    scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


# Testing function
def test(dataloader, model, loss_fn, epoch=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    # Create progress bar for test batches
    progress_bar = tqdm(dataloader, desc="Evaluation [Test]", leave=False, unit="batch")

    with torch.no_grad():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss
            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct

            # Store predictions and labels for detailed metrics
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update progress bar
            batch_acc = batch_correct / len(y)
            progress_bar.set_postfix(
                loss=f"{batch_loss:.4f}", accuracy=f"{100 * batch_acc:.2f}%"
            )

    test_loss /= num_batches
    accuracy = correct / size
    print(
        f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    # Print detailed classification report
    print("Classification Report:")
    target_names = [idx_to_diagnosis[i] for i in range(len(diagnosis_to_idx))]
    report = classification_report(
        all_labels, all_preds, target_names=target_names, output_dict=True
    )
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Log metrics to MLflow
    if epoch is not None:
        mlflow.log_metrics(
            {"test_loss": test_loss, "test_accuracy": accuracy}, step=epoch
        )

        # Log class-specific metrics
        for class_name in target_names:
            if class_name in report:
                mlflow.log_metrics(
                    {
                        f"{class_name}_precision": report[class_name]["precision"],
                        f"{class_name}_recall": report[class_name]["recall"],
                        f"{class_name}_f1-score": report[class_name]["f1-score"],
                    },
                    step=epoch,
                )

    return test_loss, accuracy


# Update the training loop to evaluate less frequently
def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs):
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # Track epochs where test evaluation was performed
    test_epochs = []

    # Create progress bar for epochs
    epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

    for t in epoch_bar:
        # Train for one epoch
        epoch_loss, epoch_acc = train(train_loader, model, loss_fn, optimizer, t)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Only evaluate on test set every 5 epochs or on the final epoch
        if (t + 1) % 5 == 0 or t == epochs - 1:
            test_loss, test_acc = test(test_loader, model, loss_fn, epoch=t)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_epochs.append(t)

            # Update epoch progress bar with test metrics
            epoch_bar.set_postfix(
                train_loss=f"{epoch_loss:.4f}",
                train_acc=f"{100 * epoch_acc:.2f}%",
                test_loss=f"{test_loss:.4f}",
                test_acc=f"{100 * test_acc:.2f}%",
            )
        else:
            # Update epoch progress bar with only training metrics
            epoch_bar.set_postfix(
                train_loss=f"{epoch_loss:.4f}",
                train_acc=f"{100 * epoch_acc:.2f}%",
                test="N/A",
            )

        # Log current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate", current_lr, step=t)

    return train_losses, train_accs, test_losses, test_accs, test_epochs


# Update the plot_training_progress function to handle sparse test data
def plot_training_progress(
    train_losses, train_accs, test_losses=None, test_accs=None, test_epochs=None
):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    if test_losses and test_epochs:
        plt.plot(test_epochs, test_losses, "o-", label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label="Training Accuracy")
    if test_accs and test_epochs:
        plt.plot(test_epochs, test_accs, "o-", label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Save figure for MLflow logging
    plt_path = "training_progress.png"
    plt.savefig(plt_path)
    plt.show()

    return plt_path


# Update the MLflow run to use the modified training function
print("Starting training with MLflow tracking...")
with mlflow.start_run(run_name=config.run_name):
    # Log parameters
    mlflow.log_params(
        {
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "epochs": config.epochs,
            "conv_channels": config.conv_channels,
            "fc_features": config.fc_features,
            "dropout_rates_conv": config.dropout_rates_conv,
            "dropout_rates_fc": config.dropout_rates_fc,
            "scheduler_factor": config.scheduler_factor,
            "scheduler_patience": config.scheduler_patience,
            "train_split": config.train_split,
            "num_classes": len(diagnosis_to_idx),
            "class_mapping": diagnosis_to_idx,
        }
    )

    # Log class distribution
    class_distribution = filtered_df["dx"].value_counts().to_dict()
    mlflow.log_params({f"class_{k}_count": v for k, v in class_distribution.items()})

    # Train the model with less frequent evaluation
    train_losses, train_accs, test_losses, test_accs, test_epochs = train_model(
        model, train_loader, test_loader, optimizer, loss_fn, config.epochs
    )
    print("Training complete!")

    # Plot and log training progress with sparse test data
    plt_path = plot_training_progress(
        train_losses, train_accs, test_losses, test_accs, test_epochs
    )
    mlflow.log_artifact(plt_path)

    # Save and log the model
    torch.save(model.state_dict(), config.model_save_path)
    mlflow.pytorch.log_model(model, "model")
    mlflow.log_artifact(config.model_save_path)
    print(f"Model saved to {config.model_save_path} and logged to MLflow")

    # Log final metrics
    mlflow.log_metrics(
        {
            "final_train_loss": train_losses[-1],
            "final_train_accuracy": train_accs[-1],
            "final_test_loss": test_losses[-1],
            "final_test_accuracy": test_accs[-1],
        }
    )

print("MLflow tracking completed")


# Modify the save function to only export to ONNX format
def save_as_onnx(model, input_shape, base_filename="skin_lesion_model"):
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"{base_filename}_{timestamp}.onnx"

    # Create a directory for models if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    onnx_path = Path("models") / onnx_filename

    # Create dummy input for ONNX export
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1], device=device)

    # Export to ONNX
    print(f"Exporting model to ONNX format: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Model successfully saved in ONNX format: {onnx_path}")
    return onnx_path


# Save the model in ONNX format
onnx_path = save_as_onnx(model, config.image_size)

if onnx_path:
    print(f"Model saved in ONNX format at: {onnx_path}")

    # Log the ONNX model to MLflow
    try:
        with mlflow.start_run(run_name=f"{config.run_name}_onnx_export"):
            mlflow.log_artifact(onnx_path)
            print("ONNX model logged to MLflow")
    except Exception as e:
        print(f"Could not log ONNX model to MLflow: {e}")
