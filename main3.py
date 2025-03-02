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
import torchvision.models as models
import datetime
from pathlib import Path
from mlflow.models.signature import infer_signature


@dataclass
class ModelConfig:
    # Data parameters
    batch_size: int = 32
    image_size: tuple = (28, 28)  # Increased resolution for better feature extraction
    train_split: float = 0.8

    # Model parameters
    conv_channels: list = (32, 64, 128, 256)  # Added another conv layer
    fc_features: list = (512, 256, 128)  # Added another FC layer
    dropout_rates_conv: list = (0.1, 0.15, 0.2, 0.25)  # More gradual dropout increase
    dropout_rates_fc: list = (0.4, 0.3, 0.2)  # Decreasing dropout in FC layers

    # Training parameters
    learning_rate: float = 5e-4  # Slightly lower learning rate
    weight_decay: float = 2e-5  # Increased regularization
    epochs: int = 3  # More epochs for better convergence
    scheduler_factor: float = 0.7  # Less aggressive LR reduction
    scheduler_patience: int = 3  # More patience before reducing LR

    # Class weight parameters
    class_weight_power: float = (
        0.3  # Use this to control weight intensity in the sampler
    )

    # Paths
    model_save_path: str = "skin_lesion_model.pth"
    csv_path: str = "data/HAM10000_metadata.csv"  # Updated path
    images_dir: str = "data/images"  # New path for images

    # MLflow parameters
    experiment_name: str = "skin-lesion-classification"
    run_name: str = "cnn-model"

    def __post_init__(self):
        pass  # No need for folders list anymore


# Initialize configuration
config = ModelConfig()

# Set up MLflow experiment
mlflow.set_experiment(config.experiment_name)

np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(config.csv_path)

# Create a mapping from diagnosis to numerical label
if "dx" in df.columns:
    unique_diagnoses = df["dx"].unique()
    diagnosis_to_idx = {
        diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)
    }
    idx_to_diagnosis = {idx: diagnosis for diagnosis, idx in diagnosis_to_idx.items()}
    print(f"Diagnosis classes: {diagnosis_to_idx}")


# Simplify the image loading function since we only have one directory now
def get_image_path(img_id):
    """Get the path to an image file."""
    path = os.path.join(config.images_dir, f"{img_id}.jpg")
    if os.path.exists(path):
        return path
    return None


# Create image paths and labels for the dataset
def load_images(df, diagnosis_to_idx):
    image_paths = []
    labels = []

    for idx, row in df.iterrows():
        img_id = row["image_id"]
        path = get_image_path(img_id)
        if path:
            image_paths.append(path)
            labels.append(diagnosis_to_idx[row["dx"]])

    print(f"Found {len(image_paths)} valid images out of {len(df)} entries")
    return image_paths, labels


# Try to load the images from the images directory
print(f"Attempting to load images from directory: {config.images_dir}")
images, labels = load_images(df, diagnosis_to_idx)

print(f"Successfully loaded {len(images)} images")

# Filter the dataframe to include only the images we found
filtered_df = df[
    df["image_id"].isin([os.path.splitext(os.path.basename(img))[0] for img in images])
]
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

# Create the dataset
dataset = SkinLesionDataset(images, labels, transform=transform)

# Split into train and test sets
train_size = int(config.train_split * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

print("Creating balanced sampler for training set")


# Add the missing create_class_balanced_sampler function
def create_class_balanced_sampler(dataset):
    """Create a sampler that balances class distribution during training."""
    # Get all labels in the dataset
    targets = [dataset[i][1] for i in range(len(dataset))]

    # Count samples per class
    class_counts = torch.bincount(torch.tensor(targets))
    print(f"Class distribution: {class_counts}")

    # Calculate weights for each sample
    weights = 1.0 / class_counts[targets]

    # Apply power to control weight intensity
    weights = weights**config.class_weight_power

    # Create and return the sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )

    print(f"Created balanced sampler with {len(weights)} weights")
    return sampler


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


# Define a secont model

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
        if (t + 1) % 3 == 0 or t == epochs - 1:
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


# Define a second model based on pre-trained ResNet
class PretrainedModel(nn.Module):
    def __init__(self, num_classes=None, pretrained_model="resnet18"):
        super().__init__()
        if num_classes is None:
            num_classes = len(diagnosis_to_idx)

        # Load pre-trained model
        if pretrained_model == "resnet18":
            self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        elif pretrained_model == "resnet50":
            self.base_model = models.resnet50(weights="IMAGENET1K_V1")
        elif pretrained_model == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model: {pretrained_model}")

        # Freeze early layers to prevent overfitting
        for param in list(self.base_model.parameters())[
            :-20
        ]:  # Freeze all but last few layers
            param.requires_grad = False

        # Replace the final fully connected layer
        if pretrained_model.startswith("resnet"):
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes)
            )
        elif pretrained_model.startswith("efficientnet"):
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes)
            )

    def forward(self, x):
        return self.base_model(x)


# Create the second model (pre-trained)
pretrained_model = PretrainedModel(pretrained_model="resnet18").to(device)
print("Pre-trained model architecture:")
print(pretrained_model)

# Create optimizer and scheduler for the pre-trained model
pretrained_optimizer = optim.Adam(
    pretrained_model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
pretrained_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    pretrained_optimizer,
    mode="min",
    factor=config.scheduler_factor,
    patience=config.scheduler_patience,
)


# Define a function to train and evaluate both models
def train_and_compare_models(
    model1,
    model2,
    train_loader,
    test_loader,
    optimizer1,
    optimizer2,
    scheduler1,
    scheduler2,
    loss_fn,
    epochs,
):
    # Dictionary to store results
    results = {
        "custom_cnn": {
            "train_losses": [],
            "train_accs": [],
            "test_losses": [],
            "test_accs": [],
            "test_epochs": [],
        },
        "pretrained": {
            "train_losses": [],
            "train_accs": [],
            "test_losses": [],
            "test_accs": [],
            "test_epochs": [],
        },
    }

    print("\n" + "=" * 50)
    print("Starting training of both models...")
    print("=" * 50 + "\n")

    # Train both models for the specified number of epochs
    for t in range(epochs):
        print(f"\nEpoch {t + 1}/{epochs}")
        print("-" * 30)

        # Train and evaluate custom CNN
        print("Training custom CNN model:")
        epoch_loss, epoch_acc = train(train_loader, model1, loss_fn, optimizer1, t)
        results["custom_cnn"]["train_losses"].append(epoch_loss)
        results["custom_cnn"]["train_accs"].append(epoch_acc)

        # Train and evaluate pre-trained model
        print("\nTraining pre-trained model:")
        epoch_loss, epoch_acc = train(train_loader, model2, loss_fn, optimizer2, t)
        results["pretrained"]["train_losses"].append(epoch_loss)
        results["pretrained"]["train_accs"].append(epoch_acc)

        # Evaluate both models on test set every 3 epochs or on the final epoch
        if (t + 1) % 3 == 0 or t == epochs - 1:
            print("\nEvaluating custom CNN model:")
            test_loss, test_acc = test(test_loader, model1, loss_fn, epoch=t)
            results["custom_cnn"]["test_losses"].append(test_loss)
            results["custom_cnn"]["test_accs"].append(test_acc)
            results["custom_cnn"]["test_epochs"].append(t)

            print("\nEvaluating pre-trained model:")
            test_loss, test_acc = test(test_loader, model2, loss_fn, epoch=t)
            results["pretrained"]["test_losses"].append(test_loss)
            results["pretrained"]["test_accs"].append(test_acc)
            results["pretrained"]["test_epochs"].append(t)

        # Update schedulers
        scheduler1.step(results["custom_cnn"]["train_losses"][-1])
        scheduler2.step(results["pretrained"]["train_losses"][-1])

    return results


# Function to plot comparison of model performances
def plot_model_comparison(results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training losses
    axes[0, 0].plot(results["custom_cnn"]["train_losses"], label="Custom CNN")
    axes[0, 0].plot(results["pretrained"]["train_losses"], label="Pre-trained")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Plot training accuracies
    axes[0, 1].plot(results["custom_cnn"]["train_accs"], label="Custom CNN")
    axes[0, 1].plot(results["pretrained"]["train_accs"], label="Pre-trained")
    axes[0, 1].set_title("Training Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Plot test losses
    axes[1, 0].plot(
        results["custom_cnn"]["test_epochs"],
        results["custom_cnn"]["test_losses"],
        "o-",
        label="Custom CNN",
    )
    axes[1, 0].plot(
        results["pretrained"]["test_epochs"],
        results["pretrained"]["test_losses"],
        "o-",
        label="Pre-trained",
    )
    axes[1, 0].set_title("Test Loss")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Plot test accuracies
    axes[1, 1].plot(
        results["custom_cnn"]["test_epochs"],
        results["custom_cnn"]["test_accs"],
        "o-",
        label="Custom CNN",
    )
    axes[1, 1].plot(
        results["pretrained"]["test_epochs"],
        results["pretrained"]["test_accs"],
        "o-",
        label="Pre-trained",
    )
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt_path = "model_comparison.png"
    plt.savefig(plt_path)
    plt.show()

    return plt_path


# Replace your existing MLflow training code with this:
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
            "class_weight_power": config.class_weight_power,
            "model_comparison": "custom_cnn_vs_pretrained_resnet18",
        }
    )

    # Log class distribution
    class_distribution = filtered_df["dx"].value_counts().to_dict()
    mlflow.log_params({f"class_{k}_count": v for k, v in class_distribution.items()})

    # Train both models and compare
    results = train_and_compare_models(
        model,
        pretrained_model,
        train_loader,
        test_loader,
        optimizer,
        pretrained_optimizer,
        scheduler,
        pretrained_scheduler,
        loss_fn,
        config.epochs,
    )

    print("Training complete!")

    # Plot and log comparison results
    comparison_plt_path = plot_model_comparison(results)
    mlflow.log_artifact(comparison_plt_path)

    # Save and log both models
    torch.save(model.state_dict(), config.model_save_path)
    torch.save(pretrained_model.state_dict(), "pretrained_" + config.model_save_path)

    # Create input examples for signature inference
    # Get a batch from the dataloader
    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs.to(device)

    # Get model outputs for the examples
    with torch.no_grad():
        model.eval()
        example_outputs_cnn = model(example_inputs)

        pretrained_model.eval()
        example_outputs_pretrained = pretrained_model(example_inputs)

    # Infer signatures
    cnn_signature = infer_signature(
        example_inputs.cpu().numpy(), example_outputs_cnn.cpu().numpy()
    )

    pretrained_signature = infer_signature(
        example_inputs.cpu().numpy(), example_outputs_pretrained.cpu().numpy()
    )

    # Log models with signatures and input examples
    mlflow.pytorch.log_model(
        model,
        "custom_cnn_model",
        signature=cnn_signature,
        input_example=example_inputs.cpu().numpy(),
    )

    mlflow.pytorch.log_model(
        pretrained_model,
        "pretrained_model",
        signature=pretrained_signature,
        input_example=example_inputs.cpu().numpy(),
    )

    mlflow.log_artifact(config.model_save_path)
    mlflow.log_artifact("pretrained_" + config.model_save_path)

    print("Models saved and logged to MLflow")

    # Log final metrics for both models
    mlflow.log_metrics(
        {
            "custom_cnn_final_train_loss": results["custom_cnn"]["train_losses"][-1],
            "custom_cnn_final_train_accuracy": results["custom_cnn"]["train_accs"][-1],
            "custom_cnn_final_test_loss": results["custom_cnn"]["test_losses"][-1],
            "custom_cnn_final_test_accuracy": results["custom_cnn"]["test_accs"][-1],
            "pretrained_final_train_loss": results["pretrained"]["train_losses"][-1],
            "pretrained_final_train_accuracy": results["pretrained"]["train_accs"][-1],
            "pretrained_final_test_loss": results["pretrained"]["test_losses"][-1],
            "pretrained_final_test_accuracy": results["pretrained"]["test_accs"][-1],
        }
    )

    # Determine which model performed better
    if results["custom_cnn"]["test_accs"][-1] > results["pretrained"]["test_accs"][-1]:
        better_model = "custom_cnn"
        better_model_path = config.model_save_path
    else:
        better_model = "pretrained"
        better_model_path = "pretrained_" + config.model_save_path

    mlflow.log_param("better_model", better_model)
    print(
        f"The {better_model} model performed better with test accuracy: {max(results[better_model]['test_accs'][-1], results[better_model]['test_accs'][-1]):.4f}"
    )


# Define the save_as_onnx function
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


print("MLflow tracking completed")

# Export the better model to ONNX
better_model_obj = model if better_model == "custom_cnn" else pretrained_model
onnx_path = save_as_onnx(
    better_model_obj, config.image_size, base_filename=f"skin_lesion_{better_model}"
)

if onnx_path:
    print(f"Better model saved in ONNX format at: {onnx_path}")

    # Log the ONNX model to MLflow
    try:
        # Get example inputs for signature
        example_inputs, _ = next(iter(train_loader))
        example_inputs = example_inputs.to(device)

        # Get model outputs
        with torch.no_grad():
            better_model_obj.eval()
            example_outputs = better_model_obj(example_inputs)

        # Infer signature
        signature = infer_signature(
            example_inputs.cpu().numpy(), example_outputs.cpu().numpy()
        )

        with mlflow.start_run(run_name=f"{config.run_name}_{better_model}_onnx_export"):
            # Log the ONNX model with signature
            mlflow.onnx.log_model(
                onnx_model=onnx_path,
                artifact_path="onnx_model",
                signature=signature,
                input_example=example_inputs.cpu().numpy(),
            )
            print("ONNX model logged to MLflow with signature")
    except Exception as e:
        print(f"Could not log ONNX model to MLflow: {e}")
