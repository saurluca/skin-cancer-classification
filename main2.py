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

np.random.seed(42)
torch.manual_seed(42)


# Update the paths to match your actual directory structure
# Assuming the CSV file is in the current directory
csv_path = "HAM10000_metadata.csv"
# Update this to the correct path where your subset of images are stored
# images_path = "data/train/"  # Changed back to your original path
# path to all images
# all_images_dir_path = "/home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/ham10000_images_part_1/"

# combine images from all 4 folders
base_path = (
    "/home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/"
)
folders = [
    "ham10000_images_part_1",
    "ham10000_images_part_2",
    "HAM10000_images_part_1",
    "HAM10000_images_part_2",
]


# Check if the CSV file exists
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)

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
print(f"Attempting to load images from multiple folders in {base_path}")
images, found_image_ids = load_images_from_folders(
    df["image_id"].tolist(), base_path, folders
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
        transforms.Resize((28, 28)),  # Resize to match our model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Add data augmentation for minority classes
transform_minority = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create image paths and labels
image_paths = []
for img_id in filtered_df["image_id"]:
    # Try to find the image in each folder
    found = False
    for folder in folders:
        path = os.path.join(base_path, folder, f"{img_id}.jpg")
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

# Create the dataset
dataset = SkinLesionDataset(image_paths, labels, transform=transform)

# Split into train and test sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Enhanced CNN model with more dropout layers
class CNNModel(nn.Module):
    def __init__(self, num_classes=len(diagnosis_to_idx)):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),  # Add dropout to convolutional layers
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Increase dropout rate in deeper layers
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.3),  # Even higher dropout rate
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout for fully connected layers
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),  # Additional fully connected layer with dropout
            nn.Linear(256, num_classes),
        )

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

model = CNNModel().to(device)
print(model)

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

loss_fn = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(
    model.parameters(), lr=1e-3, weight_decay=1e-5
)  # Use Adam with weight decay
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2
)


# Training function
def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
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

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Epoch {epoch + 1}: loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / size
    print(
        f"Epoch {epoch + 1}: Avg loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_acc:.2f}%"
    )

    # Update the learning rate scheduler
    scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


# Testing function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Store predictions and labels for detailed metrics
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss /= num_batches
    accuracy = correct / size
    print(
        f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    # Print detailed classification report
    print("Classification Report:")
    target_names = [idx_to_diagnosis[i] for i in range(len(diagnosis_to_idx))]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    return test_loss, accuracy


# Train the model
epochs = 10   # Train for more epochs
train_losses = []
train_accs = []

print("Starting training...")
for t in range(epochs):
    epoch_loss, epoch_acc = train(train_loader, model, loss_fn, optimizer, t)
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
print("Training complete!")

# Evaluate on the test set
print("\nEvaluating on test set:")
test_loss, test_acc = test(test_loader, model, loss_fn)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_acc:.2f}%")
print("Test complete!")

# Save the model
torch.save(model.state_dict(), "skin_lesion_model.pth")
print("Model saved to skin_lesion_model.pth")
