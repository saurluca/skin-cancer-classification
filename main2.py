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
all_images_dir_path = "/home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2/ham10000_images_part_1/"


# Check if the CSV file exists
if not os.path.exists(csv_path):
    print(f"Error: CSV file not found at {csv_path}")
    exit(1)

# Check if the image directory exists
if not os.path.exists(all_images_dir_path):
    print(f"Error: Image directory not found at {all_images_dir_path}")
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


def load_images(image_ids, image_dir):
    images = []
    missing_images = []
    found_images = []

    for image_id in image_ids:
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        # Check if the image file exists
        if os.path.exists(image_path):
            image = Image.open(image_path)
            images.append(image)
            found_images.append(image_id)
        else:
            missing_images.append(image_id)

    # Print information about missing and found images
    print(f"Total images in CSV: {len(image_ids)}")
    print(f"Found: {len(found_images)} images")
    print(f"Missing: {len(missing_images)} images")

    if found_images:
        print(f"First few found: {found_images[:5]}")

    if missing_images:
        print(f"First few missing: {missing_images[:5]}")

    return images, found_images


# Try to load the images
print(f"Attempting to load images from {all_images_dir_path}")
images, found_image_ids = load_images(df["image_id"].tolist(), all_images_dir_path)

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

# Create image paths and labels
image_paths = [
    os.path.join(all_images_dir_path, f"{img_id}.jpg") for img_id in filtered_df["image_id"]
]
labels = [diagnosis_to_idx[diagnosis] for diagnosis in filtered_df["dx"]]

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


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=len(diagnosis_to_idx)):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28 * 3, 512),  # 3 channels for RGB
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device", "\n")

model = NeuralNetwork().to(device)
print(model)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


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
epochs = 5
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
