import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class SkinLesionDataset(Dataset):
    """Dataset for skin lesion images."""

    def __init__(
        self, images, labels, transform, transform_minority=None, diagnosis_to_idx=None
    ):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.transform_minority = transform_minority

        # Store class frequencies for minority detection
        if diagnosis_to_idx:
            self.class_counts = {}
            for label in labels:
                if label not in self.class_counts:
                    self.class_counts[label] = 0
                self.class_counts[label] += 1

            # Find classes with fewer samples (minority classes)
            avg_count = len(labels) / len(diagnosis_to_idx)
            self.minority_classes = [
                cls
                for cls, count in self.class_counts.items()
                if count < avg_count * 0.5  # Classes with less than 50% of average
            ]
            print(
                f"Identified {len(self.minority_classes)} minority classes: {self.minority_classes}"
            )
        else:
            self.minority_classes = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        # Load the image from the path
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        # Apply minority class transforms if applicable
        if self.transform_minority and label in self.minority_classes:
            image = self.transform_minority(image)
        else:
            image = self.transform(image)

        return image, label


def get_image_path(img_id, config):
    """Get the path to an image file."""
    path = os.path.join(config.images_dir, f"{img_id}.jpg")
    if os.path.exists(path):
        return path
    return None


def load_images(df, diagnosis_to_idx, config):
    """Load images and labels from the dataframe."""
    image_paths = []
    labels = []

    for idx, row in df.iterrows():
        img_id = row["image_id"]
        path = get_image_path(img_id, config)
        if path:
            image_paths.append(path)
            labels.append(diagnosis_to_idx[row["dx"]])

    print(f"Found {len(image_paths)} valid images out of {len(df)} entries")
    return image_paths, labels


def create_class_balanced_sampler(dataset, config):
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


def prepare_data(config):
    """Prepare data for training and evaluation."""
    # Check if the CSV file exists
    try:
        df = pd.read_csv(config.csv_path)
        print(f"Loaded {len(df)} rows from {config.csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {config.csv_path}")
        return None

    # Create a mapping from diagnosis to numerical label
    if "dx" in df.columns:
        unique_diagnoses = df["dx"].unique()
        diagnosis_to_idx = {
            diagnosis: idx for idx, diagnosis in enumerate(unique_diagnoses)
        }
        idx_to_diagnosis = {
            idx: diagnosis for diagnosis, idx in diagnosis_to_idx.items()
        }
        print(f"Diagnosis classes: {diagnosis_to_idx}")
    else:
        print("Error: 'dx' column not found in CSV file")
        return None

    # Load images and labels
    image_paths, labels = load_images(df, diagnosis_to_idx, config)

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
    dataset = SkinLesionDataset(
        image_paths,
        labels,
        transform=transform,
        transform_minority=transform_minority,
        diagnosis_to_idx=diagnosis_to_idx,
    )

    # Split into train and test sets
    train_size = int(config.train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")

    # Create a balanced sampler for training
    train_sampler = create_class_balanced_sampler(train_dataset, config)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler
    )
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "diagnosis_to_idx": diagnosis_to_idx,
        "idx_to_diagnosis": idx_to_diagnosis,
        "df": df,
    }
