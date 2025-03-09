import os
import torch
from torch.utils.data import Dataset
from PIL import Image


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
