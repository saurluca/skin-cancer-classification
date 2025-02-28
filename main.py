import pandas as pd
import os
import cv2
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
from pathlib import Path


# Function to load images
def load_images(image_ids, image_dirs):
    images = []
    not_found = []

    for img_id in image_ids:
        img_filename = img_id + ".jpg"
        found = False

        # Try to find the image in any of the directories
        for img_dir in image_dirs:
            img_path = os.path.join(img_dir, img_filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                found = True
                break

        if not found:
            not_found.append(img_id)

    if not_found:
        print(f"Warning: Could not find {len(not_found)} images")
        if (
            len(not_found) < 10
        ):  # Only print a few missing IDs to avoid cluttering output
            print(f"Missing image IDs: {not_found}")
        else:
            print(f"First 10 missing image IDs: {not_found[:10]}...")

    return images


# Function to load a small test dataset (10 images per class)
def load_test_dataset(df, image_dirs, sample_size=10):
    print(f"\nLoading test dataset with {sample_size} images per class...")

    # Get a balanced sample of images from each class
    classes = df["dx"].unique()
    sampled_df = pd.DataFrame()

    for class_name in classes:
        class_df = df[df["dx"] == class_name].head(sample_size)
        sampled_df = pd.concat([sampled_df, class_df])

    # Load the images
    sampled_images = load_images(sampled_df["image_id"].tolist(), image_dirs)

    print(f"Successfully loaded {len(sampled_images)} test images")

    # Split data for training and validation
    train_df, val_df = train_test_split(sampled_df, test_size=0.2, random_state=42)
    test_df = sampled_df.sample(min(sample_size, len(sampled_df)), random_state=42)

    return train_df, val_df, test_df, sampled_images


# Function to load the full dataset
def load_full_dataset(df, image_dirs):
    print("\nLoading full dataset...")

    # Load all images
    all_images = load_images(df["image_id"].tolist(), image_dirs)

    print(f"Successfully loaded {len(all_images)} images out of {len(df)} records")

    # Split data for training and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    test_df = df.sample(
        min(100, len(df)), random_state=42
    )  # Use 100 images for testing

    return train_df, val_df, test_df, all_images


# Function to save images for classification (organized by class)
def save_images_for_classification(df, split, images_list, images_dir):
    saved_count = 0
    for i, row in df.iterrows():
        img_id = row["image_id"]
        class_name = row["dx"]  # Get the class name from the dx column

        # Find the image in the loaded images
        img_index = df.index.get_loc(i)
        if img_index < len(images_list):
            img = images_list[img_index]

            # Create class directory if it doesn't exist
            class_dir = images_dir / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save image in the class directory
            img_path = class_dir / f"{img_id}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            saved_count += 1

    return saved_count


def main():
    # Download latest version
    # path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    path = "/home/luca/.cache/kagglehub/datasets/kmader/skin-cancer-mnist-ham10000/versions/2"

    print("Path to dataset files:", path)

    df = pd.read_csv(os.path.join(path, "HAM10000_metadata.csv"))

    # Define all possible image directory paths (both uppercase and lowercase variants)
    image_dirs = [
        os.path.join(path, "HAM10000_images_part_1"),
        os.path.join(path, "HAM10000_images_part_2"),
        os.path.join(path, "ham10000_images_part_1"),
        os.path.join(path, "ham10000_images_part_2"),
    ]

    # Filter out directories that don't exist
    image_dirs = [dir_path for dir_path in image_dirs if os.path.exists(dir_path)]

    print(f"Found {len(image_dirs)} image directories:")
    for dir_path in image_dirs:
        print(f"  - {dir_path}")

    # Check for additional data files
    additional_files = [
        "hmnist_28_28_RGB.csv",
        "hmnist_28_28_L.csv",
        "hmnist_8_8_L.csv",
        "hmnist_8_8_RGB.csv",
    ]

    print("\nChecking for additional data files:")
    for file_name in additional_files:
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            print(f"  - {file_name}: Found")
        else:
            print(f"  - {file_name}: Not found")

    # Choose which dataset to use
    USE_TEST_DATASET = True  # Set to False to use the full dataset
    TEST_SAMPLE_SIZE = 10

    if USE_TEST_DATASET:
        train_df, val_df, test_df, images = load_test_dataset(
            df, image_dirs, TEST_SAMPLE_SIZE
        )
    else:
        train_df, val_df, test_df, images = load_full_dataset(df, image_dirs)

    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")

    # Create dataset directories for classification
    dataset_dir = Path("dataset")
    images_dir = dataset_dir / "images"

    # Create train/val/test directories
    for split in ["train", "val", "test"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)

    print("Saving training images...")
    train_saved = save_images_for_classification(train_df, "train", images, images_dir)
    print(f"Saved {train_saved} training images")

    print("Saving validation images...")
    val_saved = save_images_for_classification(val_df, "val", images, images_dir)
    print(f"Saved {val_saved} validation images")

    print("Saving test images...")
    test_saved = save_images_for_classification(test_df, "test", images, images_dir)
    print(f"Saved {test_saved} test images")

    # Create data.yaml file
    data_yaml_path = dataset_dir / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(f"""
path: {dataset_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 7  # Number of classes in the HAM10000 dataset
names: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Class names from HAM10000
""")

    print("Initializing classification model")
    # Initialize YOLO classification model instead of detection
    model = YOLO("yolov8n-cls.pt")  # Load a pretrained YOLOv8 classification model

    # Train the model with the proper data.yaml path
    results = model.train(
        data=str(data_yaml_path),  # Use the created YAML file
        epochs=5,  # Increased for better learning
        imgsz=224,  # Standard size for classification
        batch=16,  # Adjusted batch size
        name="skin_cancer_classifier",
    )

    # Evaluate the model on test data
    print("\nEvaluating model on test data...")
    metrics = model.val()

    print("\nClassification Results:")
    print(f"Accuracy: {metrics.top1:.4f}")
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")

    # Perform inference on a few test images
    test_images_list = list(images_dir.glob("test/*.jpg"))
    if test_images_list:
        print("\nRunning inference on test images...")
        # Get ground truth labels
        test_labels = test_df["dx"].tolist()

        # Run inference on test images
        predictions = []
        for i, img_path in enumerate(
            test_images_list[:10]
        ):  # Just use first 10 for demo
            result = model(str(img_path))
            pred_class = result[0].probs.top1
            pred_class_name = result[0].names[pred_class]
            true_class = test_labels[i] if i < len(test_labels) else "unknown"

            predictions.append((true_class, pred_class_name))
            print(f"Image {i + 1}: True: {true_class}, Predicted: {pred_class_name}")

        # Calculate classification error
        correct = sum(1 for true, pred in predictions if true == pred)
        total = len(predictions)
        error_rate = 1.0 - (correct / total) if total > 0 else 0

        print(
            f"\nClassification Error: {error_rate:.4f} ({total - correct} incorrect out of {total})"
        )
        print(
            f"Classification Accuracy: {1 - error_rate:.4f} ({correct} correct out of {total})"
        )

        # Save a visualization of results
        result = model(str(test_images_list[0]))
        result[0].save("classification_result.jpg")
    else:
        print("No test images available for inference")


if __name__ == "__main__":
    main()
