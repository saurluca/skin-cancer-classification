import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np

# Import from our modules
from config import ModelConfig
from preprocessing import load_images, SkinLesionDataset, create_class_balanced_sampler
from models import CNNModel, PretrainedModel
from training import train_and_compare_models
from visualization import plot_model_comparison
from utils import save_as_onnx, save_pytorch_model, log_model_to_mlflow


def main():
    # Initialize configuration
    config = ModelConfig()

    # Set up MLflow experiment
    mlflow.set_experiment(config.experiment_name)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Check if the CSV file exists
    try:
        df = pd.read_csv(config.csv_path)
        print(f"Loaded {len(df)} rows from {config.csv_path}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at {config.csv_path}")
        return

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
        return

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

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device", "\n")

    # Create the custom CNN model
    model = CNNModel(config, len(diagnosis_to_idx)).to(device)
    print(model)

    # Define loss function with class weights if needed
    loss_fn = nn.CrossEntropyLoss()

    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )

    # Create the pre-trained model
    pretrained_model = PretrainedModel(
        len(diagnosis_to_idx), pretrained_model="resnet18"
    ).to(device)
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

    # Start MLflow run and train the models
    with mlflow.start_run(run_name=config.run_name) as run:
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
        class_distribution = df["dx"].value_counts().to_dict()
        mlflow.log_params(
            {f"class_{k}_count": v for k, v in class_distribution.items()}
        )

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
            device,
            idx_to_diagnosis,
            diagnosis_to_idx,
        )

        print("Training complete!")

        # Plot and log comparison results
        comparison_plt_path = plot_model_comparison(results)
        mlflow.log_artifact(comparison_plt_path)

        # Save both models using the utilities
        save_pytorch_model(model, config.model_save_path)
        save_pytorch_model(pretrained_model, "pretrained_" + config.model_save_path)

        # Create input examples for signature inference
        example_inputs, _ = next(iter(train_loader))
        example_inputs = example_inputs.to(device)

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
                "custom_cnn_final_train_loss": results["custom_cnn"]["train_losses"][
                    -1
                ],
                "custom_cnn_final_train_accuracy": results["custom_cnn"]["train_accs"][
                    -1
                ],
                "custom_cnn_final_test_loss": results["custom_cnn"]["test_losses"][-1],
                "custom_cnn_final_test_accuracy": results["custom_cnn"]["test_accs"][
                    -1
                ],
                "pretrained_final_train_loss": results["pretrained"]["train_losses"][
                    -1
                ],
                "pretrained_final_train_accuracy": results["pretrained"]["train_accs"][
                    -1
                ],
                "pretrained_final_test_loss": results["pretrained"]["test_losses"][-1],
                "pretrained_final_test_accuracy": results["pretrained"]["test_accs"][
                    -1
                ],
            }
        )

        # Determine which model performed better
        if (
            results["custom_cnn"]["test_accs"][-1]
            > results["pretrained"]["test_accs"][-1]
        ):
            better_model = "custom_cnn"
            better_model_obj = model
        else:
            better_model = "pretrained"
            better_model_obj = pretrained_model

        mlflow.log_param("better_model", better_model)
        print(
            f"The {better_model} model performed better with test accuracy: {results[better_model]['test_accs'][-1]:.4f}"
        )

        # Export the better model to ONNX
        onnx_path = save_as_onnx(
            better_model_obj,
            config.image_size,
            device,
            base_filename=f"skin_lesion_{better_model}",
        )
        print(f"Better model saved in ONNX format at: {onnx_path}")

        # Log both PyTorch and ONNX models to MLflow
        try:
            # Get example input for inference
            example_inputs = torch.randn(
                1, 3, config.image_size[0], config.image_size[1], device=device
            )

            # Use the consolidated logging function - no need to pass run_id since we're in an active run
            log_model_to_mlflow(better_model_obj, onnx_path, example_inputs)

        except Exception as e:
            print(f"Could not log models to MLflow: {e}")

    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
