import torch
import mlflow
import numpy as np

# Import from our modules
from config_loader import load_config
from preprocessing import prepare_data
from models import setup_models
from training import train_and_compare_models
from utils import log_parameters, log_training_results, save_model


def main():
    # Initialize configuration
    # config = ModelConfig()
    config = load_config()

    # Set up MLflow experiment
    mlflow.set_experiment(config.experiment_name)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Prepare data
    data = prepare_data(config)
    if data is None:
        return

    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    diagnosis_to_idx = data["diagnosis_to_idx"]
    idx_to_diagnosis = data["idx_to_diagnosis"]
    df = data["df"]

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device", "\n")

    # Set up models, optimizers, and loss function
    model_setup = setup_models(config, len(diagnosis_to_idx), device)
    model = model_setup["model"]
    pretrained_model = model_setup["pretrained_model"]
    optimizer = model_setup["optimizer"]
    pretrained_optimizer = model_setup["pretrained_optimizer"]
    scheduler = model_setup["scheduler"]
    pretrained_scheduler = model_setup["pretrained_scheduler"]
    loss_fn = model_setup["loss_fn"]

    # Get model save format from config (default to "pth" if not specified)
    model_save_format = getattr(config, "model_save_format", "pth")
    print(f"Using model save format: {model_save_format}")

    # Start MLflow run and train the models
    with mlflow.start_run(run_name=config.run_name):
        # Log parameters
        log_parameters(config, diagnosis_to_idx, df["dx"].value_counts().to_dict())

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

        # Log results and determine the better model
        better_model, better_model_obj = log_training_results(
            results,
            model,
            pretrained_model,
            config,
            train_loader,
            device,
            save_format=model_save_format,
        )

        # Export the better model in the specified format
        save_model(
            better_model,
            config,
            device,
            model_type=model_save_format,
        )

    print("Training and evaluation completed!")


if __name__ == "__main__":
    main()
