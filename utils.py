import torch
import datetime
import os
import mlflow
from pathlib import Path
from visualization import plot_model_comparison
from mlflow.models.signature import infer_signature
import onnx


def determine_better_model(results, model, pretrained_model):
    """Determine which model performed better and return it."""
    if results["custom_cnn"]["test_accs"][-1] > results["pretrained"]["test_accs"][-1]:
        better_model = "custom_cnn"
        better_model_obj = model
    else:
        better_model = "pretrained"
        better_model_obj = pretrained_model

    mlflow.log_param("better_model", better_model)
    print(
        f"The {better_model} model performed better with test accuracy: {results[better_model]['test_accs'][-1]:.4f}"
    )

    return better_model, better_model_obj


def log_parameters(config, diagnosis_to_idx, class_distribution):
    """Log model and dataset parameters to MLflow."""
    # Log model and training parameters
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
    mlflow.log_params({f"class_{k}_count": v for k, v in class_distribution.items()})


def log_model_to_mlflow(model, model_path, example_inputs, model_type="pth"):
    """
    Log model to MLflow.

    Args:
        model: The model to log
        model_path: Path to the saved model
        example_inputs: Example input tensor for signature inference
        model_type: 'pth' for PyTorch format or 'onnx' for ONNX format
    """
    try:
        # Generate model outputs for signature
        with torch.no_grad():
            model.eval()
            example_outputs = model(example_inputs)

        # Infer signature for the model
        signature = infer_signature(
            example_inputs.cpu().numpy(), example_outputs.cpu().numpy()
        )

        # Log the appropriate model type
        if model_type.lower() == "pth":
            mlflow.pytorch.log_model(
                model,
                artifact_path="pytorch_model",
                signature=signature,
                input_example=example_inputs.cpu().numpy(),
            )
            mlflow.log_artifact(model_path)

        elif model_type.lower() == "onnx":
            # Load the ONNX model
            onnx_model = onnx.load(model_path)

            # Log ONNX model
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="onnx_model",
                signature=signature,
                input_example=example_inputs.cpu().numpy(),
            )

        print(f"Model successfully logged to MLflow in {model_type} format")

    except Exception as e:
        print(f"Error logging model to MLflow: {e}")
        import traceback

        traceback.print_exc()


def log_training_results(
    results, model, pretrained_model, config, train_loader, device, save_format="pth"
):
    """Log training results, save models, and determine the better model."""
    # Plot and log comparison results
    comparison_plt_path = plot_model_comparison(results)
    mlflow.log_artifact(comparison_plt_path)

    # Save both models using the unified save_model function
    model_path = save_model(
        model,
        config,
        device,
        model_type=save_format,
        base_filename=config.model_save_path,
    )

    pretrained_model_path = save_model(
        pretrained_model,
        config,
        device,
        model_type=save_format,
        base_filename="pretrained_" + config.model_save_path,
    )

    # Create input examples for signature inference
    example_inputs, _ = next(iter(train_loader))
    example_inputs = example_inputs.to(device)

    # Log models with MLflow
    log_model_to_mlflow(model, model_path, example_inputs, model_type=save_format)
    log_model_to_mlflow(
        pretrained_model, pretrained_model_path, example_inputs, model_type=save_format
    )

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

    return determine_better_model(results, model, pretrained_model)


def save_model(model, config, device, model_type="pth", base_filename=None):
    """
    Save model to disk in either PyTorch or ONNX format.

    Args:
        model: The model to save
        config: Configuration object
        device: The device (CPU/GPU) to use
        model_type: 'pth' for PyTorch format or 'onnx' for ONNX format
        base_filename: Base filename for the saved model

    Returns:
        Path to the saved model
    """
    # Create a directory for models if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    if model_type.lower() == "pth":
        # Use provided filename or default from config
        path = base_filename or config.model_save_path

        # If path doesn't include directory, add it
        if "/" not in path and "\\" not in path:
            path = os.path.join("models", path)

        # Save the model
        torch.save(model.state_dict(), path)
        print(f"PyTorch model successfully saved at: {path}")
        return path

    elif model_type.lower() == "onnx":
        # Use provided filename or generate a default with timestamp
        if base_filename is None:
            base_filename = "skin_lesion_model"

        # Create a timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        onnx_filename = f"{base_filename}_{timestamp}.onnx"
        onnx_path = str(Path("models") / onnx_filename)

        # Create dummy input for ONNX export
        dummy_input = torch.randn(
            1, 3, config.image_size[0], config.image_size[1], device=device
        )

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

    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'pth' or 'onnx'.")
