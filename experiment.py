import torch
import mlflow
from mlflow.models.signature import infer_signature
from visualization import plot_model_comparison
from utils import save_as_onnx, save_pytorch_model, log_model_to_mlflow


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


def log_training_results(
    results, model, pretrained_model, config, train_loader, device
):
    """Log training results, save models, and determine the better model."""
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


def export_best_model(better_model, better_model_obj, config, device):
    """Export the better model to ONNX format and log it to MLflow."""
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

    return onnx_path
