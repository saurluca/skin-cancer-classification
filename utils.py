import torch
import datetime
import os
import mlflow
from pathlib import Path
from mlflow.models.signature import infer_signature
import onnx


def save_pytorch_model(model, path="skin_lesion_model.pth"):
    """Save PyTorch model to disk."""
    # Create a directory for models if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    # If path doesn't include directory, add it
    if "/" not in path and "\\" not in path:
        path = os.path.join("models", path)

    # Save the model
    torch.save(model.state_dict(), path)
    print(f"PyTorch model successfully saved at: {path}")
    return path


def save_as_onnx(model, input_shape, device, base_filename="skin_lesion_model"):
    """Export model to ONNX format."""
    # Create a timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    onnx_filename = f"{base_filename}_{timestamp}.onnx"

    # Create a directory for models if it doesn't exist
    Path("models").mkdir(exist_ok=True)

    onnx_path = str(Path("models") / onnx_filename)

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


def log_model_to_mlflow(model, onnx_path, example_inputs, run_id=None):
    """Log model to MLflow including PyTorch and ONNX formats."""
    try:
        # If run_id is provided, set the active run
        if run_id:
            mlflow.start_run(run_id=run_id)

        # Log PyTorch model
        with torch.no_grad():
            model.eval()
            example_outputs = model(example_inputs)

        # Infer signature for the model
        signature = infer_signature(
            example_inputs.cpu().numpy(), example_outputs.cpu().numpy()
        )

        # Log PyTorch model
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            signature=signature,
            input_example=example_inputs.cpu().numpy(),
        )

        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)

        # Log ONNX model
        mlflow.onnx.log_model(
            onnx_model=onnx_model,
            artifact_path="onnx_model",
            signature=signature,
            input_example=example_inputs.cpu().numpy(),
        )

        print("Models successfully logged to MLflow")

        # End the run if we started one
        if run_id:
            mlflow.end_run()

    except Exception as e:
        print(f"Error logging models to MLflow: {e}")
        import traceback

        traceback.print_exc()
