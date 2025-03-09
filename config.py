from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Data parameters
    batch_size: int = 32
    image_size: tuple = (28, 28)
    train_split: float = 0.8

    # Model parameters
    conv_channels: list = (32, 64, 128, 256)
    fc_features: list = (512, 256, 128)
    dropout_rates_conv: list = (0.1, 0.15, 0.2, 0.25)
    dropout_rates_fc: list = (0.4, 0.3, 0.2)

    # Training parameters
    learning_rate: float = 5e-4
    weight_decay: float = 2e-5
    epochs: int = 1
    scheduler_factor: float = 0.7
    scheduler_patience: int = 3

    # Class weight parameters
    class_weight_power: float = 0.3

    # Paths
    model_save_path: str = "skin_lesion_model.pth"
    csv_path: str = "data/HAM10000_metadata.csv"
    images_dir: str = "data/images"

    # MLflow parameters
    experiment_name: str = "skin-lesion-classification"
    run_name: str = "cnn-model"

    def __post_init__(self):
        pass
