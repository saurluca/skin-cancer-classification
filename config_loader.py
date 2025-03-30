import yaml
from types import SimpleNamespace


def load_config(config_path="config.yaml"):
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A SimpleNamespace object with configuration values accessible as attributes.
    """
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)

    # Convert dictionary to object with attribute access
    config = SimpleNamespace(**config_dict)

    # Ensure tuple type for specific parameters that need it
    config.image_size = tuple(config.image_size)
    config.conv_channels = tuple(config.conv_channels)
    config.fc_features = tuple(config.fc_features)
    config.dropout_rates_conv = tuple(config.dropout_rates_conv)
    config.dropout_rates_fc = tuple(config.dropout_rates_fc)

    return config