import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class CNNModel(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        # Build convolutional layers dynamically
        conv_layers = []
        in_channels = 3  # RGB images

        for i, out_channels in enumerate(config.conv_channels):
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout2d(config.dropout_rates_conv[i]),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate the size of flattened features after convolutions
        # For 28x28 input with 4 MaxPool2d layers: 28 -> 14 -> 7 -> 3 -> 1
        x = config.image_size[0]
        y = config.image_size[1]
        for _ in range(len(config.conv_channels)):
            x = x // 2
            y = y // 2

        conv_output_size = config.conv_channels[-1] * max(x, 1) * max(y, 1)

        # Build fully connected layers dynamically
        fc_layers = []
        in_features = conv_output_size

        for i, out_features in enumerate(config.fc_features):
            fc_layers.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                    nn.Dropout(config.dropout_rates_fc[i]),
                ]
            )
            in_features = out_features

        # Add final classification layer
        fc_layers.append(nn.Linear(config.fc_features[-1], num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x):
        # Pass through convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc_layers(x)

        return x


class PretrainedModel(nn.Module):
    def __init__(self, num_classes, pretrained_model="resnet18"):
        super().__init__()

        # Load pre-trained model
        if pretrained_model == "resnet18":
            self.base_model = models.resnet18(weights="IMAGENET1K_V1")
        elif pretrained_model == "resnet50":
            self.base_model = models.resnet50(weights="IMAGENET1K_V1")
        elif pretrained_model == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        else:
            raise ValueError(f"Unsupported model: {pretrained_model}")

        # Freeze early layers to prevent overfitting
        for param in list(self.base_model.parameters())[
            :-20
        ]:  # Freeze all but last few layers
            param.requires_grad = False

        # Replace the final fully connected layer
        if pretrained_model.startswith("resnet"):
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes)
            )
        elif pretrained_model.startswith("efficientnet"):
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes)
            )

    def forward(self, x):
        return self.base_model(x)


def setup_models(config, num_classes, device):
    """Set up models, optimizers, and scheduler."""
    # Create the custom CNN model
    model = CNNModel(config, num_classes).to(device)
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
    pretrained_model = PretrainedModel(num_classes, pretrained_model="resnet18").to(
        device
    )
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

    return {
        "model": model,
        "pretrained_model": pretrained_model,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "pretrained_optimizer": pretrained_optimizer,
        "scheduler": scheduler,
        "pretrained_scheduler": pretrained_scheduler,
    }
