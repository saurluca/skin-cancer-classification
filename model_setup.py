import torch.nn as nn
import torch.optim as optim
from models import CNNModel, PretrainedModel


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
