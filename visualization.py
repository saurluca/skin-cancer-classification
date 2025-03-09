import matplotlib.pyplot as plt


def plot_training_progress(
    train_losses, train_accs, test_losses=None, test_accs=None, test_epochs=None
):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    if test_losses and test_epochs:
        plt.plot(test_epochs, test_losses, "o-", label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label="Training Accuracy")
    if test_accs and test_epochs:
        plt.plot(test_epochs, test_accs, "o-", label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Save figure for MLflow logging
    plt_path = "training_progress.png"
    plt.savefig(plt_path)
    plt.show()

    return plt_path


def plot_model_comparison(results):
    """Plot comparison of multiple models' performances."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot training losses
    axes[0, 0].plot(results["custom_cnn"]["train_losses"], label="Custom CNN")
    axes[0, 0].plot(results["pretrained"]["train_losses"], label="Pre-trained")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epochs")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    # Plot training accuracies
    axes[0, 1].plot(results["custom_cnn"]["train_accs"], label="Custom CNN")
    axes[0, 1].plot(results["pretrained"]["train_accs"], label="Pre-trained")
    axes[0, 1].set_title("Training Accuracy")
    axes[0, 1].set_xlabel("Epochs")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # Plot test losses
    axes[1, 0].plot(
        results["custom_cnn"]["test_epochs"],
        results["custom_cnn"]["test_losses"],
        "o-",
        label="Custom CNN",
    )
    axes[1, 0].plot(
        results["pretrained"]["test_epochs"],
        results["pretrained"]["test_losses"],
        "o-",
        label="Pre-trained",
    )
    axes[1, 0].set_title("Test Loss")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()

    # Plot test accuracies
    axes[1, 1].plot(
        results["custom_cnn"]["test_epochs"],
        results["custom_cnn"]["test_accs"],
        "o-",
        label="Custom CNN",
    )
    axes[1, 1].plot(
        results["pretrained"]["test_epochs"],
        results["pretrained"]["test_accs"],
        "o-",
        label="Pre-trained",
    )
    axes[1, 1].set_title("Test Accuracy")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()

    plt.tight_layout()
    plt_path = "model_comparison.png"
    plt.savefig(plt_path)
    plt.show()

    return plt_path
