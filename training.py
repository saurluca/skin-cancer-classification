import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import mlflow


def train(dataloader, model, loss_fn, optimizer, epoch, device, scheduler=None):
    """Train the model for one epoch."""
    size = len(dataloader.dataset)
    model.train()
    running_loss = 0.0
    correct = 0

    # Create progress bar for training batches
    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1} [Train]",
        leave=False,
        unit="batch",
    )

    for X, y in progress_bar:
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Update progress bar with current loss and accuracy
        batch_loss = loss.item()
        batch_acc = (pred.argmax(1) == y).type(torch.float).mean().item()
        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}", accuracy=f"{100 * batch_acc:.2f}%"
        )

    # Calculate epoch statistics
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / size
    print(
        f"Epoch {epoch + 1}: Avg loss: {epoch_loss:.4f}, Accuracy: {100 * epoch_acc:.2f}%"
    )

    # Update the learning rate scheduler if provided
    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, epoch_acc


def test(
    dataloader,
    model,
    loss_fn,
    epoch=None,
    device=None,
    idx_to_diagnosis=None,
    diagnosis_to_idx=None,
):
    """Evaluate the model on the test set."""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_preds = []
    all_labels = []

    # Create progress bar for test batches
    progress_bar = tqdm(dataloader, desc="Evaluation [Test]", leave=False, unit="batch")

    with torch.no_grad():
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss
            batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct

            # Store predictions and labels for detailed metrics
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            # Update progress bar
            batch_acc = batch_correct / len(y)
            progress_bar.set_postfix(
                loss=f"{batch_loss:.4f}", accuracy=f"{100 * batch_acc:.2f}%"
            )

    test_loss /= num_batches
    accuracy = correct / size
    print(
        f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    # Print detailed classification report
    if idx_to_diagnosis and diagnosis_to_idx:
        print("Classification Report:")
        target_names = [idx_to_diagnosis[i] for i in range(len(diagnosis_to_idx))]
        report = classification_report(
            all_labels, all_preds, target_names=target_names, output_dict=True
        )
        print(classification_report(all_labels, all_preds, target_names=target_names))

        # Log metrics to MLflow if epoch is provided
        if epoch is not None:
            mlflow.log_metrics(
                {"test_loss": test_loss, "test_accuracy": accuracy}, step=epoch
            )

            # Log class-specific metrics
            for class_name in target_names:
                if class_name in report:
                    mlflow.log_metrics(
                        {
                            f"{class_name}_precision": report[class_name]["precision"],
                            f"{class_name}_recall": report[class_name]["recall"],
                            f"{class_name}_f1-score": report[class_name]["f1-score"],
                        },
                        step=epoch,
                    )

    return test_loss, accuracy


def train_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    loss_fn,
    epochs,
    device,
    idx_to_diagnosis,
    diagnosis_to_idx,
):
    """Train and evaluate the model for multiple epochs."""
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    # Track epochs where test evaluation was performed
    test_epochs = []

    # Create progress bar for epochs
    epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

    for t in epoch_bar:
        # Train for one epoch
        epoch_loss, epoch_acc = train(
            train_loader, model, loss_fn, optimizer, t, device
        )
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # Log metrics to MLflow
        mlflow.log_metrics(
            {"train_loss": epoch_loss, "train_accuracy": epoch_acc}, step=t
        )

        # Only evaluate on test set every 3 epochs or on the final epoch
        if (t + 1) % 3 == 0 or t == epochs - 1:
            test_loss, test_acc = test(
                test_loader,
                model,
                loss_fn,
                epoch=t,
                device=device,
                idx_to_diagnosis=idx_to_diagnosis,
                diagnosis_to_idx=diagnosis_to_idx,
            )
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            test_epochs.append(t)

            # Update epoch progress bar with test metrics
            epoch_bar.set_postfix(
                train_loss=f"{epoch_loss:.4f}",
                train_acc=f"{100 * epoch_acc:.2f}%",
                test_loss=f"{test_loss:.4f}",
                test_acc=f"{100 * test_acc:.2f}%",
            )
        else:
            # Update epoch progress bar with only training metrics
            epoch_bar.set_postfix(
                train_loss=f"{epoch_loss:.4f}",
                train_acc=f"{100 * epoch_acc:.2f}%",
                test="N/A",
            )

        # Log current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate", current_lr, step=t)

    return train_losses, train_accs, test_losses, test_accs, test_epochs


def train_and_compare_models(
    model1,
    model2,
    train_loader,
    test_loader,
    optimizer1,
    optimizer2,
    scheduler1,
    scheduler2,
    loss_fn,
    epochs,
    device,
    idx_to_diagnosis,
    diagnosis_to_idx,
):
    """Train and compare two models."""
    # Dictionary to store results
    results = {
        "custom_cnn": {
            "train_losses": [],
            "train_accs": [],
            "test_losses": [],
            "test_accs": [],
            "test_epochs": [],
        },
        "pretrained": {
            "train_losses": [],
            "train_accs": [],
            "test_losses": [],
            "test_accs": [],
            "test_epochs": [],
        },
    }

    print("\n" + "=" * 50)
    print("Starting training of both models...")
    print("=" * 50 + "\n")

    # Train both models for the specified number of epochs
    for t in range(epochs):
        print(f"\nEpoch {t + 1}/{epochs}")
        print("-" * 30)

        # Train and evaluate custom CNN
        print("Training custom CNN model:")
        epoch_loss, epoch_acc = train(
            train_loader, model1, loss_fn, optimizer1, t, device
        )
        results["custom_cnn"]["train_losses"].append(epoch_loss)
        results["custom_cnn"]["train_accs"].append(epoch_acc)

        # Train and evaluate pre-trained model
        print("\nTraining pre-trained model:")
        epoch_loss, epoch_acc = train(
            train_loader, model2, loss_fn, optimizer2, t, device
        )
        results["pretrained"]["train_losses"].append(epoch_loss)
        results["pretrained"]["train_accs"].append(epoch_acc)

        # Evaluate both models on test set every 3 epochs or on the final epoch
        if (t + 1) % 3 == 0 or t == epochs - 1:
            print("\nEvaluating custom CNN model:")
            test_loss, test_acc = test(
                test_loader,
                model1,
                loss_fn,
                epoch=t,
                device=device,
                idx_to_diagnosis=idx_to_diagnosis,
                diagnosis_to_idx=diagnosis_to_idx,
            )
            results["custom_cnn"]["test_losses"].append(test_loss)
            results["custom_cnn"]["test_accs"].append(test_acc)
            results["custom_cnn"]["test_epochs"].append(t)

            print("\nEvaluating pre-trained model:")
            test_loss, test_acc = test(
                test_loader,
                model2,
                loss_fn,
                epoch=t,
                device=device,
                idx_to_diagnosis=idx_to_diagnosis,
                diagnosis_to_idx=diagnosis_to_idx,
            )
            results["pretrained"]["test_losses"].append(test_loss)
            results["pretrained"]["test_accs"].append(test_acc)
            results["pretrained"]["test_epochs"].append(t)

        # Update schedulers
        scheduler1.step(results["custom_cnn"]["train_losses"][-1])
        scheduler2.step(results["pretrained"]["train_losses"][-1])

    return results
