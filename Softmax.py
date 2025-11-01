import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


# ==============================================================
# 1. Data Loading (you already implemented it)
# ==============================================================

def load_data(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    Y = data['label'].values
    X = data.drop(columns=['label'], axis=1).values
    X = X / 255.0  # Normalize features

    # Split the data 60-20-20 for train, validation, and test
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=30)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=30)

    # Convert numpy arrays to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(Y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.long))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ==============================================================
# 2. Model Functions (from scratch)
# ==============================================================

def softmax(z):
    exp_z = torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])  # numerical stability
    return exp_z / torch.sum(exp_z, dim=1, keepdim=True)


def cross_entropy_loss(y_pred, y_true):
    # y_true is class indices (0–9)
    batch_size = y_pred.shape[0]
    log_probs = -torch.log(y_pred[range(batch_size), y_true] + 1e-9)
    return torch.mean(log_probs)


def accuracy(y_pred, y_true):
    preds = torch.argmax(y_pred, dim=1)
    return (preds == y_true).float().mean().item()


# ==============================================================
# 3. Training and Evaluation Functions
# ==============================================================

def train_softmax(train_loader, val_loader, input_dim, num_classes=10, lr=0.01, epochs=30):
    # Initialize parameters
    W = torch.randn(input_dim, num_classes, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(num_classes, dtype=torch.float32, requires_grad=True)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0

        for X_batch, y_batch in train_loader:
            # Forward
            logits = X_batch @ W + b
            y_pred = softmax(logits)

            # Compute loss
            loss = cross_entropy_loss(y_pred, y_batch)

            # Backward
            loss.backward()

            # Update parameters (SGD)
            with torch.no_grad():
                W -= lr * W.grad
                b -= lr * b.grad

            # Zero gradients
            W.grad.zero_()
            b.grad.zero_()

            # Track accuracy and loss
            total_loss += loss.item()
            correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
            total += y_batch.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # Validation
        val_loss, val_acc = evaluate_softmax(W, b, val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    return W, b, train_losses, val_losses, train_accs, val_accs


def evaluate_softmax(W, b, loader):
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = X_batch @ W + b
            y_pred = softmax(logits)
            loss = cross_entropy_loss(y_pred, y_batch)

            total_loss += loss.item()
            preds = y_pred.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    avg_loss = total_loss / len(loader)
    acc = correct / total
    return avg_loss, acc


# ==============================================================
# 4. Plotting and Analysis
# ==============================================================

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# ==============================================================
# 5. Main Script
# ==============================================================

if __name__ == "__main__":
    file_path = "mnist_All.csv"  # update with your MNIST CSV path
    train_loader, val_loader, test_loader = load_data(file_path)

    input_dim = 784  # 28x28 pixels
    num_classes = 10

    # Train model
    W, b, train_losses, val_losses, train_accs, val_accs = train_softmax(
        train_loader, val_loader, input_dim, num_classes, lr=0.01, epochs=30
    )

    # Plot metrics
    plot_metrics(train_losses, val_losses, train_accs, val_accs)

    # Test evaluation
    total_loss, total_acc = evaluate_softmax(W, b, test_loader)
    print(f"\nTest Accuracy: {total_acc:.4f}")

    # Confusion matrix & per-class accuracy
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = X_batch @ W + b
            y_pred = softmax(logits)
            preds = y_pred.argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y_batch.tolist())

    plot_confusion_matrix(all_labels, all_preds)
    print("\nPer-Class Accuracy and Report:\n")
    print(classification_report(all_labels, all_preds))
