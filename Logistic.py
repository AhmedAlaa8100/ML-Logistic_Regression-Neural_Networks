import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ========================== DATA LOADER ==========================
def load_data(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    Y = data['label'].values
    X = data.drop(columns=['label'], axis=1).values
    X = X / 255.0  # Normalize features

    mask = (Y == 0) | (Y == 1)  # Binary classification mask
    X_binary = X[mask]
    Y_binary = Y[mask]

    # Split the data 60-20-20
    X_train_binary, X_temp_binary, Y_train_binary, Y_temp_binary = train_test_split(
        X_binary, Y_binary, test_size=0.4, stratify=Y_binary, random_state=30
    )
    X_val_binary, X_test_binary, Y_val_binary, Y_test_binary = train_test_split(
        X_temp_binary, Y_temp_binary, test_size=0.5, stratify=Y_temp_binary, random_state=30
    )

    train_dataset = TensorDataset(torch.tensor(X_train_binary, dtype=torch.float32),
                                  torch.tensor(Y_train_binary, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val_binary, dtype=torch.float32),
                                torch.tensor(Y_val_binary, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_binary, dtype=torch.float32),
                                 torch.tensor(Y_test_binary, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ========================== MODEL CLASS ==========================
class LogisticRegressionScratch:
    def __init__(self, input_dim=784, lr=0.01):
        self.W = torch.zeros((input_dim, 1), dtype=torch.float32 )
        self.b = torch.zeros(1, dtype=torch.float32)
        self.lr = lr

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

    def gradient(self, X, y_pred, y_true):
        m = X.shape[0]
        dw = torch.matmul(X.T, (y_pred - y_true)) / m
        db = torch.sum(y_pred - y_true) / m
        return dw, db

    def update_weights(self, dw, db):
        self.W -= self.lr * dw
        self.b -= self.lr * db

    def accuracy(self, y_pred, y_true):
        preds = (y_pred >= 0.5).float()
        correct = (preds == y_true).float().sum()
        return correct / y_true.shape[0]

    def train(self, train_loader, val_loader, epochs=15, patience=3):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_loss = float('inf')
        best_weights = (self.W.clone(), self.b.clone())
        patience_counter = 0

        for epoch in range(epochs):
            # TRAIN
            epoch_train_loss = 0
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                y_batch = y_batch.view(-1, 1).float()
                y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                loss = self.loss(y_pred, y_batch)
                epoch_train_loss += loss.item()

                dw, db = self.gradient(X_batch, y_pred, y_batch)
                self.update_weights(dw, db)

                correct += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
                total += y_batch.size(0)

            train_loss = epoch_train_loss / len(train_loader)
            train_acc = correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # VALIDATION
            epoch_val_loss = 0
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    y_batch = y_batch.view(-1, 1).float()
                    y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                    loss = self.loss(y_pred, y_batch)
                    epoch_val_loss += loss.item()
                    correct += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
                    total += y_batch.size(0)
            val_loss = epoch_val_loss / len(val_loader)
            val_acc = correct / total
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            # EARLY STOPPING
            if val_loss + 1e-10 < best_val_loss:
                best_val_loss = val_loss
                best_weights = (self.W.clone(), self.b.clone())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}.")
                    break

        self.W, self.b = best_weights

        #self.plot_curves(train_losses, val_losses, train_accs, val_accs)
        return train_losses, val_losses, train_accs, val_accs

    def plot_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(12, 5))
        plt.suptitle("Training Progress of Logistic Regression", fontsize=14)
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Val Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy', color='blue')
        plt.plot(val_accs, label='Val Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curves')
        plt.show()

    def test(self, test_loader):
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_batch = y_batch.view(-1, 1).float()
                y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                loss = self.loss(y_pred, y_batch)
                total_loss += loss.item()

                preds = (y_pred >= 0.5).float()
                correct += (preds == y_batch).float().sum().item()
                total += y_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        print(f"\nFinal Test Loss: {avg_loss:.4f}")
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
        return avg_loss, accuracy, cm


# ========================== METRICS ==========================
def convergence_speed(val_losses):
    best_loss = val_losses[0]
    for i in range(1, len(val_losses)):
        if abs(val_losses[i] - best_loss) / best_loss < 0.001:
            return i
        best_loss = min(best_loss, val_losses[i])
    return len(val_losses)

def stability_measure(val_losses):
    tail = val_losses[-5:] if len(val_losses) >= 5 else val_losses
    return np.std(tail)

def gradient_noise_measure(train_losses):
    diffs = np.diff(train_losses)
    return np.var(diffs)


# ========================== MAIN ANALYSIS ==========================
def run_analysis(file_path):
    
    lr_values = [0.001, 0.01, 0.1, 1.0]
    batch_sizes = [16, 32, 64, 128]

    lr_results, bs_results = [], []
    train_loader, val_loader, _ = load_data(file_path, batch_size=64)

    # ---- LEARNING RATE ANALYSIS ----
    val_losses = []
    for lr in lr_values:
        
        model = LogisticRegressionScratch(784,lr)
        train_loss, val_loss, _, _ = model.train(train_loader, val_loader)
        conv_speed = convergence_speed(val_loss)
        stab = stability_measure(val_loss)
        lr_results.append((lr, val_loss[-1], conv_speed, stab))
        val_losses.append(val_loss)
        
    plt.figure(figsize=(8, 5))
    for i, lr in enumerate(lr_values):
        plt.plot(val_losses[i], label=f"LR={lr}")

    plt.title("Learning Rate Effect on Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend(title="Learning Rates")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\n=== Learning Rate Quantitative Results ===")
    print("LR\tFinalLoss\tConvergeEpoch\tStability(Std)")
    for lr, loss, conv, stab in lr_results:
        print(f"{lr}\t{loss:.5f}\t{conv}\t\t{stab:.6f}")

    # ---- BATCH SIZE ANALYSIS ----
    val_losses = []
    for bs in batch_sizes:
        train_loader, val_loader, _ = load_data(file_path, batch_size=bs)
        model = LogisticRegressionScratch(784, lr=0.1)
        train_loss, val_loss, _, _ = model.train(train_loader, val_loader)
        noise = gradient_noise_measure(train_loss)
        bs_results.append((bs, val_loss[-1], noise))
        val_losses.append(val_loss)
    plt.figure(figsize=(8, 5))
    for i, bs in enumerate(batch_sizes):
        plt.plot(val_losses[i], label=f"Batch={bs}")

    plt.title("Batch Size Effect on Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend(title="Batch Sizes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n=== Batch Size Quantitative Results ===")
    print("BatchSize\tFinalLoss\tGradientNoise(Var)")
    for bs, loss, noise in bs_results:
        print(f"{bs}\t\t{loss:.5f}\t\t{noise:.6f}")


# ========================== ENTRY POINT ==========================
if __name__ == "__main__":

   run_analysis("mnist_All.csv")
