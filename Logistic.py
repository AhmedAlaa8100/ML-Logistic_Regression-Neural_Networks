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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ========================== MODEL CLASS ==========================
class LogisticRegressionScratch:
    def __init__(self, input_dim=784, lr=0.01):
        self.W = torch.zeros((input_dim, 1), dtype=torch.float32)
        self.b = torch.zeros(1, dtype=torch.float32)
        self.lr = lr

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

    def gradient(self, X, y_pred, y_true):
        m = y_true.shape[0]
        dw = torch.matmul(X.T, (y_pred - y_true)) 
        db = torch.sum(y_pred - y_true) 
        return dw, db

    def update_weights(self, dw, db):
        self.W -= self.lr * dw
        self.b -= self.lr * db



    def train(self, train_loader, val_loader, epochs=15, patience=5):
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        best_val_loss = float('inf')
        best_weights = (self.W.clone(), self.b.clone())
        patience_counter = 0

        for epoch in range(epochs):
            # TRAIN
            epoch_train_loss = 0
            correct, total = 0, 0
            for X_batch, y_batch in train_loader:
                # Reshape y_batch to be of shape (batch_size, 1)
                y_batch = y_batch.view(-1, 1).float()
                #calculate predictions
                y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                #calculate loss
                loss = self.loss(y_pred, y_batch)
                epoch_train_loss += loss.item()
                #calculate gradients
                dw, db = self.gradient(X_batch, y_pred, y_batch)
                self.update_weights(dw, db)
                #calculate accuracy for the batch
                correct += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
                total += y_batch.size(0)

            #calculate average loss and accuracy for the epoch
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
            #patience variable to avoid overfitting
            #if validation loss does not improve for 'patience' epochs, stop training
            if val_loss  < best_val_loss - 1e-4:
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
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        print("\nEvaluate Logistic Regression:")
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
    #defined as the epoch when validation loss stabilizes (less than 0.1% change)
    best_loss = val_losses[0]
    for i in range(1, len(val_losses)):
        if abs(val_losses[i] - best_loss) / best_loss < 0.001:
            return i
        best_loss = min(best_loss, val_losses[i])
    return len(val_losses)

def stability_measure(val_losses):
    #standard deviation of the last 5 validation losses
    tail = val_losses[-5:] if len(val_losses) >= 5 else val_losses
    return np.std(tail)

def gradient_noise_measure(train_losses):
    #variance of the differences between consecutive training losses
    diffs = np.diff(train_losses)
    return np.var(diffs)

def run_learning_rate_analysis():
    train_loader, val_loader, _ = load_data('mnist_All.csv', batch_size=64)

    results = {}
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    print("\n===== Learning Rate Analysis =====")

    for lr in learning_rates:
        print(f"\nTesting Learning Rate = {lr}")
        model = LogisticRegressionScratch(input_dim=784, lr=lr)
        train_losses, val_losses, train_acc, val_acc = model.train(train_loader, val_loader, epochs=30, patience=5)
        #calculate metrics
        conv_speed = convergence_speed(val_losses)
        stability = stability_measure(val_losses)
        #store results
        results[lr] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'convergence_speed': conv_speed,
            'stability': stability
        }


    # Best learning rate based on maximum validation accuracy
    best_lr = max(results, key=lambda x: max(results[x]['val_acc']))
    best_lr_acc = max(results[best_lr]['val_acc'])
    print(f"\n Best Learning Rate: {best_lr} (Val Accuracy: {best_lr_acc:.4f})")

    return results, best_lr, best_lr_acc

def run_batch_size_analysis():
    results = {}
    batch_sizes = [16, 32, 64, 128]
    print("\n===== Batch Size Analysis =====")

    for bs in batch_sizes:
        print(f"\nTesting Batch Size = {bs}")
        train_loader, val_loader, test_loader = load_data('mnist_All.csv', batch_size=bs)

        model = LogisticRegressionScratch(input_dim=784, lr=0.01)
  

        train_losses, val_losses,  train_acc, val_acc = model.train( train_loader, val_loader, epochs=30, patience=5)

        grad_noise = gradient_noise_measure(train_losses)

        results[bs] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gradient_noise': grad_noise
        }


    best_bs = max(results, key=lambda x: max(results[x]['val_acc']))
    best_bs_acc = max(results[best_bs]['val_acc'])
    print(f"\n Best Batch Size: {best_bs} (Val Accuracy: {best_bs_acc:.4f})")

    return results, best_bs, best_bs_acc
def plot_learning_rate_analysis(results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    plt.suptitle('Learning Rate Analysis', fontsize=14, fontweight='bold')

    # (1) Train Loss
    plt.subplot(2, 2, 1)
    for lr, data in results.items():
        plt.plot(data['train_losses'], label=f'LR={lr}')
    plt.xlabel('Epochs'); plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (2) Validation Loss
    plt.subplot(2, 2, 2)
    for lr, data in results.items():
        plt.plot(data['val_losses'], label=f'LR={lr} (conv={data["convergence_speed"]}, stab={data["stability"]:.4f})')
    plt.xlabel('Epochs'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (3) Train Accuracy
    plt.subplot(2, 2, 3)
    for lr, data in results.items():
        plt.plot(data['train_acc'], label=f'LR={lr}')
    plt.xlabel('Epochs'); plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (4) Validation Accuracy
    plt.subplot(2, 2, 4)
    for lr, data in results.items():
        plt.plot(data['val_acc'], label=f'LR={lr}')
    plt.xlabel('Epochs'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_batch_size_analysis(results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    plt.suptitle('Batch Size Analysis', fontsize=14, fontweight='bold')

    # (1) Train Loss
    plt.subplot(2, 2, 1)
    for bs, data in results.items():
        plt.plot(data['train_losses'], label=f'BS={bs}')
    plt.xlabel('Epochs'); plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (2) Validation Loss
    plt.subplot(2, 2, 2)
    for bs, data in results.items():
        plt.plot(data['val_losses'], label=f'BS={bs} (grad_noise={data["gradient_noise"]:.4f})')
    plt.xlabel('Epochs'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (3) Train Accuracy
    plt.subplot(2, 2, 3)
    for bs, data in results.items():
        plt.plot(data['train_acc'], label=f'BS={bs}')
    plt.xlabel('Epochs'); plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    # (4) Validation Accuracy
    plt.subplot(2, 2, 4)
    for bs, data in results.items():
        plt.plot(data['val_acc'], label=f'BS={bs}')
    plt.xlabel('Epochs'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend(fontsize=8); plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def analyze_results(lr_results, bs_results):
    print("\n" + "="*60)
    print(" DETAILED ANALYSIS: LEARNING RATE & BATCH SIZE")
    print("="*60)

    print("\n LEARNING RATE ANALYSIS SUMMARY:")
    print(f"{'LR':<10} {'Best Val Acc':<15} {'Conv Speed':<15} {'Stability':<15}")
    print("-"*60)
    for lr, data in lr_results.items():
        best_val_acc = max(data['val_acc'])
        conv_speed = data['convergence_speed']
        stability = data['stability']
        print(f"{lr:<10} {best_val_acc:<15.4f} {conv_speed:<15} {stability:<15.6f}")



    print("\n BATCH SIZE ANALYSIS SUMMARY:")
    print(f"{'Batch Size':<15} {'Best Val Acc':<15} {'Grad Noise':<15}")
    print("-"*50)
    for bs, data in bs_results.items():
        best_val_acc = max(data['val_acc'])
        grad_noise = data['gradient_noise']
        print(f"{bs:<15} {best_val_acc:<15.4f} {grad_noise:<15.6f}")



