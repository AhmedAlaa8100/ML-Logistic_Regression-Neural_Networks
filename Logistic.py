import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    Y = data['label'].values
    X = data.drop(columns=['label'], axis=1).values
    X = X/255.0  # Normalize features
    mask= (Y==0) | (Y==1)  # Binary classification mask
    X_binary=X[mask]
    Y_binary=Y[mask]

    # Split the data 60-20-20 for train, validation, and test
    X_train_binary, X_temp_binary, Y_train_binary, Y_temp_binary = train_test_split(X_binary, Y_binary, test_size=0.4, stratify=Y_binary, random_state=30)
    X_val_binary, X_test_binary, Y_val_binary, Y_test_binary = train_test_split(X_temp_binary, Y_temp_binary, test_size=0.5, stratify=Y_temp_binary, random_state=30)

    

    # Create TensorDatasets for each split
    # Convert data (numpy arrays) to PyTorch tensors
    train_dataset_binary = TensorDataset(torch.tensor(X_train_binary, dtype=torch.float32),
                                  torch.tensor(Y_train_binary, dtype=torch.long))
    val_dataset_binary = TensorDataset(torch.tensor(X_val_binary, dtype=torch.float32),
                                torch.tensor(Y_val_binary, dtype=torch.long))
    test_dataset_binary = TensorDataset(torch.tensor(X_test_binary, dtype=torch.float32),
                                 torch.tensor(Y_test_binary, dtype=torch.long))

    # Data loaders for batching and shuffling the data
    train_loader_binary = DataLoader(train_dataset_binary, batch_size=batch_size, shuffle=True)
    val_loader_binary = DataLoader(val_dataset_binary, batch_size=batch_size, shuffle=False)
    test_loader_binary = DataLoader(test_dataset_binary, batch_size=batch_size, shuffle=False)

    return train_loader_binary, val_loader_binary, test_loader_binary


class LogisticRegressionScratch:
    def __init__(self, input_dim, lr=0.01):
        self.W = torch.zeros((input_dim, 1), dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.lr = lr

    # Sigmoid activation
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    
    # Binary cross-entropy loss
    def loss(self, y_pred, y_true):
        eps = 1e-9
        return -torch.mean(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps))

    # Gradient computation
    def gradient(self, X, y_pred, y_true):
        m = X.shape[0]
        dw = torch.matmul(X.T, (y_pred - y_true)) / m
        db = torch.sum(y_pred - y_true) / m
        return dw, db

    # Update weights
    def update_weights(self, dw, db):
        self.W -= self.lr * dw
        self.b -= self.lr * db

    # Accuracy
    def accuracy(self, y_pred, y_true):
        preds = (y_pred >= 0.5).float()
        correct = (preds == y_true).float().sum()
        return correct / y_true.shape[0]

    # Training and validation
    def train(self, train_loader, val_loader, epochs=10):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            # ----- TRAIN -----
            self.W.requires_grad = False
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

            # ----- VALIDATE -----
            val_loss, val_acc = self.validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # Plot curves
        self.plot_curves(train_losses, val_losses, train_accs, val_accs)
        return train_losses, val_losses, train_accs, val_accs

    # Validation
    def validate(self, val_loader):
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_batch = y_batch.view(-1, 1).float()
                y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                loss = self.loss(y_pred, y_batch)
                total_loss += loss.item()
                correct += ((y_pred >= 0.5).float() == y_batch).float().sum().item()
                total += y_batch.size(0)
        return total_loss / len(val_loader), correct / total

    # Plot loss and accuracy
    def plot_curves(self, train_losses, val_losses, train_accs, val_accs):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')

        plt.show()

    # Testing
    def test(self, test_loader):
        total_loss = 0.0
        correct = 0
        total = 0
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
        plt.title('Confusion Matrix (Test Data)')
        plt.show()

        print(f"\nFinal Test Loss: {avg_loss:.4f}")
        print(f"Final Test Accuracy: {accuracy*100:.2f}%")
        return avg_loss, accuracy, cm
if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = load_data('mnist_All.csv', batch_size=64)

    # Initialize model
    input_dim = 28 * 28  # MNIST images are 28x28
    model = LogisticRegressionScratch(input_dim=input_dim, lr=0.1)

    # Train model
    model.train(train_loader, val_loader, epochs=50)

    # Test model
    model.test(test_loader)
    


