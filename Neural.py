import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_data(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    Y = data['label'].values
    X = data.drop(columns=['label'], axis=1).values
    X = X/255.0  # Normalize features
    # Split the data 60-20-20 for train, validation, and test
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=30)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=30)

    # Create TensorDatasets for each split
    # Convert data (numpy arrays) to PyTorch tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(Y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                torch.tensor(Y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(Y_test, dtype=torch.long))

    # Data loaders for batching and shuffling the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=[128, 64], layers=2, output_size=10):
        super(NeuralNetwork, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU()
        self.best_val_loss = float('inf')
        self.best_model_state = None

        # Create hidden layers dynamically
        for i in range(self.layers):
            if i == 0:
                setattr(self, f'fc{i+1}', nn.Linear(input_size, hidden_size[i]))
            else:
                setattr(self, f'fc{i+1}', nn.Linear(hidden_size[i-1], hidden_size[i]))

        # Output layer
        setattr(self, f'fc{self.layers + 1}', nn.Linear(hidden_size[-1], output_size))

        # Weight initialization (He)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(self.layers):
            fc_layer = getattr(self, f'fc{i+1}')
            x = self.relu(fc_layer(x))
        output_layer = getattr(self, f'fc{self.layers + 1}')
        return output_layer(x)

    # ---------------- TRAINING -----------------
    def train_model(self, train_loader, val_loader, loss_function, optimizer, epochs=10, patience=3):
        train_losses, val_losses = [], []
        train_std, val_std = [], []
        train_accuracies, val_accuracies = [], []
        self.early_stopping_counter = 0
        previous_loss = None

        for epoch in range(epochs):
            # ---- TRAIN ----
            self.train()
            batch_losses = []
            correct_train, total_train = 0, 0

            for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Training]'):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = torch.tensor(batch_losses).mean().item()
            std_train_loss = torch.tensor(batch_losses).std().item()
            train_losses.append(avg_train_loss)
            train_std.append(std_train_loss)
            train_accuracies.append(correct_train / total_train)

            # ---- VALIDATION ----
            self.eval()
            val_batch_losses = []
            correct_val, total_val = 0, 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Validation]'):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    loss = loss_function(outputs, labels)
                    val_batch_losses.append(loss.item())

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            avg_val_loss = torch.tensor(val_batch_losses).mean().item()
            std_val_loss = torch.tensor(val_batch_losses).std().item()
            val_losses.append(avg_val_loss)
            val_std.append(std_val_loss)
            val_accuracies.append(correct_val / total_val)

            # ---- CONVERGENCE ----
            if previous_loss is not None:
                delta = abs(avg_train_loss - previous_loss)
                print(f"Δ Convergence: {delta:.6f}")
            previous_loss = avg_train_loss

            # ---- LOGGING ----
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f} | "
                  f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

            # ---- EARLY STOPPING ----
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.state_dict()
                torch.save(self.best_model_state, 'best_model.pth')
                self.early_stopping_counter = 0
                print("New best model saved.")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= patience:
                    print("No improvement, stopping early.")
                    break

        return self.best_model_state, train_losses, val_losses, train_std, val_std, train_accuracies, val_accuracies

    # ---------------- EVALUATION -----------------
    def evaluate_model(self, test_loader, loss_function):
        self.eval()
        test_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        print(f'\n Test Loss: {avg_test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        return avg_test_loss, test_accuracy

    # ---------------- VISUALIZATION -----------------
    @staticmethod
    def plot_results(train_losses, val_losses, train_std, val_std, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)
        loss_gap = [abs(t - v) for t, v in zip(train_losses, val_losses)]  # convergence metric

        plt.figure(figsize=(15, 4))

        # (1) Loss curves with error bars
        plt.subplot(1, 3, 1)
        plt.errorbar(epochs, train_losses, yerr=train_std, label='Train Loss', capsize=3)
        plt.errorbar(epochs, val_losses, yerr=val_std, label='Validation Loss', capsize=3)
        plt.xlabel('Epochs'); plt.ylabel('Loss')
        plt.title('Learning Curve (with Error Bars)')
        plt.legend(); plt.grid(True)

        # (2) Accuracy curves
        plt.subplot(1, 3, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs'); plt.ylabel('Accuracy')
        plt.title('Learning Curve (Accuracy)')
        plt.legend(); plt.grid(True)

        # (3) Convergence analysis
        plt.subplot(1, 3, 3)
        plt.plot(epochs, loss_gap, color='purple', label='|Train - Val Loss|')
        plt.xlabel('Epochs'); plt.ylabel('Loss Gap')
        plt.title('Convergence Analysis')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.show()

# Example usage:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_data('mnist_All.csv', batch_size=64)

    model = NeuralNetwork(input_size=784, hidden_size=[128, 64], layers=2, output_size=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    best_model_state, train_losses, val_losses, train_std, val_std, train_accuracies, val_accuracies = model.train_model(
        train_loader, val_loader, loss_function, optimizer, epochs=20, patience=5)

    model.load_state_dict(best_model_state)
    model.evaluate_model(test_loader, loss_function)
    model.plot_results(train_losses, val_losses, train_std, val_std, train_accuracies, val_accuracies)

