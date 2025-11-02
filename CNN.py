import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# --------------------------- DATA LOADER --------------------------- #
def load_data_CNN(file_path, batch_size=64):
    data = pd.read_csv(file_path)
    Y = data['label'].values
    X = data.drop(columns=['label'], axis=1).values
    X = X / 255.0  # Normalize pixel values
    X = X.reshape(-1, 1, 28, 28)  # Reshape to (N, 1, 28, 28) for CNN input

    # Split the data 60-20-20 for train, validation, and test
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.4, stratify=Y, random_state=30)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, stratify=Y_temp, random_state=30)

    # Convert to tensors
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


# --------------------------- MODEL DEFINITION --------------------------- #
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    # --------------------------- TRAINING --------------------------- #
    def train_model(self, train_loader, val_loader, loss_function, optimizer, device, epochs=6, patience=3):
        train_losses, val_losses = [], []
        train_std, val_std = [], []
        train_accuracies, val_accuracies = [], []

        # Initialize early stopping tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.early_stopping_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            correct_train, total_train = 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = torch.tensor(epoch_losses).mean().item()
            std_train_loss = torch.tensor(epoch_losses).std().item()
            train_losses.append(avg_train_loss)
            train_std.append(std_train_loss)
            train_acc = 100 * correct_train / total_train
            train_accuracies.append(train_acc)

            # ---------------- Validation ---------------- #
            self.eval()
            val_batch_losses = []
            correct_val, total_val = 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
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
            val_acc = 100 * correct_val / total_val
            val_accuracies.append(val_acc)

            # ---- LOGGING ----
            print(f"Epoch [{epoch+1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f} | "
                  f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

            # ---- EARLY STOPPING ----
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.state_dict()
                # torch.save(self.best_model_state, 'best_model.pth')
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= patience:
                    print("No improvement, stopping early.")
                    break

        return (self.best_model_state, train_losses, val_losses,
                train_std, val_std, train_accuracies, val_accuracies)

    # --------------------------- TESTING --------------------------- #
    def test_model(self, test_loader, device):
        self.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        return accuracy

    # --------------------------- PLOTTING --------------------------- #
    def plot_metrics(self, train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 4))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.show()


# --------------------------- MAIN EXECUTION --------------------------- #
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = load_data_CNN('mnist_All.csv', batch_size=64)
    
    model = SimpleCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    best_state_CNN, train_losses_CNN, val_losses_CNN, train_std_CNN, val_std_CNN, train_accs_CNN, val_accs_CNN = model.train_model(
        train_loader, val_loader, criterion, optimizer, device, epochs=6, patience=3) 

    model.load_state_dict(best_state_CNN)
    model.test_model(test_loader, device)
    model.plot_metrics(train_losses_CNN, val_losses_CNN, train_accs_CNN, val_accs_CNN)
    torch.save(best_state_CNN, 'cnn_best_model.pth')