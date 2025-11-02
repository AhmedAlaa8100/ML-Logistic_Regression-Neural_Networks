import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        # fc1, fc2, ..., fcN (input_size -> hidden_size[0], hidden_size[i-1] -> hidden_size[i])
        # fc stands for fully connected layer
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
            x = fc_layer(x)
            x = self.relu(x)
        output_layer = getattr(self, f'fc{self.layers + 1}')
        return output_layer(x)


    def train_model(self, train_loader, val_loader, loss_function, optimizer, epochs=10, patience=3):
        train_losses, val_losses = [], []
        train_std, val_std = [], []
        train_accuracies, val_accuracies = [], []
        self.early_stopping_counter = 0

        for epoch in range(epochs):
            self.train()
            epoch_losses = []
            correct_train, total_train = 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Zero the gradients as optimizer accumulates them
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                # Backpropagation (compute gradients dL/dW)
                loss.backward()
                # Update weights (W = W - lr * dL/dW)
                optimizer.step()

                epoch_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            avg_train_loss = torch.tensor(epoch_losses).mean().item()
            std_train_loss = torch.tensor(epoch_losses).std().item()
            train_losses.append(avg_train_loss)
            train_std.append(std_train_loss)
            train_accuracies.append(correct_train / total_train)

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
            val_accuracies.append(correct_val / total_val)

            # ---- LOGGING ----
            # print(f"Epoch [{epoch+1}/{epochs}] | "
            #       f"Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f} | "
            #       f"Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f} | "
            #       f"Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

            # ---- EARLY STOPPING ----
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.state_dict()
                torch.save(self.best_model_state, 'best_model.pth')
                self.early_stopping_counter = 0
                # print("New best model saved.")
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= patience:
                    print("No improvement, stopping early.")
                    break

        return self.best_model_state, train_losses, val_losses, train_std, val_std, train_accuracies, val_accuracies

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
        plt.plot(epochs, loss_gap, color='purple', label='|Train - Val Loss| (Convergence Analysis)')
        plt.xlabel('Epochs'); plt.ylabel('Loss Gap')
        plt.title('Convergence Analysis')
        plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.show()


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

def run_learning_rate_analysis():
    train_loader, val_loader, _ = load_data('mnist_All.csv', batch_size=64)

    results = {}
    learning_rates = [0.001, 0.01, 0.1, 1.0]
    print("\n===== Learning Rate Analysis =====")

    for lr in learning_rates:
        print(f"\nTesting Learning Rate = {lr}")
        model = NeuralNetwork(input_size=784, hidden_size=[128, 64], layers=2, output_size=10).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        best_model_state, train_losses, val_losses, train_std, val_std, train_acc, val_acc = model.train_model(
            train_loader, val_loader, loss_function, optimizer, epochs=100, patience=5)

        conv_speed = convergence_speed(val_losses)
        stability = stability_measure(val_losses)

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

        model = NeuralNetwork(input_size=784, hidden_size=[128, 64], layers=2, output_size=10).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        best_model_state, train_losses, val_losses, train_std, val_std, train_acc, val_acc = model.train_model(
            train_loader, val_loader, loss_function, optimizer, epochs=100, patience=5)

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

def run_architecture_analysis():

    train_loader, val_loader, _ = load_data('mnist_All.csv', batch_size=64)

    results = {}
    layer_options = [2, 3, 4, 5]
    neuron_options = [64, 128, 256, 512]

    print("\n===== Architecture Analysis =====")

    for layers in layer_options:
        for neurons in neuron_options:
            print(f"\nTesting Architecture: Layers={layers}, Neurons={neurons}")
            hidden_size = [neurons] * layers

            model = NeuralNetwork(input_size=784, hidden_size=hidden_size, layers=layers, output_size=10).to(device)
            loss_function = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            best_model_state, train_losses, val_losses, train_std, val_std, train_acc, val_acc = model.train_model(
                train_loader, val_loader, loss_function, optimizer, epochs=100, patience=5)

            conv_speed = convergence_speed(val_losses)
            stability = stability_measure(val_losses)

            results[(layers, neurons)] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'convergence_speed': conv_speed,
                'stability': stability
            }

    # ---- Find best architecture ----
    best_arch = max(results, key=lambda x: max(results[x]['val_acc']))
    best_acc = max(results[best_arch]['val_acc'])
    print(f"\n Best Architecture: Layers={best_arch[0]}, Neurons={best_arch[1]} (Val Accuracy: {best_acc:.4f})")

    # ---- Create comparison table ----
    table_data = []
    for (layers, neurons), data in results.items():
        table_data.append({
            'Layers': layers,
            'Neurons': neurons,
            'Best_Val_Accuracy': max(data['val_acc']),
            'Convergence_Speed': data['convergence_speed'],
            'Stability': data['stability']
        })

    df = pd.DataFrame(table_data).sort_values(by='Best_Val_Accuracy', ascending=False)
    print("\n=== Architecture Comparison Table ===")
    print(df.to_string(index=False))

    return results, df, best_arch, best_acc


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

def plot_architecture_analysis(results):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 10))
    plt.suptitle('Architecture Analysis', fontsize=14, fontweight='bold')

    # (1) Train Loss
    plt.subplot(2, 2, 1)
    for (layers, neurons), data in results.items():
        plt.plot(data['train_losses'], label=f'L={layers}, N={neurons}')
    plt.xlabel('Epochs'); plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epochs')
    plt.legend(fontsize=7); plt.grid(True)

    # (2) Validation Loss
    plt.subplot(2, 2, 2)
    for (layers, neurons), data in results.items():
        plt.plot(data['val_losses'], label=f'L={layers}, N={neurons}')
    plt.xlabel('Epochs'); plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epochs')
    plt.legend(fontsize=7); plt.grid(True)

    # (3) Train Accuracy
    plt.subplot(2, 2, 3)
    for (layers, neurons), data in results.items():
        plt.plot(data['train_acc'], label=f'L={layers}, N={neurons}')
    plt.xlabel('Epochs'); plt.ylabel('Train Accuracy')
    plt.title('Train Accuracy vs Epochs')
    plt.legend(fontsize=7); plt.grid(True)

    # (4) Validation Accuracy
    plt.subplot(2, 2, 4)
    for (layers, neurons), data in results.items():
        plt.plot(data['val_acc'], label=f'L={layers}, N={neurons}')
    plt.xlabel('Epochs'); plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy vs Epochs')
    plt.legend(fontsize=7); plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_architecture_surface(df):
    # Prepare data
    layers = sorted(df['Layers'].unique())
    neurons = sorted(df['Neurons'].unique())

    # Create grid
    X, Y = np.meshgrid(layers, neurons)
    Z = np.zeros_like(X, dtype=float)

    for i, n in enumerate(neurons):
        for j, l in enumerate(layers):
            row = df[(df['Layers'] == l) & (df['Neurons'] == n)]
            if not row.empty:
                Z[i, j] = row['Best_Val_Accuracy'].values[0]
            else:
                Z[i, j] = np.nan

    # ---- 3D Surface ----
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_title('3D Surface: Layers vs Neurons vs Accuracy')
    ax.set_xlabel('Layers')
    ax.set_ylabel('Neurons per Layer')
    ax.set_zlabel('Validation Accuracy')
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)

    # ---- Heatmap ----
    ax2 = fig.add_subplot(1, 2, 2)
    sns.heatmap(Z, annot=True, fmt=".3f", cmap='coolwarm',
                xticklabels=layers, yticklabels=neurons)
    ax2.set_title('Heatmap: Validation Accuracy')
    ax2.set_xlabel('Layers')
    ax2.set_ylabel('Neurons per Layer')

    plt.tight_layout()
    plt.show()







class basic_conventional_nn(nn.Module):
    def __init__(self):
        super(basic_conventional_nn, self).__init__()
        
        # Define layers here

    def forward(self, x):
        # Define forward pass here
        return x
















# Example usage:
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    lr_results, best_lr, best_lr_acc = run_learning_rate_analysis()
    plot_learning_rate_analysis(lr_results)

    bs_results, best_bs, best_bs_acc = run_batch_size_analysis()
    plot_batch_size_analysis(bs_results)

    results_arch, df_arch, best_arch, best_acc = run_architecture_analysis()
    plot_architecture_analysis(results_arch)
    plot_architecture_surface(df_arch)

    #best model
    # train_loader, val_loader, test_loader = load_data('mnist_All.csv', batch_size=best_bs)

    # best_hidden_size = [best_arch[1]] * best_arch[0]   # repeat neurons per layer
    # model = NeuralNetwork(input_size=784, hidden_size=best_hidden_size,
    #                       layers=best_arch[0], output_size=10).to(device)

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=best_lr)

    # # ---- Train final best model ----
    # best_model_state, train_losses, val_losses, train_std, val_std, train_acc, val_acc = model.train_model(
    #     train_loader, val_loader, loss_function, optimizer, epochs=100, patience=5)

    # # ---- Evaluate on test set ----
    # model.load_state_dict(best_model_state)
    # model.evaluate_model(test_loader, loss_function)