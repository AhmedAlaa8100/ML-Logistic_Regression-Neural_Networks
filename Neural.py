import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch


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
    def __init__(self, input_size, hidden_size=[5, 3], layers=2, output_size=2):
        super(NeuralNetwork, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU()

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
        # Hidden layers with ReLU
        for i in range(self.layers):
            fc_layer = getattr(self, f'fc{i+1}')
            x = self.relu(fc_layer(x))
        # Output layer (no activation)
        output_layer = getattr(self, f'fc{self.layers + 1}')
        x = output_layer(x)
        return x
    
    # Custom training loop
    # • Optimizer: Stochastic Gradient Descent (SGD)
    # • Learning rate: 0.01 (baseline)
    # • Loss function: Cross-entropy
    # • Batch size: 64 (baseline)
def train_model(self, train_loader, val_loader, loss_function, optimizer, epochs=10):
    self.best_val_loss = float('inf')
    self.best_model_state = None

    for epoch in range(epochs):
        self.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validation phase
        self.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        # Track progress
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model during training
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_model_state = self.state_dict()
            torch.save(self.best_model_state, 'best_model.pth')

# Example usage:
if __name__ == "__main__":
    model = NeuralNetwork(10, [32, 16, 8], layers=3, output_size=4)
    print(model)
