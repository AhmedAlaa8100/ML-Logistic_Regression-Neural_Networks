import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch

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
class LogisticRegressionModel():
    def __init__(self, input_dim,lr=0.01):
        self.W = torch.zeros((input_dim,1), dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.lr = lr
    
    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))
    def  loss(self, y, t):
        m = t.shape[0]
        loss = -(1/m) * torch.sum(t * torch.log(y+1e-9 ) + (1 - t) * torch.log(1 - y+1e-9 ))
        return loss
    def gradient(self, X, y, t):
        m = t.shape[0]
        dw = (1/m) * torch.matmul(X.T, (y - t))
        db = (1/m) * torch.sum(y - t)
        return dw, db
    def update_weights(self, dw, db):
        self.W -= self.lr * dw
        self.b -= self.lr * db
    def accuracy(self, data):
        correct=0
        total=0
        for X_batch, y_batch in data:
            y_batch = y_batch.view(-1, 1).float()
            y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
            predicted = (y_pred >= 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        return correct / total if total > 0 else 0

    def train(self, train, val, epochs=1000):
        train_losses, train_accs = [], []
        for epoch in range(epochs):
            for X_batch, y_batch in train:
                y_batch = y_batch.view(-1, 1).float()
                y_pred = self.sigmoid(torch.matmul(X_batch, self.W) + self.b)
                loss = self.loss(y_pred, y_batch)
                epoch_losses+=loss
                
                dw, db = self.gradient(X_batch, y_pred, y_batch)
                self.update_weights(dw, db)
            train_losses.append(epoch_losses.item()/len(train))
                
            
        
        
