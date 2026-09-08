import random
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(train_data, val_data, test_data, batch_size=32):
    train_dataset = SequenceDataset(train_data.X, train_data.y)
    val_dataset = SequenceDataset(val_data.X, val_data.y)
    test_dataset = SequenceDataset(test_data.X, test_data.y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu', batch_size=32, seed=42, patience=10, alpha=0.1):
    set_seed(seed)
    model.to(device)
    best_model_state = None
    ema_val_loss = None  # For early stopping based on EMA of validation loss
    best_ema_val_rmse = float('inf')  
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_rmse = torch.sqrt(loss).item() * X_batch.size(0)  # Calculate RMSE (sqrt of MSE)

        train_loss /= len(train_loader.dataset)
        train_rmse /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                # Calculate RMSE (sqrt of MSE)
                val_rmse += torch.sqrt(loss).item() * X_batch.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)

        if ema_val_loss is None:
            ema_val_loss = val_rmse
        else:
            # Update EMA of validation RMSE
            ema_val_loss = alpha * val_rmse + (1 - alpha) * ema_val_loss  # alpha is the smoothing factor for EMA

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.8f}, Train RMSE: {train_rmse:.8f}, Val Loss: {val_rmse:.8f} Val RMSE: {val_rmse:.8f}')

        min_delta = 1e-5
        if best_ema_val_rmse - ema_val_loss > min_delta:
            best_ema_val_rmse = ema_val_loss
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, best_ema_val_rmse  # Return the best EMA validation RMSE for early stopping


def select_optimizer(model, optimizer_name='adam', learning_rate=0.001):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")