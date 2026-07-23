import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import AnomalyAutoEncoder


def train_model(train_loader: DataLoader, val_loader: DataLoader, input_dim: int, epochs: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnomalyAutoEncoder(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed = model(batch)
                loss_v = criterion(reconstructed, batch)
                val_loss += loss_v.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    torch.save(model.state_dict(), "anomaly_model.pth")
    return model