import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.model import AnomalyAutoEncoder

from config import DEVICE


def main():
    device = DEVICE

    data = torch.load("data/bidding_tensors.pt", weights_only=True)

    print(f"Loaded data shape: {data.shape}")
    if len(data) == 0:
        raise ValueError("Dataset is empty. Run prepare_data.py again and verify data extraction.")

    dataset = TensorDataset(data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = data.shape[1]
    model = AnomalyAutoEncoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            all_test_data = test_dataset[:][0].to(device)

            reconstructions = model(all_test_data)
            errors = torch.mean((all_test_data - reconstructions) ** 2, dim=1)

            errors_np = errors.cpu().numpy()

            import numpy as np
            threshold = np.percentile(errors_np, 95)
            anomalies = errors_np > threshold

            total_anomalies = np.sum(anomalies)

            print("\n--- Inference Results (Whole Test Set) ---")
            print(f"Dynamic Threshold (95th percentile): {threshold:.6f}")
            print(f"Total anomalies found: {total_anomalies} out of {len(all_test_data)}")

            top_5_indices = np.argsort(errors_np)[-5:]
            print(f"Top 5 Anomaly Errors: {errors_np[top_5_indices]}")

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    torch.save(model.state_dict(), "anomaly_model.pth")

    model.load_state_dict(torch.load("anomaly_model.pth", weights_only=True))
    model.eval()

    with torch.no_grad():
        sample_data = test_dataset[:10][0].to(device)
        reconstructions = model(sample_data)
        errors = torch.mean((sample_data - reconstructions) ** 2, dim=1)

        threshold = 1.5
        anomalies = errors > threshold

        print("\nInference Results:")
        print(f"Errors: {errors.cpu().numpy()}")
        print(f"Anomalies: {anomalies.cpu().numpy()}")


if __name__ == "__main__":
    main()