import torch
from models.model import AnomalyAutoEncoder

def infer_new_data(tensor_data: torch.Tensor, input_dim: int, threshold: float = 0.5):
    model = AnomalyAutoEncoder(input_dim)
    model.load_state_dict(torch.load("anomaly_model.pth", weights_only=True))
    model.eval()

    with torch.no_grad():
        reconstructions = model(tensor_data)
        errors = torch.mean((tensor_data - reconstructions) ** 2, dim=1)
        anomalies = errors > threshold
        return anomalies, errors