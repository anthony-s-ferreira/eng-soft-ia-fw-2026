import torch
from torch.utils.data import Dataset
import numpy as np

class BiddingDataset(Dataset):
    def __init__(self, features: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]