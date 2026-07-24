import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.autoencoder import Autoencoder, reconstruction_error
from src.utils.config import Config


@dataclass
class TrainHistory:
    train_loss: List[float]
    test_loss: List[float]


def _make_loader(X: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Cria um DataLoader do PyTorch a partir de um array NumPy.
    """
    tensor = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor, tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _avg_loss(model: Autoencoder, X: np.ndarray, loss_fn: nn.Module) -> float:
    """
    Calcula a perda média do modelo para o conjunto de dados fornecido.
    """
    tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(tensor)
        return float(loss_fn(pred, tensor).item())


def train_autoencoder(
    X_train: np.ndarray,
    X_test: np.ndarray,
    config: Config,
) -> Tuple[Autoencoder, TrainHistory]:
    """
    Treina o modelo Autoencoder e registra a evolução do erro em treino e teste.
    """
    torch.manual_seed(config.random_seed)

    input_dim = X_train.shape[1]
    model = Autoencoder(
        input_dim=input_dim,
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        bottleneck=config.bottleneck,
    )

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = _make_loader(X_train, config.batch_size, shuffle=True)

    history = TrainHistory(train_loss=[], test_loss=[])

    print(f"\nTreinando autoencoder | entrada={input_dim} features | "
          f"{X_train.shape[0]} treino / {X_test.shape[0]} teste\n")

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_x, batch_target in train_loader:
            optimizer.zero_grad()          # zera gradientes antigos
            pred = model(batch_x)          # reconstrói o lote
            loss = loss_fn(pred, batch_target)
            loss.backward()                # calcula os gradientes
            optimizer.step()               # ajusta os pesos

        # ---- ENTREGA 3: imprimir erro de treino e de teste ----
        train_loss = _avg_loss(model, X_train, loss_fn)
        test_loss = _avg_loss(model, X_test, loss_fn)
        history.train_loss.append(train_loss)
        history.test_loss.append(test_loss)
        print(f"Época {epoch:3d}/{config.epochs} | "
              f"erro treino = {train_loss:.6f} | erro teste = {test_loss:.6f}")

    return model, history


def save_artifacts(
    model: Autoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: List[str],
    config: Config,
) -> None:
    """
    Salva os artefatos de treinamento (pesos do modelo, scaler e nomes das features) em disco.
    """
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), config.model_path)              # ENTREGA 3: salvar
    np.savez(config.scaler_path, mean=mean, std=std)
    Path(config.feature_names_path).write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nModelo salvo em:  {config.model_path}")
    print(f"Scaler salvo em:  {config.scaler_path}")
    print(f"Colunas salvas:   {config.feature_names_path}")


def compute_anomaly_scores(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    """
    Calcula as pontuações de anomalia (erro de reconstrução) para o conjunto de dados.
    """
    tensor = torch.tensor(X, dtype=torch.float32)
    return reconstruction_error(model, tensor).numpy()