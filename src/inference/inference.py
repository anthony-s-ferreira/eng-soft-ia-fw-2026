import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.features.feature import build_feature_matrix
from src.models.autoencoder import Autoencoder, reconstruction_error
from src.utils.config import CONFIG, Config


def load_artifacts(
    config: Config = CONFIG,
) -> Tuple[Autoencoder, np.ndarray, np.ndarray, List[str]]:
    """Carrega o modelo atual e os metadados necessários para inferência."""
    with Path(config.feature_names_path).open(encoding="utf-8") as file:
        feature_names = json.load(file)

    scaler = np.load(config.scaler_path)
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    std = np.asarray(scaler["std"], dtype=np.float64)

    if len(feature_names) != len(mean) or len(mean) != len(std):
        raise ValueError("Modelo, scaler e nomes das features têm dimensões diferentes")

    model = Autoencoder(
        input_dim=len(feature_names),
        hidden1=config.hidden1,
        hidden2=config.hidden2,
        bottleneck=config.bottleneck,
    )
    state_dict = torch.load(
        config.model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, mean, std, feature_names


def infer_new_data(
    features: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    config: Config = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula anomalias para uma matriz de features não padronizada.

    A matriz deve ter as mesmas colunas, na mesma ordem, usadas no treinamento.
    A padronização é feita com a média e o desvio salvos em ``scaler.npz``.
    """
    model, mean, std, feature_names = load_artifacts(config)

    if isinstance(features, torch.Tensor):
        values = features.detach().cpu().numpy()
    else:
        values = np.asarray(features)

    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError(
            f"Esperadas features com shape (n, {len(feature_names)}); "
            f"recebido {values.shape}"
        )

    standardized = ((values.astype(np.float64) - mean) / std).astype(np.float32)
    tensor = torch.from_numpy(standardized)
    scores = reconstruction_error(model, tensor).cpu().numpy()
    anomalies = scores > threshold
    return anomalies, scores


def infer_from_tables(
    tables: Dict[str, np.ndarray],
    threshold: float = 0.5,
    config: Config = CONFIG,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Gera features das tabelas e executa inferência com os artefatos salvos."""
    features, feature_names, meta = build_feature_matrix(
        tables, config.categorical_features
    )
    _, _, _, saved_feature_names = load_artifacts(config)
    if feature_names != saved_feature_names:
        raise ValueError(
            "A ordem das features geradas é diferente da ordem usada no treinamento"
        )

    anomalies, scores = infer_new_data(features, threshold, config)
    return anomalies, scores, meta
