import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.models.isolation_forest import IsolationForest
from src.utils.config import Config


@dataclass
class TrainHistory:
    """
    Armazena o histórico e métricas do Isolation Forest.
    
    Attributes:
        avg_train_score: Pontuação média de anomalia nos dados de treino.
        avg_test_score: Pontuação média de anomalia nos dados de teste.
    """
    avg_train_score: float
    avg_test_score: float


def train_isolation_forest(
    X_train: np.ndarray,
    X_test: np.ndarray,
    config: Config,
) -> Tuple[IsolationForest, TrainHistory]:
    """
    Treina o Isolation Forest com os dados de treino e avalia nos dados de teste.
    
    Args:
        X_train: Array 2D de dados de treino.
        X_test: Array 2D de dados de teste.
        config: Configurações do treinamento (hiperparâmetros, caminhos, etc.).
        
    Returns:
        Tuple[PyTorchIsolationForest, TrainHistory]: O modelo treinado e o histórico de avaliação.
    """
    # Define a semente aleatória para reprodudutibilidade do NumPy e PyTorch
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    input_dim = X_train.shape[1] # Obtém o número de features (colunas) da matriz de treino.

    # Instancia o modelo Isolation Forest utilizando parâmetros do config (com fallbacks se não definidos)
    n_estimators = getattr(config, "n_estimators", 100)
    max_samples = getattr(config, "max_samples", 256)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
    )

    # Imprimir informações de treino
    print(f"\nTreinando Isolation Forest | entrada={input_dim} features | "
          f"{X_train.shape[0]} treino / {X_test.shape[0]} teste\n")

    # Ajusta o Isolation Forest aos dados de treino
    model.fit(X_train)

    # Define o dispositivo de inferência (GPU/CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calcula as pontuações médias de anomalia para treino e teste
    train_scores = compute_anomaly_scores(model, X_train, device=device)
    test_scores = compute_anomaly_scores(model, X_test, device=device)

    avg_train_score = float(np.mean(train_scores))
    avg_test_score = float(np.mean(test_scores))

    history = TrainHistory(
        avg_train_score=avg_train_score,
        avg_test_score=avg_test_score,
    )

    print(f"Treino concluído | "
          f"score médio treino = {avg_train_score:.6f} | "
          f"score médio teste = {avg_test_score:.6f}")

    return model, history # Retorna o modelo treinado e o histórico.


def save_artifacts(
    model: IsolationForest,
    mean: np.ndarray,
    std: np.ndarray,
    feature_names: List[str],
    config: Config,
) -> None:
    """
    Salva os artefatos do modelo treinado, incluindo o estado do Isolation Forest,
    o scaler e os nomes das colunas.
    
    Args:
        model: O Isolation Forest treinado.
        mean: Média dos dados de treino (para normalização).
        std: Desvio padrão dos dados de treino (para normalização).
        feature_names: Lista com os nomes das colunas.
        config: Objeto de configuração.

    Returns:
        None
    """
    # Cria o diretório de artefatos, se não existir.
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Salva o modelo Isolation Forest via Pickle (por conter estruturas de árvore em Python/NumPy)
    with open(config.model_path, "wb") as f:
        pickle.dump(model, f)

    # Salva o scaler (média e desvio padrão) e os nomes das colunas em arquivos separados.
    np.savez(config.scaler_path, mean=mean, std=std)
    Path(config.feature_names_path).write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Imprime os caminhos dos artefatos salvos.
    print(f"\nModelo salvo em:  {config.model_path}")
    print(f"Scaler salvo em:  {config.scaler_path}")
    print(f"Colunas salvas:   {config.feature_names_path}")


def compute_anomaly_scores(
    model: IsolationForest,
    X: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """
    Calcula os scores de anomalia para os dados fornecidos usando o Isolation Forest treinado.
    
    Args:
        model: O modelo Isolation Forest treinado.
        X: Array 2D de dados para os quais calcular os scores de anomalia.
        device: Dispositivo ('cpu' ou 'cuda') para acelerar a inferência com PyTorch.
        
    Returns:
        np.ndarray: Array 1D com os scores de anomalia para cada amostra (valores entre 0 e 1).
    """
    scores_tensor = model.compute_anomaly_score(X, device=device)
    return scores_tensor.cpu().numpy() # Retorna os scores de anomalia como um array NumPy.