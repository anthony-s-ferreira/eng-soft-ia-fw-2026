import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Raiz do projeto (a pasta que contém 'src/'). Calculada a partir deste arquivo.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    """
    Onde estão os arquivos .zip do Portal da Transparência.

    Ordem de busca:
    1. variável de ambiente LICITACOES_DATA_DIR (se definida);
    2. a pasta data/ do repositório Git irmão (../eng-soft-ia-fw-2026/data);
    3. a pasta data/ local deste esqueleto.
    """
    env = os.environ.get("LICITACOES_DATA_DIR")
    if env:
        return Path(env)

    sibling_repo = PROJECT_ROOT.parent / "eng-soft-ia-fw-2026" / "data"
    if sibling_repo.exists():
        return sibling_repo

    return PROJECT_ROOT / "data"


@dataclass
class Config:
    """Todas as configurações do pipeline em um só objeto."""

    # --- dados ---
    data_dir: Path = field(default_factory=_default_data_dir)
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")

    # --- reprodutibilidade ---
    random_seed: int = 42
    test_size: float = 0.2

    # --- modelo (autoencoder) ---
    hidden1: int = 32
    hidden2: int = 16
    bottleneck: int = 8

    # --- treino ---
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3

    # --- avaliação ---
    top_k: int = 20  # quantas licitações mais anômalas listar

    categorical_features: List[str] = field(
        default_factory=lambda: ["modalidade"]
    )

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "autoencoder.pt"

    @property
    def scaler_path(self) -> Path:
        return self.artifacts_dir / "scaler.npz"

    @property
    def feature_names_path(self) -> Path:
        return self.artifacts_dir / "feature_names.json"


# instância única usada por todo o projeto
CONFIG = Config()
