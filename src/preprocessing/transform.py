from typing import Tuple
import numpy as np
from numpy.lib import recfunctions as rfn

TEST_SIZE = 0.2
RANDOM_SEED = 42
DATE_FIELDS = ('data_abertura', 'data_resultado_compra')


def parse_valor(values: np.ndarray) -> np.ndarray:
    """Converte valores monetários brasileiros para um array float64."""
    result = []
    for value in np.asarray(values).astype("U"):
        text = value.strip().replace("R$", "").replace(" ", "")
        if not text:
            result.append(np.nan)
            continue
        try:
            result.append(float(text.replace(".", "").replace(",", ".")))
        except ValueError:
            result.append(np.nan)
    return np.asarray(result, dtype=np.float64)

def clean_data(data: np.ndarray) -> np.ndarray:
    valid_mask = np.ones(data.shape[0], dtype=bool)
    for field_name in data.dtype.names:
        field_values = np.char.strip(data[field_name])
        valid_mask &= field_values != ''
    return data[valid_mask]

def to_iso_date(value: str) -> str:
    day, month, year = value.strip().split('/')
    return f'{year}-{month}-{day}'


def _parse_date_values(values: np.ndarray) -> np.ndarray:
    parsed = []
    for value in np.asarray(values).astype("U"):
        text = value.strip()
        if not text:
            parsed.append(np.datetime64("NaT", "D"))
            continue
        try:
            parsed.append(np.datetime64(to_iso_date(text), "D"))
        except (ValueError, TypeError):
            parsed.append(np.datetime64("NaT", "D"))
    return np.asarray(parsed, dtype="datetime64[D]")


def parse_dates(data: np.ndarray) -> np.ndarray:
    """Converte datas dd/mm/aaaa; aceita uma coluna ou array estruturado."""
    if getattr(data.dtype, "names", None) is None:
        return _parse_date_values(data)

    result = data
    for field_name in DATE_FIELDS:
        if field_name not in result.dtype.names:
            continue
        iso_dates = _parse_date_values(result[field_name])
        result = rfn.drop_fields(result, field_name)
        result = rfn.append_fields(result, field_name, iso_dates, usemask=False)
    return result

def extract_numerical_features(data: np.ndarray) -> np.ndarray:
    days_diff = (data['data_resultado_compra'] - data['data_abertura']) / np.timedelta64(1, 'D')
    features = np.vstack([days_diff.astype(np.float32)]).T
    features = np.nan_to_num(features, nan=0.0)
    return features

def split_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = data.shape[0]
    shuffled_idx = rng.permutation(n)
    test_size = int(np.round(n * TEST_SIZE))
    test_idx = shuffled_idx[:test_size]
    train_idx = shuffled_idx[test_size:]
    return data[train_idx], data[test_idx]


def standardize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padroniza features e devolve também média e desvio para inferência."""
    values = np.asarray(data, dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    safe_std = np.where(std == 0, 1.0, std)
    return ((values - mean) / safe_std).astype(np.float32), mean, safe_std


def train_test_split_numpy(
    data: np.ndarray, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide uma matriz em treino/teste usando uma permutação determinística."""
    values = np.asarray(data)
    if not 0 < test_size < 1:
        raise ValueError("test_size deve estar entre 0 e 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(values))
    n_test = int(round(len(values) * test_size))
    return values[indices[n_test:]], values[indices[:n_test]]
