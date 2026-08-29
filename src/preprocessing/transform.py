from typing import Tuple
import numpy as np
from numpy.lib import recfunctions as rfn

TEST_SIZE = 0.2
RANDOM_SEED = 42
DATE_FIELDS = ('data_abertura', 'data_resultado_compra')

def clean_data(data: np.ndarray) -> np.ndarray:
    valid_mask = np.ones(data.shape[0], dtype=bool)
    for field_name in data.dtype.names:
        field_values = np.char.strip(data[field_name])
        valid_mask &= field_values != ''
    return data[valid_mask]

def to_iso_date(value: str) -> str:
    day, month, year = value.split('/')
    return f'{year}-{month}-{day}'

def parse_dates(data: np.ndarray) -> np.ndarray:
    result = data
    for field_name in DATE_FIELDS:
        iso_dates = np.array(
            [to_iso_date(v) for v in data[field_name]],
            dtype='datetime64[D]'
        )
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