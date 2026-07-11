from typing import Tuple

import numpy as np
from numpy.lib import recfunctions as rfn

TEST_SIZE = 0.2
RANDOM_SEED = 42

DATE_FIELDS = ('data_abertura', 'data_resultado_compra')


def clean_data(data: np.ndarray) -> np.ndarray:
    """
    Remove linhas com algum campo vazio ou só de espaços, olhando todas
    as colunas de texto do structured array.
    """
    valid_mask = np.ones(data.shape[0], dtype=bool)

    for field_name in data.dtype.names:
        field_values = np.char.strip(data[field_name])
        valid_mask &= field_values != ''

    return data[valid_mask]


def _to_iso_date(value: str) -> str:
    """'dd/mm/yyyy' -> 'yyyy-mm-dd' (formato que datetime64 aceita nativamente)."""
    day, month, year = value.split('/')
    return f'{year}-{month}-{day}'


def parse_dates(data: np.ndarray) -> np.ndarray:
    """
    Converte os campos de data (texto 'dd/mm/yyyy') para datetime64[D].

    A renomeação de colunas já acontece no loader.py (na origem), então
    essa função substitui a antiga normalize_columns e só cuida das datas.
    """
    result = data
    for field_name in DATE_FIELDS:
        iso_dates = np.array(
            [_to_iso_date(v) for v in data[field_name]],
            dtype='datetime64[D]'
        )
        result = rfn.drop_fields(result, field_name)
        result = rfn.append_fields(result, field_name, iso_dates, usemask=False)

    return result


def split_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide o structured array em treino (80%) e teste (20%), com seed fixa
    (substitui o train_test_split do sklearn por uma versão pura NumPy).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = data.shape[0]
    shuffled_idx = rng.permutation(n)

    test_size = int(np.round(n * TEST_SIZE))
    test_idx = shuffled_idx[:test_size]
    train_idx = shuffled_idx[test_size:]

    return data[train_idx], data[test_idx]


def classify_dates(data: np.ndarray) -> np.ndarray:
    """
    Calcula os dias de diferença entre abertura e resultado, e classifica:
    <=0 alerta | 1-3 suspeito | 4-365 normal | >365 muito demorado
    """
    days_diff = (
        data['data_resultado_compra'] - data['data_abertura']
    ) / np.timedelta64(1, 'D')

    classification = np.where(
        days_diff <= 0,
        'alerta',
        np.where(
            days_diff <= 3,
            'suspeito',
            np.where(
                days_diff <= 365,
                'normal',
                'muito demorado'
            )
        )
    )

    data = rfn.append_fields(data, 'dias_demorados', days_diff, usemask=False)
    data = rfn.append_fields(data, 'classificacao', classification, usemask=False)

    values, counts = np.unique(data['classificacao'], return_counts=True)
    for value, count in zip(values, counts):
        print(f'{value}: {count}')

    return data