import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata
from typing import Tuple
import numpy as np

TEST_SIZE = 0.2
RANDOM_SEED = 42

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa o DataFrame removendo linhas que contenham valores nulos (NaN)
    ou strings completamente vazias/compostas apenas por espaços.
    Também transforma colunas de datas em formato datetime.

    Args:
        df (pd.DataFrame): O DataFrame original.

    Returns:
        pd.DataFrame: Um novo DataFrame sem linhas vazias.
    """
    cleaned_df = df.replace(r'^\s*$', pd.NA, regex=True)

    cleaned_df = cleaned_df.dropna()

    return cleaned_df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas do DataFrame:
    - Converte tudo para letras minúsculas.
    - Remove acentos e caracteres especiais.
    - Substitui espaços em branco por subtraços (_).

    Args:
        df (pd.DataFrame): O DataFrame original.

    Returns:
        pd.DataFrame: DataFrame com os nomes das colunas padronizados.
    """
    normalized_df = df.copy()

    new_columns = []

    for col in normalized_df.columns:
        col_str = str(col).lower()

        unaccented_col = ''.join(
            c for c in unicodedata.normalize('NFD', col_str)
            if unicodedata.category(c) != 'Mn'
        )

        final_col = unaccented_col.strip().replace(' ', '_')

        new_columns.append(final_col)

    normalized_df.columns = new_columns
    normalized_df['data_abertura'] = pd.to_datetime(normalized_df['data_abertura'], format='%d/%m/%Y')
    normalized_df['data_resultado_compra'] = pd.to_datetime(normalized_df['data_resultado_compra'], format='%d/%m/%Y')

    return normalized_df

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide o DataFrame em conjuntos de treino (80%) e teste (20%).

    Args:
        data (pd.DataFrame): O DataFrame limpo a ser dividido.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Uma tupla contendo (train_data, test_data).
    """
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    return train_data, test_data


def classify_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula os dias de diferença e classifica as compras.
    Regras:
    - <= 0 dias: alerta
    - 1 a 3 dias: suspeito
    - 4 a 365 dias: normal
    - > 365 dias: muito demorado
    """
    df['dias_demorados'] = (df['data_resultado_compra'] - df['data_abertura']) / np.timedelta64(1, 'D')

    df['classificacao'] = np.where(
        df['dias_demorados'] <= 0,
        'alerta',
        np.where(
            df['dias_demorados'] <= 3,
            'suspeito',
            np.where(
                df['dias_demorados'] <= 365,
                'normal',
                'muito demorado'
            )
        )
    )

    print(df['classificacao'].value_counts())

    return df