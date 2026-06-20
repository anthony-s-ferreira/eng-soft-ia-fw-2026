import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Optional

def extract_tender_from_zip(zip_path: Path) -> Optional[pd.DataFrame]:
    """
    Abre um arquivo ZIP, procura pelo CSV de Licitação e retorna seus dados.

    Args:
        zip_path (Path): Caminho completo para o arquivo .zip.

    Returns:
        Optional[pd.DataFrame]: DataFrame com os dados, ou None se o arquivo não for encontrado.
    """
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('_Licitação.csv'):
                with z.open(file_name) as f:
                    return pd.read_csv(
                        f,
                        sep=';',
                        encoding='latin1',
                        dtype=str,
                        on_bad_lines='skip'
                    )
    return None


def load_data(path: str) -> pd.DataFrame:
    """
    Carrega e concatena exclusivamente os dados de Licitação de uma pasta com arquivos ZIP.

    Args:
        path (str): Caminho para o diretório contendo os arquivos .zip.

    Returns:
        pd.DataFrame: DataFrame único com todas as licitações concatenadas.
    """
    directory = Path(path)
    zip_files = list(directory.glob('*.zip'))

    df_list: List[pd.DataFrame] = []

    for zip_path in zip_files:
        df = extract_tender_from_zip(zip_path)

        if df is not None and not df.empty:
            df_list.append(df)

    if df_list:
        return pd.concat(df_list, ignore_index=True)

    return pd.DataFrame()