import zipfile
import csv
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def normalize_field_name(name: str) -> str:
    name = name.strip().lower()
    unaccented = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    return unaccented.replace(' ', '_')


def read_csv_from_zip(zip_path: Path, file_end: str) -> Optional[Tuple[List[str], List[tuple]]]:
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith(file_end):
                with z.open(file_name) as raw_file:
                    text_lines = (line.decode('latin1') for line in raw_file)
                    reader = csv.reader(text_lines, delimiter=';')
                    try:
                        header = next(reader)
                    except StopIteration:
                        return None
                    rows = [tuple(row) for row in reader if len(row) == len(header)]
                    return header, rows
    return None


def load_data(path: str, file_end: str) -> np.ndarray:
    directory = Path(path)
    zip_files = list(directory.glob('*.zip'))
    header: Optional[List[str]] = None
    all_rows: List[tuple] = []

    for zip_path in zip_files:
        result = read_csv_from_zip(zip_path, file_end)
        if result is None:
            continue
        file_header, rows = result
        if header is None:
            header = file_header
        elif file_header != header:
            raise ValueError(file_header)
        all_rows.extend(rows)

    if header is None or not all_rows:
        return np.array([])

    field_names = [normalize_field_name(col) for col in header]
    max_lens = [max(len(row[i]) for row in all_rows) for i in range(len(field_names))]
    dtype = np.dtype([(name, f'U{max(length, 1)}') for name, length in zip(field_names, max_lens)])

    return np.array(all_rows, dtype=dtype)