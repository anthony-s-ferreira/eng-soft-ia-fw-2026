import zipfile
import csv
import unicodedata
from pathlib import Path
import numpy as np
import torch

def normalize_field_name(name: str) -> str:
    name = name.strip().lower()
    unaccented = ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
    )
    return unaccented.replace(' ', '_')


def read_csv_from_zip(zip_path: Path, file_end: str):
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


def process_and_save_data(data_dir: str, output_file: str):
    directory = Path(data_dir)
    zip_files = list(directory.glob('*.zip'))
    header = None
    all_rows = []

    for zip_path in zip_files:
        result = read_csv_from_zip(zip_path, "_Licitação.csv")
        if result is None:
            continue
        file_header, rows = result
        if header is None:
            header = file_header
        all_rows.extend(rows)

    if not all_rows:
        torch.save(torch.empty(0), output_file)
        return

    field_names = [normalize_field_name(col) for col in header]
    idx_abertura = field_names.index('data_abertura')
    idx_resultado = field_names.index('data_resultado_compra')

    features = []
    for row in all_rows:
        val_abertura = row[idx_abertura].strip()
        val_resultado = row[idx_resultado].strip()
        if not val_abertura or not val_resultado:
            continue

        try:
            d1, m1, y1 = val_abertura.split('/')
            d2, m2, y2 = val_resultado.split('/')
            date1 = np.datetime64(f'{y1}-{m1}-{d1}')
            date2 = np.datetime64(f'{y2}-{m2}-{d2}')
            days_diff = (date2 - date1) / np.timedelta64(1, 'D')
            features.append([float(days_diff)])
        except:
            continue

    tensor_data = torch.tensor(features, dtype=torch.float32)
    tensor_data = torch.nan_to_num(tensor_data, nan=0.0)

    torch.save(tensor_data, output_file)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir.parent.parent / "data"
    output_file = base_dir / "bidding_tensors.pt"
    process_and_save_data(str(data_path), str(output_file))