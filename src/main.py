from pathlib import Path

from scipy.spatial import transform

from data.loader import load_data
from preprocessing.transform import clean_data, normalize_columns, split_data, classify_dates


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data"

    data = load_data(str(DATA_PATH))
    cleaned_data = clean_data(data)
    normalized_data = normalize_columns(cleaned_data)
    train_data, test_data = split_data(normalized_data)
    classify_dates(normalized_data)


if __name__ == "__main__":
    main()
