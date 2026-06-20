from pathlib import Path

from data.loader import load_data
from preprocessing.transform import clean_data, normalize_columns, split_data


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data"

    data = load_data(str(DATA_PATH))
    cleaned_data = clean_data(data)
    normalized_data = normalize_columns(cleaned_data)
    train_data, test_data = split_data(normalized_data)

    print(train_data.head())
    print(test_data.head())


if __name__ == "__main__":
    main()
