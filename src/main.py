from pathlib import Path

from data.loader import load_data
from preprocessing.transform import clean_data, parse_dates, split_data, classify_dates


def main() -> None:
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data"

    data = load_data(str(DATA_PATH))
    cleaned_data = clean_data(data)
    parsed_data = parse_dates(cleaned_data)
    train_data, test_data = split_data(parsed_data)
    classify_dates(parsed_data)

if __name__ == "__main__":
    main()
