import data_loader
import preprocess
from pathlib import Path

def main():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR.parent / "data"
    data = data_loader.load_data(str(DATA_PATH))
    cleaned_data = preprocess.clean_data(data)
    normalized_data = preprocess.normalize_columns(cleaned_data)
    train_data, test_data = preprocess.split_data(normalized_data)
    print(train_data.head())
    print(test_data.head())

if __name__ == "__main__":
    main()
