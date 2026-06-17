import data_loader
import preprocess

def main():
    data = data_loader.load_data("../data")
    cleaned_data = preprocess.clean_data(data)
    normalized_data = preprocess.normalize_columns(cleaned_data)
    train_data, test_data = preprocess.split_data(normalized_data)
    print(train_data.head())
    print(test_data.head())

if __name__ == "__main__":
    main()
