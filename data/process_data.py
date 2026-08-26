import os

from data_cleaning import clean_data
import sklearn.preprocessing as preprocessing


def process_data():
    clean_df = clean_data()
    processing_dir = os.getenv("PROCESSING_DIR", "../data/processed")
    clean_df.to_csv(os.path.join(processing_dir, "cleaned_data.csv"), index=False)
    return clean_df


def split_data(df, test_size=0.2, random_state=42):
    


if __name__ == "__main__":
    print(process_data().head())