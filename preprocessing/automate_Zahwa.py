import pandas as pd

def preprocess_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Encoding categorical columns (jika ada)
    categorical_cols = df.select_dtypes(include='object').columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save processed data
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    preprocess_data(
        input_path="../heart_raw/heart.csv",
        output_path="heart_preprocessed.csv"
    )
