import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from datetime import datetime

def load_data(input_file):
    """
    Loads historical data from a JSON file.

    Args:
        input_file (str): Path to the JSON file containing historical data.

    Returns:
        pd.DataFrame: DataFrame containing the historical data.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Convert the JSON data into a pandas DataFrame
    df = pd.DataFrame(data)
    # Convert timestamp to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def add_features(df):
    """
    Adds additional features to the DataFrame to aid in prediction.

    Args:
        df (pd.DataFrame): The DataFrame containing historical data.

    Returns:
        pd.DataFrame: DataFrame with additional features.
    """
    # Calculate daily returns
    df['return'] = df['close'].pct_change()

    # Calculate moving averages
    df['ma7'] = df['close'].rolling(window=7).mean()  # 7-day moving average
    df['ma21'] = df['close'].rolling(window=21).mean()  # 21-day moving average

    # Calculate price change compared to the previous day
    df['price_diff'] = df['close'].diff()

    # Calculate volatility (standard deviation of returns over 7 days)
    df['volatility'] = df['return'].rolling(window=7).std()

    # Forward shift the close price to compare with the current price
    df['future_close'] = df['close'].shift(-1)

    # Create target: 1 if future close is higher, 0 if lower
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # Drop rows with NaN values caused by rolling calculations
    df = df.dropna()

    return df

def split_data(df, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The DataFrame containing features and target.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    # Features (excluding the timestamp and target columns)
    X = df[['open', 'high', 'low', 'close', 'volume', 'ma7', 'ma21', 'price_diff', 'volatility']]
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test

def save_preprocessed_data(X_train, X_test, y_train, y_test, output_dir='trainingdata'):
    """
    Saves the preprocessed data to CSV files for training and testing.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training targets.
        y_test (pd.Series): Testing targets.
        output_dir (str): The directory where the preprocessed data will be saved.
    """
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the datasets to CSV files
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Preprocessed data saved successfully in", output_dir)

def main():
    # Load and preprocess the data
    input_file = 'historical_data.json'  # Modify the path if needed
    df = load_data(input_file)
    df = add_features(df)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)

    # Save the preprocessed data
    save_preprocessed_data(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
