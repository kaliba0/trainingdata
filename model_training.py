import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
import os

def load_preprocessed_data(input_dir='trainingdata'):
    """
    Loads preprocessed training and testing data from CSV files.

    Args:
        input_dir (str): The directory where the preprocessed data is stored.

    Returns:
        tuple: X_train, X_test, y_train, y_test datasets.
    """
    X_train = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).squeeze()  # Corrected way to squeeze
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv')).squeeze()    # Corrected way to squeeze
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains a Random Forest classifier on the training data.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on the test data and prints performance metrics.

    Args:
        model (RandomForestClassifier): The trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing targets.
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, output_dir='trainingdata', model_filename='trained_model.joblib'):
    """
    Saves the trained model to a file.

    Args:
        model (RandomForestClassifier): The trained model.
        output_dir (str): The directory where the model will be saved.
        model_filename (str): The filename for the saved model.
    """
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model using joblib
    model_path = os.path.join(output_dir, model_filename)
    dump(model, model_path)
    print(f"Model saved successfully at {model_path}")

def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
