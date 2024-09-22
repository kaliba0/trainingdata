import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import load
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data(input_dir='trainingdata'):
    """
    Loads the testing data from CSV files.

    Args:
        input_dir (str): The directory where the test data is stored.

    Returns:
        tuple: X_test, y_test datasets.
    """
    X_test = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv')).squeeze()
    return X_test, y_test

def load_trained_model(model_path='trainingdata/trained_model.joblib'):
    """
    Loads the trained model from a file.

    Args:
        model_path (str): Path to the trained model file.

    Returns:
        RandomForestClassifier: The loaded trained model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = load(model_path)
    print("Model loaded successfully.")
    return model

def evaluate_predictions(y_test, y_pred):
    """
    Evaluates the performance of the model predictions.

    Args:
        y_test (pd.Series): True labels from the test set.
        y_pred (pd.Series): Predicted labels from the model.

    Returns:
        dict: A dictionary with evaluation metrics.
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calculate successful and failed predictions
    successes = (y_test == y_pred).sum()
    failures = (y_test != y_pred).sum()
    
    # Display evaluation metrics
    print(f"\n--- Model Evaluation ---")
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Successful Predictions: {successes}")
    print(f"Failed Predictions: {failures}")
    
    # Display detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Display confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Return metrics for further analysis or logging
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'successes': successes,
        'failures': failures
    }

def plot_confusion_matrix(conf_matrix):
    """
    Plots the confusion matrix using seaborn for better visualization.

    Args:
        conf_matrix (ndarray): The confusion matrix to be plotted.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_predictions(y_test, y_pred):
    """
    Plots the successful and failed predictions for better understanding.

    Args:
        y_test (pd.Series): True labels.
        y_pred (pd.Series): Predicted labels.
    """
    # Convert to DataFrame for easier manipulation
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results['Correct'] = results['Actual'] == results['Predicted']
    
    # Count successes and failures
    success_count = results['Correct'].sum()
    failure_count = (~results['Correct']).sum()
    
    # Plot successes and failures
    plt.figure(figsize=(8, 5))
    sns.countplot(data=results, x='Correct', palette=['red', 'green'])
    plt.title('Successful vs Failed Predictions')
    plt.xlabel('Prediction Correctness')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Failed', 'Successful'])
    plt.show()

def generate_performance_report(metrics):
    """
    Generates a performance report based on the evaluation metrics.

    Args:
        metrics (dict): A dictionary containing evaluation metrics.
    """
    accuracy = metrics['accuracy']
    report = metrics['classification_report']
    successes = metrics['successes']
    failures = metrics['failures']
    
    print("\n--- Performance Report ---")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Total Successful Predictions: {successes}")
    print(f"Total Failed Predictions: {failures}")
    print("Precision for 'Increase' predictions:", report['1']['precision'])
    print("Recall for 'Increase' predictions:", report['1']['recall'])
    print("F1 Score for 'Increase' predictions:", report['1']['f1-score'])
    print("Precision for 'Decrease' predictions:", report['0']['precision'])
    print("Recall for 'Decrease' predictions:", report['0']['recall'])
    print("F1 Score for 'Decrease' predictions:", report['0']['f1-score'])

def main():
    # Load test data
    X_test, y_test = load_test_data()
    
    # Load the trained model
    model = load_trained_model()
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model predictions
    metrics = evaluate_predictions(y_test, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'])
    
    # Plot successful and failed predictions
    plot_predictions(y_test, y_pred)
    
    # Generate a performance report
    generate_performance_report(metrics)

if __name__ == "__main__":
    main()
