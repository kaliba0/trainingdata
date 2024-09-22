import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
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
    y_train = pd.read_csv(os.path.join(input_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(input_dir, 'y_test.csv')).squeeze()
    
    return X_train, X_test, y_train, y_test

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

def optimize_model(X_train, y_train):
    """
    Optimizes the Random Forest model using GridSearchCV to find the best hyperparameters.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.

    Returns:
        RandomForestClassifier: The optimized Random Forest model.
    """
    # Define the model
    model = RandomForestClassifier(random_state=42)
    
    # Define the hyperparameters grid to search
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    # Fit the Grid Search model
    grid_search.fit(X_train, y_train)
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    
    # Return the best model found
    return grid_search.best_estimator_

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

def save_optimized_model(model, output_dir='trainingdata', model_filename='optimized_model.joblib'):
    """
    Saves the optimized model to a file.

    Args:
        model (RandomForestClassifier): The optimized model.
        output_dir (str): The directory where the model will be saved.
        model_filename (str): The filename for the saved model.
    """
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the model using joblib
    model_path = os.path.join(output_dir, model_filename)
    dump(model, model_path)
    print(f"Optimized model saved successfully at {model_path}")

def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    # Optimize the model
    optimized_model = optimize_model(X_train, y_train)
    
    # Evaluate the optimized model
    evaluate_model(optimized_model, X_test, y_test)
    
    # Save the optimized model
    save_optimized_model(optimized_model)

if __name__ == "__main__":
    main()
