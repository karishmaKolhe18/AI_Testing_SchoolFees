import pickle
import pandas as pd
import os

# Define paths to the model and test data files
model_path = os.path.join("models", "linear_regression_model.pkl")
test_data_path = os.path.join("data", "school_fees_data.csv")

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at: {model_path}")
    exit()

# Load test data
try:
    test_data = pd.read_csv(test_data_path)
    print("Test data loaded successfully.")
except FileNotFoundError:
    print(f"Test data file not found at: {test_data_path}")
    exit()

# Perform predictions
try:
    for index, row in test_data.iterrows():
        grade = row['Grade']
        # Predict school fees for the grade
        predicted_fee = model.predict([[grade]])[0]
        print(
            f"Predicted fee for grade {grade}: "
            f"{predicted_fee:.2f}"
        )
except Exception as e:
    print(f"An error occurred during prediction: {e}")
