import pickle
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

# Set paths to the model and test data
# Path to the model file
model_path = 'models/linear_regression_model.pkl'  
# Path to the test data CSV
test_data_path = 'data/school_fees_data.csv'  

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load test data
try:
    test_data = pd.read_csv(test_data_path)
    print(f"Test data loaded successfully from {test_data_path}")
except FileNotFoundError:
    print(f"Error: Test data file not found at {test_data_path}")
    exit()
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# Ensure the test data contains the expected 'Grade' column
if 'Grade' not in test_data.columns:
    print("Error: Test data must contain a 'Grade' column.")
    exit()

# Ensure 'Grade' column is numeric
test_data['Grade'] = pd.to_numeric(test_data['Grade'], errors='coerce')
if test_data['Grade'].isnull().any():
    print("Error: 'Grade' column contains non-numeric values.")
    exit()

# Perform predictions and reverse the log transformation
try:
    log_predictions = model.predict(test_data[['Grade']])
    # Reverse the transformation
    test_data['Predicted Fee'] = np.exp(log_predictions)  
    print("Predictions completed successfully.")
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# Save the results to a new CSV file
output_path = 'data/school_fees_predictions.csv'
try:
    test_data.to_csv(output_path, index=False)
    print(f"Predictions saved successfully to {output_path}")
except Exception as e:
    print(f"Error saving predictions: {e}")
    exit()

# Print scalar predictions for verification
print("Predicted Fees for Each Grade:")
for index, row in test_data.iterrows():
    grade = row['Grade']
    predicted_fee = row['Predicted Fee']
    print(
        f"Grade {int(grade)}: Fee Prediction = {predicted_fee:.2f}"
    )
