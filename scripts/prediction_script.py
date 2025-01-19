import pickle
import pandas as pd
import numpy as np

# Define paths to the CSV file and model file
data_file_path = 'data/school_fees_data.csv'  # Update this to your CSV file path
model_file_path = 'models/linear_regression_model.pkl'

# Load the saved model
try:
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found: {model_file_path}")
    exit()

# Load data from CSV file
try:
    data = pd.read_csv(data_file_path)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"CSV file not found: {data_file_path}")
    exit()

# Ensure the data has the expected column 'Grade'
if 'Grade' not in data.columns:
    print("The data file must contain a 'Grade' column.")
    exit()

# Prepare grades for prediction
grades = data['Grade'].values.reshape(-1, 1)

# Make predictions
predicted_fees = model.predict(grades)

# Add predictions to the original data
data['Predicted_Fee'] = predicted_fees

# Print results
print(data)

# Save the results to a new CSV file
output_file_path = 'data/predicted_school_fees.csv'
data.to_csv(output_file_path, index=False)
print(f"Predicted fees saved to: {output_file_path}")
