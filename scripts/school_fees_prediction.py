import pickle
import pandas as pd
import os

# Define paths relative to the repository root
test_data_path = os.path.join('data', 'test_inputs.csv')
model_path = os.path.join('models', 'linear_regression_model.pkl')

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Model file not found: {model_path}")
    exit()

# Load test inputs
try:
    test_data = pd.read_csv(test_data_path)
    for index, row in test_data.iterrows():
        grade = row['grade']
        predicted_fee = model.predict([[grade]])
        print(f"Predicted fee for Grade {grade}: {predicted_fee[0]:.2f}")
except Exception as e:
    print(f"An error occurred: {e}")
