import pandas as pd
import pickle

# Define paths
data_path = 'data/school_fees_data.csv'
model_path = 'models/linear_regression_model.pkl'

# Load the trained model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Model file not found: {model_path}")
    exit()

# Load the data
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Data file not found: {data_path}")
    exit()

# Predict fees for each grade
for _, row in data.iterrows():
    grade = row['Grade']
    predicted_fee = model.predict([[grade]])[0]  # Extract the scalar value
    print(
        f"Predicted fee for grade {grade}: "
        f"{predicted_fee:.2f}"
    )
