import pickle
import pandas as pd

# Define the paths for model and data
model_path = 'models/linear_regression_model.pkl'
data_path = 'data/school_fees_data.csv'

# Load the model
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Model file not found: {model_path}")
    exit()

# Load test data
try:
    test_data = pd.read_csv(data_path)
    for index, row in test_data.iterrows():
        grade = row['Grade']
        # Predict the fee for each grade
        predicted_fee = model.predict([[grade]])[0]
        print(
            f"Predicted fee for grade {grade}: "
            f"{predicted_fee:.2f}"
        )
except Exception as e:
    print(f"An error occurred: {e}")
