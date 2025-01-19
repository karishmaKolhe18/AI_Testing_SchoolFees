import pandas as pd
import pickle

# Path to model and test data
model_path = 'models/linear_regression_model.pkl'
test_data_path = 'data/school_fees_data.csv'

# Load the trained model
try:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}.")
    exit()

# Load test data
try:
    test_data = pd.read_csv(test_data_path)
    print("Test data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Test data file not found at {test_data_path}.")
    exit()
except Exception as e:
    print(f"An error occurred while loading test data: {e}")
    exit()

# Predict fees for grades in the test data
try:
    for index, row in test_data.iterrows():
        grade = row['Grade']
        predicted_fee = model.predict([[grade]]).item()  # Convert array to scalar
        print(
            f"Predicted fee for grade {grade}: "
            f"{predicted_fee:.2f}"
        )

except Exception as e:
    print(f"An error occurred during prediction: {e}")
