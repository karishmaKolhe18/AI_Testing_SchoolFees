import pickle
import pandas as pd
from sklearn.exceptions import NotFittedError

# Set paths to the model and test data
model_path = 'models/linear_regression_model.pkl'  # Path to the model file
test_data_path = 'data/school_fees_test.csv'  # Path to the test data CSV

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

# Perform predictions and extract scalar values
try:
    predictions = model.predict(test_data[['Grade']])
    test_data['Predicted Fee'] = predictions.flatten()  # Ensure scalar values are stored
    print("Predictions completed successfully.")
except NotFittedError:
    print("Error: Model is not properly trained.")
    exit()
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
    print(f"Predicted fee for grade {int(grade)}: {predicted_fee:.2f}")
