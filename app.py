from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Linear Regression Model using Gradient Descent
class LinearRegression:
    def __init__(self, alpha=0.001, beta=0.001, n_iterations=1000):
        self.alpha = alpha  # Learning rate for weights
        self.beta = beta    # Learning rate for bias
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def mean_squared_error(self, y_true, y_predicted):
        """Calculate the mean squared error."""
        return np.mean((y_true - y_predicted) ** 2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iteration in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute the gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Check for NaN values
            if np.any(np.isnan(dw)) or np.isnan(db):
                print("NaN encountered in gradient calculations. Stopping training.")
                break

            # Update weights and bias
            self.weights -= self.alpha * dw
            self.bias -= self.beta * db

            # Log cost every 100 iterations for monitoring
            if iteration % 100 == 0:
                cost = self.mean_squared_error(y, y_predicted)
                print(f"Iteration {iteration}: Cost {cost}, Weights {self.weights}, Bias {self.bias}")

            # Check for overflow
            if np.any(np.isinf(self.weights)) or np.isinf(self.bias):
                print("Overflow encountered in weights or bias. Stopping training.")
                break

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Function to load data and train the model
def train_model_from_file(file_path, feature_columns, target_column, alpha, beta):
    # Load data from the uploaded Excel/CSV file
    data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

    X = data[feature_columns].values
    y = data[target_column].values

    # Feature scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LinearRegression(alpha=alpha, beta=beta, n_iterations=1000)
    model.fit(X[y != 0], y[y != 0])  # Train only on non-zero target values

    return model, data, scaler

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith(('.xls', '.xlsx', '.csv')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Read the data to show available columns for dynamic selection
        data = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

        # Get available columns
        available_columns = data.columns.tolist()

        # You can customize alpha and beta values here
        alpha = 0.0001  # Learning rate for weights
        beta = 0.0001   # Learning rate for bias

        # Initialize a scaler to be used later for prediction
        scaler = StandardScaler()

        # Create a copy of the original data for updating zero values
        updated_data = data.copy()

        # Iterate through each column
        for target_column in available_columns:
            # Treat all other columns as features
            feature_columns = [col for col in available_columns if col != target_column]

            if not feature_columns:
                continue  # Skip if there are no features to use for training

            # Check for rows with zero in the current target column
            zero_rows = updated_data[updated_data[target_column] == 0]
            non_zero_rows = updated_data[updated_data[target_column] != 0]

            if not zero_rows.empty:
                try:
                    # Train model only on non-zero target values
                    model, _, scaler = train_model_from_file(file_path, feature_columns, target_column, alpha, beta)
                except ValueError as e:
                    return jsonify({'error': str(e)}), 400

                # Prepare the features for zero rows
                X_zero = zero_rows[feature_columns].values
                X_zero_scaled = scaler.transform(X_zero)

                # Predict the missing values for zero entries in the current target column
                predicted_values = abs(model.predict(X_zero_scaled))

                # Update the zero values in the current target column with the predicted values
                updated_data.loc[updated_data[target_column] == 0, target_column] = predicted_values

                # Debug: Check predicted values
                print(f"Predicted values for column '{target_column}': {predicted_values}")

        # Convert the updated DataFrame to CSV
        csv_data = updated_data.to_csv(index=False)

        # Send back the modified CSV data as a response
        return jsonify({'csv': csv_data, 'available_columns': available_columns}), 200

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
