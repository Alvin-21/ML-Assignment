import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data.csv'
data = pd.read_csv(file_path)
print(data.head())

# Function to compute Mean Squared Error (MSE)
def compute_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to perform Gradient Descent to update weights
def gradient_descent(X, y, m, c, learning_rate):
    N = len(y)
    y_pred = m * X + c

    # Calculate gradients
    dm = (-2 / N) * np.sum(X * (y - y_pred))
    dc = (-2 / N) * np.sum(y - y_pred)

    # Update weights
    m = m - learning_rate * dm
    c = c - learning_rate * dc
    return m, c

# Extracting feature and target arrays
X = data['SIZE'].values
y = data['PRICE'].values

# Initialize random values for slope (m) and intercept (c)
np.random.seed(42)
m = np.random.rand()
c = np.random.rand()

# Training parameters
learning_rate = 0.01
epochs = 10

# List to store MSE for each epoch
mse_list = []

# Loop for 10 epochs
for epoch in range(epochs):
    # Perform a gradient descent step
    m, c = gradient_descent(X, y, m, c, learning_rate)

    # Calculate the MSE for the current epoch
    y_pred = m * X + c
    mse = compute_mse(y, y_pred)
    mse_list.append(mse)

    # Display the MSE for the current epoch
    print(f"Epoch {epoch + 1}: MSE = {mse:.4f}")

# Plotting the line of best fit
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Line of Best Fit')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Linear Regression Line of Best Fit')
plt.legend()
plt.show()

# Predict the price when the size is 100 using the learned line
predicted_price = m * 100 + c
print(predicted_price)
