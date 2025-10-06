#import packages
import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 3, 5, 4, 5])

# Compute the mean of X and Y
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Calculate the slope
numerator = np.sum((X - X_mean) * (Y - Y_mean))
denominator = np.sum((X - X_mean) ** 2)
m = numerator / denominator

# Calculate the intercept
c = Y_mean - m * X_mean

print(f"Slope (m): {m}")
print(f"Intercept (c): {c}")

Y_pred = m * X + c

plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, Y_pred, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
