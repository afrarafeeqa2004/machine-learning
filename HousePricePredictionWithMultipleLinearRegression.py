#import dataset and packages
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

#load the dataset
california = fetch_california_housing()
X, y = california.data, california.target
X_simple = X[:, [0]]

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression()

#train the model
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

#calculate metrics
mae_mlr = mean_absolute_error(y_test, y_pred_mlr)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
r2_mlr = r2_score(y_test, y_pred_mlr)
print("Multiple Linear Regression")
print("MAE:", mae_mlr)
print("RMSE:", rmse_mlr)
print("R²:", r2_mlr)
