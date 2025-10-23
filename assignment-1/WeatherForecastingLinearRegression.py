#import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#load dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/weatherHistory.csv")

data = data.dropna()
data["Precip Type"] = data["Precip Type"].map({"rain": 0, "snow": 1})
X = data[["Humidity", "Pressure (millibars)", "Wind Speed (km/h)", "Precip Type"]]
y = data["Temperature (C)"]

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

#train the model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#calculate metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
