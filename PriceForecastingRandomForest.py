#import dataset and packages
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#load the dataset
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")
data = data[['Close']]

#Value of Close from 1,2,3 days before
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['Lag3'] = data['Close'].shift(3)

data = data.dropna()

X = data[['Lag1', 'Lag2', 'Lag3']]
y = data['Close']

#split the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

#train the model
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

#calculate metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Random Forest Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

plt.figure(figsize=(10,5))
plt.plot(y_test.index, y_test, label="Actual Price", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted Price", color="red")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Forecasting with Random Forest")
plt.legend()
plt.show()
