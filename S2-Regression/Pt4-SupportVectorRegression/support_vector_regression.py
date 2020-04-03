### Support Vector Regression

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Position_Salaries.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## Feature Scalling
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x = x_scaler.fit_transform(x)
y = y_scaler.fit_transform(y.reshape(-1, 1))

## Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(x, y)

## Visualizing the SVR results
plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result
y_prediction = y_scaler.inverse_transform(regressor.predict(x_scaler.transform(np.array([[6.5]]))))
