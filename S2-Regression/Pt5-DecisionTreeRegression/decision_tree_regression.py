### Decision Tree Regression

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Position_Salaries.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)

## Visualizing the Decision Tree results
x_grid = np.arange(min(x), max(x), 0.001)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result
y_prediction = regressor.predict([[6.5]])
