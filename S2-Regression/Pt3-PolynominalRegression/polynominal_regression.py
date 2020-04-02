### Polynomial Regression

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Position_salaries.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

## Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

## Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=4)
x_poly = polynomial.fit_transform(x)
polynomial_linear = LinearRegression()
polynomial_linear.fit(x_poly, y)

## Visualizing Linear Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lr.predict(x), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Visualizing Polynomial Regression results
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid)), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, polynomial_linear.predict(polynomial.fit_transform(x_grid)), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

## Predicting a new result with Linear Regression
lr.predict([[6.5]])

## Predicting a new result with Polynomial Regression
polynomial_linear.predict(polynomial.fit_transform([[6.5]]))