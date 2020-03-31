### Simple Linear Regression

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Salary_Data.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

## Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split

# train_test_split >> Split arrays or matrices into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/4, random_state=0)

# ## Feature Scaling
# from sklearn.preprocessing import StandardScaler

# # StandardScaler() >> Standardize features by removing the mean and scaling to unit variance
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

## Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

# LinearRegression >> Ordinary least squares Linear Regression
lr = LinearRegression()
lr.fit(x_train, y_train)

## Predicting the Test set results
y_prediction = lr.predict(x_test)

## Visualizing the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

## Visualizing the Test set results
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, lr.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()