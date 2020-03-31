### Data Preprocessing

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Startups.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

## Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

## ColumnTransformer >> Applies transformers to columns of an array or pandas DataFrame
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype=np.float)

## Avoiding the dummy variable trap
x = x[:, 1:]

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

## Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## Predicting the Test set results
y_predict = regressor.predict(x_test)

## Steps of Backward Eliminations
# 1. Select a signficance level to stay in the model (e.g. SL = 0.05)
# 2. Fit the full model with all possible predictors
# 3. Consider the predictor with the highest P value. if P > SL, go to step 4, otherwise it's done
# 4. Remove the predictor
# 5. Fit model without his variable

## Building the optimal model using backward elimintaion
import statsmodels.api as sm
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)

# Step 3 - 5
x_opt = x
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())

## Automatic Backward Elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_modeled = backwardElimination(x_opt, SL)