### Data Preprocessing

## Import all the library that needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Import dataset with pandas
dataset = pd.read_csv('Data.csv')

# pandas.DataFrame.iloc >> Purely integer-location based indexing for selection by position.
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

## Taking care of missing data
from sklearn.impute import SimpleImputer

# Imputer >> Imputation transformer for completing missing values.
imputer = SimpleImputer(missing_values = np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

## Enoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# # OneHotEncoder >> Encode categorical features as a one-hot numeric array
# oneHotEncoder = OneHotEncoder(categories = x[0])
# x = oneHotEncoder.fit_transform(x).toarray()

## ColumnTransformer >> Applies transformers to columns of an array or pandas DataFrame
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(columnTransformer.fit_transform(x), dtype=np.float)

# LabelEncoder >> Encode target labels with value between 0 and n_classes-1
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(y)

## Splitting the dataset into Training set and Test set
from sklearn.model_selection import train_test_split

# train_test_split >> Split arrays or matrices into random train and test subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler

# StandardScaler() >> Standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
