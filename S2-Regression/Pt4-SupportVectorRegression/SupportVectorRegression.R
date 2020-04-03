### Support Vector Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt4-SupportVectorRegression///')
getwd()

## Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting the SVR to the dataset
install.packages('e1071')
library(e1071)

regressor = svm(formula=Salary~., data=dataset, type='eps-regression')

## Visualizing the SVR results
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level, y=predict(regressor, newdata=dataset)), color='blue') +
  ggtitle('Truth or Bluff (SVR') +
  xlab('Level') +
  ylab('Salary')

## Predicting a new result
y_prediction =predict(regressor, data.frame(Level=6.5))
