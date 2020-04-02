### Polynomial Linear Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt3-PolynominalRegression//')
getwd()

## Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting Linear Regression to the dataset
linear_reg = lm(formula=Salary~., data=dataset)
summary(linear_reg)

## Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Salary^2
dataset$Level3 = dataset$Salary^3
polynomial_reg = lm(formula=Salary~., data=dataset)
summary(polynomial_reg)

## Visualizing
library(ggplot2)

# Linear Regression
ggplot() + 
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colour='red') +
  geom_line(aes(x=dataset$Level, y=predict(linear_reg, newdata=dataset)), colour='blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

## Polynomial Regression
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), colour='red') +
  geom_line(aes(x=dataset$Level, y=predict(polynomial_reg, newdata=dataset)), colour='blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

## Predicting a new result with Linear Regression
linear_pred = predict(linear_reg, data.frame(Level=6.5))

## Predicting a new result with Polynomial Regression
polynomial_pred = predict(polynomial_reg, data.frame(Level=6.5, Level2=6.5^2, Level3=6.5^3))
