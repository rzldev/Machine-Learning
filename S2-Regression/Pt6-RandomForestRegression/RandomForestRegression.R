### Random Forest Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt6-RandomForestRegression/')
getwd()

## Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting the Random Forest Regression to the dataset
install.packages('randomForest')
set.seed(1234)
library(randomForest)

regressor = randomForest(x=dataset[1], y=dataset$Salary, ntree=500)

## Visualizing the Random Forest Regression results
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.001)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color='red') +
  geom_line(aes(x=x_grid, y=predict(regressor, newdata=data.frame(Level=x_grid))), color='blue') +
  ggtitle('Truth or Bluff (Random Forest Regression') +
  xlab('Level') +
  ylab('Salary')

## Predicting a new result
y_prediction = predict(regressor, data.frame(Level=6.5))

