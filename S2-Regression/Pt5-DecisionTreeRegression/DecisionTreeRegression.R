### Decision Tree Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt5-DecisionTreeRegression')
getwd()

## Import data
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

## Fitting the Decision Tree Regression to the dataset
install.packages('rpart')
library(rpart)

regressor = rpart(formula=Salary~., data=dataset, control=rpart.control(minsplit=1))

## Visualizing the Decision Tree results
library(ggplot2)
x_grid = seq(min(dataset$Salary), max(dataset$Salary), 0.001)
ggplot() +
  geom_point(aes(x=dataset$Level, y=dataset$Salary), color='red') +
  geom_line(aes(x=dataset$Level, y=predict(regressor, newdata=dataset)), color='blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression') +
  xlab('Level') +
  ylab('Salary')

## Predicting a new result
y_prediction = predict(regressor, data.frame(Level=6.5))
