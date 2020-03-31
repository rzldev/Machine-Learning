### Simple Linear Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt1-SimpleLinearRegression/')
getwd()

## Import data
dataset = read.csv('Salary_Data.csv')

## Splitting the dataset into Training set and Test set
# Import caTools
install.packages('caTools')
library('caTools')

# set.seed() -> Set the seed of R's random number generator, which is useful for creating simulations
set.seed(123)

# sample.split() -> Split Data Into Test And Train Set
dataset_split = sample.split(dataset$Salary, SplitRatio=3/4)
training_set = subset(dataset, dataset_split == TRUE)
test_set = subset(dataset, dataset_split == FALSE)

# ## Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

## Fitting Simple Linear Regression to the Training set
regressor = lm(formula=Salary~YearsExperience, data=training_set)
summary(regressor)

## Prediction the Test set results
y_predict = predict(regressor, newdata=test_set)
y_predict

## Visualizing with ggplot2
install.packages('ggplot2')
library(ggplot2)

# Visualizing the Training set results
ggplot() +geom_point(aes(x=training_set$YearsExperience, y=training_set$Salary), colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)), colour='blue') +
  ggtitle('Salary vs Experience(Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')

# Visualizing the Test set results
ggplot() +geom_point(aes(x=test_set$YearsExperience, y=test_set$Salary), colour='red') +
  geom_line(aes(x=training_set$YearsExperience, y=predict(regressor, newdata=training_set)), colour='blue') +
  ggtitle('Salary vs Experience(Training set)') +
  xlab('Years of Experience') +
  ylab('Salary')
