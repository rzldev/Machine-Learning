### Multipole Linear Regression

## Set current working directory
getwd()
setwd('C:\\Users\\rzl\\Documents\\Github\\Machine-Learning\\S2-Regression\\Pt2-MultipleLinearRegression/')
getwd()

## Import data
dataset = read.csv('Startups.csv')

## Encoding categorical data
dataset$State = factor(dataset$State, levels=c('New York', 'California', 'Florida'), labels=c(1, 2, 3))

## Splitting the dataset into Training set and Test set
# Import caTools
install.packages('caTools')
library('caTools')

# set.seed() -> Set the seed of R's random number generator, which is useful for creating simulations
set.seed(123)

# sample.split() -> Split Data Into Test And Train Set
dataset_split = sample.split(dataset$Profit, SplitRatio=3/4)
training_set = subset(dataset, dataset_split == TRUE)
test_set = subset(dataset, dataset_split == FALSE)

# ## Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

## Fitting Simple Linear Regression to the Training set
regressor = lm(formula=Profit~., data=training_set)
summary(regressor)

## Prediction the Test set results
y_predict = predict(regressor, newdata=test_set)
y_predict

## Steps of Backward Eliminations
# 1. Select a signficance level to stay in the model (e.g. SL = 0.05)
# 2. Fit the full model with all possible predictors
# 3. Consider the predictor with the highest P value. if P > SL, go to step 4, otherwise it's done
# 4. Remove the predictor
# 5. Fit model without his variable

## Building the optimal model using Backward Elimination
# Step 3 - 5
regressor = lm(formula=Profit~R.D.Spend+Administration+Marketing.Spend+State, data=dataset)
summary(regressor)

regressor = lm(formula=Profit~R.D.Spend+Administration+Marketing.Spend, data=dataset)
summary(regressor)

regressor = lm(formula=Profit~R.D.Spend+Marketing.Spend, data=dataset)
summary(regressor)

regressor = lm(formula=Profit~R.D.Spend, data=dataset)
summary(regressor)

## Automatic Backward Elimination
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)