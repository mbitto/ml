import tests2 as t
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)


a = 'regression'
b = 'classification'
c = 'both regression and classification'

models = {
    'decision trees': c,
    'random forest': c,
    'adaptive boosting': c,
    'logistic regression': b,
    'linear regression': a
}

t.q1_check(models)

linearRegression = LinearRegression()
randomForestRegressor = RandomForestRegressor()
decisionTreeRegressor = DecisionTreeRegressor()
adaBoostRegressor = AdaBoostRegressor()


# Fit each of the models using the training data
linearRegression.fit(X_train, y_train)
randomForestRegressor.fit(X_train, y_train)
decisionTreeRegressor.fit(X_train, y_train)
adaBoostRegressor.fit(X_train, y_train)

# Use each of the models to predict on the test data.
y_linearRegressionPred = linearRegression.predict(X_test)
y_randomForestRegressorPred = randomForestRegressor.predict(X_test)
y_decisionTreeRegressorPred = decisionTreeRegressor.predict(X_test)
y_adaBoostRegressorPred = adaBoostRegressor.predict(X_test)


# Now for the information related to this lesson.
# Use the dictionary to match the metrics that are used for regression and
# those that are for classification.


# potential model options
a = 'regression'
b = 'classification'
c = 'both regression and classification'

#
metrics = {
    'precision': b,
    'recall': b,
    'accuracy': b,
    'r2_score': a,
    'mean_squared_error': a,
    'area_under_curve': b,
    'mean_absolute_area': a
}

# checks your answer, no need to change this code
t.q6_check(metrics)


# Similar to what you did with classification models,
# let's make sure you are comfortable with how exactly each
# of these metrics is being calculated.
# We can then match the value to what sklearn provides.


def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst


# # Check solution matches sklearn
print(r2(y_test, y_decisionTreeRegressorPred))
print(r2_score(y_test, y_decisionTreeRegressorPred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")


# Your turn fill in the function below and see if your result matches
# the built in for mean_squared_error.


def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''

    return np.sum((actual-preds)**2)/len(actual)

# from sklearn.metrics import precision_score, metrics.recall_score,
# metrics.accuracy_score, metrics.r2_score, metrics.mean_squared_error,
# metrics.mean_absolute_error,


# Check your solution matches sklearn
print(mse(y_test, y_decisionTreeRegressorPred))
print(mean_squared_error(y_test, y_decisionTreeRegressorPred))
print("If the above match, you are all set!")


# Now one last time - complete the function related to mean absolute error.
# Then check your function against the sklearn metric to assure they match.


def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''

    return np.sum(np.abs(actual-preds))/len(actual)


# Check your solution matches sklearn
print(mae(y_test, y_decisionTreeRegressorPred))
print(mean_absolute_error(y_test, y_decisionTreeRegressorPred))
print("If the above match, you are all set!")


# Which model performed the best in terms of each of the metrics?
# Note that r2 and mse will always match, but the mae may give
# a different best model.
# Use the dictionary and space below to match the best model via
# each metric.


# match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'


best_fit = {
    'mse':  b,
    'r2':  b,
    'mae': b
}

# Tests your answer - don't change this code
t.check_ten(best_fit)
