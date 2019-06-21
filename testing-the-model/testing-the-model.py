# Import statements
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


# Read in the data.
data = np.asarray(pd.read_csv('data.csv', header=None))

# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# Use train test split to split your data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Instantiate your decision tree model
model = DecisionTreeClassifier(
    max_depth=10, min_samples_leaf=5, min_samples_split=5)

model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy and assign it to the variable acc on the test data.
acc = accuracy_score(y_test, y_pred)
