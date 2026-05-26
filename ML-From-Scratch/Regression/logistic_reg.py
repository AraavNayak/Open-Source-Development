# Logistic Regression is primarily used for binary classification.
# Main idea is to wrap any input equation inside sigmoid function, and then use more-likely-than-not classifier

import numpy as np
from sklearn.linear_model import LogisticRegression

# Features: [Hours Studied, Sleep (hours)]
X = np.array([
    [1, 5], [2, 6], [3, 4], [7, 8], [8, 9], [9, 7]
])

# Target: [0 = Fail, 1 = Pass]
y = np.array([0, 0, 0, 1, 1, 1])

# 1. Initialize and Fit
clf = LogisticRegression()
clf.fit(X, y)

# 2. Predict a Category (0 or 1)
test_student = np.array([[5, 7]])
print(f"Prediction: {clf.predict(test_student)}") 

# 3. Predict Probability (The "raw" confidence)
# Returns [Prob of 0, Prob of 1]
probs = clf.predict_proba(test_student)
print(f"Probability of Passing: {probs[0][1]:.2f}")