#Imports necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

#Prepare data. Note: x must be 2D, rescale if needed using x.reshape(-1,1)
x = np.array([[1], [2], [3], [4], [5]]) 
y = np.array([2, 4, 5, 4, 5])

#Initialize the model and fit to data
model = LinearRegression()
model.fit(x, y)

# Access slope/intercept
intercept = model.intercept_
slope = model.coef_[0]

print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

#Predict
pred = model.predict([[6]])
print(f"Prediction for x=6: {pred[0]}")


'''
Evaluation metrics (sklearn.metrics):
 - Mean Absolute Error (MAE): mean(abs(error))
 - Root Mean Squared Error (RMSE): (np.mean(error**2)) ** 0.5
        Large errors are weighted heavily.
- Coefficient of determination (R^2)
        Indicates what proportion of info model captured 
'''