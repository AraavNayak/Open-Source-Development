import numpy as np
from sklearn.linear_model import LinearRegression

# Features: [Size (sqft), Age (years), Bedrooms]
X = np.array([
    [2104, 5, 3],
    [1600, 2, 3],
    [2400, 15, 4],
    [1416, 10, 2]
])

# Target: [Price (k)]
y = np.array([400, 330, 369, 232])

model = LinearRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficients: {model.coef_}") # One for each column in X