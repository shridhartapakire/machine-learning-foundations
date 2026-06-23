# ML Foundations - Day 7

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset
data = {
    "Hours_Studied": [2, 4, 6, 8, 10, 12, 14, 16],
    "Marks": [30, 45, 60, 80, 95, 110, 125, 140]
}

df = pd.DataFrame(data)

X = df[["Hours_Studied"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)