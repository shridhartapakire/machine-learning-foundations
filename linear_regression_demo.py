# ML Foundations - Day 5

import pandas as pd
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Hours_Studied": [2, 4, 6, 8, 10],
    "Marks": [30, 45, 60, 80, 95]
}

df = pd.DataFrame(data)

# Features (X)
X = df[["Hours_Studied"]]

# Target (y)
y = df["Marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict marks for 7 study hours
prediction = model.predict([[7]])

print("Predicted Marks:", prediction[0])