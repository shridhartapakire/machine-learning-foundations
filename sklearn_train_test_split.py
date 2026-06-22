# ML Foundations - Day 6

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Dataset
data = {
    "Hours_Studied": [2, 4, 6, 8, 10, 12, 14, 16],
    "Marks": [30, 45, 60, 80, 95, 110, 125, 140]
}

df = pd.DataFrame(data)

# Features and Target
X = df[["Hours_Studied"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("\nActual Values:")
print(list(y_test))

print("\nPredicted Values:")
print(predictions)