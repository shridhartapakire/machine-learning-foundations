# ML Foundations - Day 4

import pandas as pd

# Sample dataset

data = {
    "Hours_Studied": [2, 4, 6, 8, 10],
    "Marks": [30, 45, 60, 80, 95]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Features (X)
X = df[["Hours_Studied"]]

# Target (y)
y = df["Marks"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)