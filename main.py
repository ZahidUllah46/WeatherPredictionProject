import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
data = pd.read_csv("Project 1 - Weather Dataset.csv")

# Show first rows
print(data.head())

# Remove Date Column
data = data.drop(columns=["Date/Time"])

# Convert Weather text into numbers
encoder = LabelEncoder()
data["Weather"] = encoder.fit_transform(data["Weather"])

# Input Features
X = data.drop(columns=["Weather"])

# Output
y = data["Weather"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=500,
    learning_rate_init=0.01,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Plot Loss Curve
plt.plot(model.loss_curve_)

plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.show()