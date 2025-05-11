# Iris-classification
This project demonstrates a simple Machine Learning workflow using the popular Iris dataset. The goal is to classify iris flowers into one of three species (Setosa, Versicolor, Virginica) based on four features: sepal length, sepal width, petal length, and petal width.
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split the Data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create and Train the Model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Make Predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

Result images:
![Screenshot 2025-05-11 120723](https://github.com/user-attachments/assets/5b1af667-10d7-4925-aec1-b8449ecdc6ad)
![Screenshot 2025-05-11 120742](https://github.com/user-attachments/assets/19e7a660-2080-4005-a1f1-6048798ff580)

