
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

X_train = np.array([
    [2, 50], [3, 60], [4, 65], [5, 70],
    [6, 75], [7, 80], [8, 85], [9, 90]
])

y_train = np.array([0, 0, 0, 1, 1, 1, 1, 1])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

X_test = np.array([
    [3, 55], [5, 72], [8, 88], [4, 60]
])

y_test = np.array([0, 1, 1, 0])
y_pred = model.predict(X_test)

print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Error Rate:", 1 - accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
