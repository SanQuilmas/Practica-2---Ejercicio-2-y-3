from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd

df = pd.read_csv('winequality.csv', header=None)

X = df.iloc[1:4899, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
y = df.iloc[1:4899, 11].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler = StandardScaler().fit(X_test)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Número de puntos mal etiquetados de un total de %d puntos : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(classification_report(y_test, y_pred))
print("accuracy = " + str(accuracy_score(y_test, y_pred)))