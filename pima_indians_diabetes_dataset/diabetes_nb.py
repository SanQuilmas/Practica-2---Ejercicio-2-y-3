from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
import numpy as np
import pandas as pd

df = pd.read_csv('pima-indians-diabetes.csv', header=None)

X = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
y = df.iloc[:, 8].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

clf = GaussianNB()

y_pred = clf.fit(X_train, y_train).predict(X_test)

print("Número de puntos mal etiquetados de un total de %d puntos : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(classification_report(y_test, y_pred))
print("accuracy = " + str(accuracy_score(y_test, y_pred)))