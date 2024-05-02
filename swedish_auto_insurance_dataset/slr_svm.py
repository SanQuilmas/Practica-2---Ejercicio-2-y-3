from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, classification_report
import numpy as np
import pandas as pd

df = pd.read_csv('slr.csv', header=None)

X = df.iloc[1:, 0].values
X = np.reshape(X, (len(df.index)-1, 1))
y = df.iloc[1:, 1].values
y = np.reshape(y, (len(df.index)-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.50, random_state=0)

clf = svm.LinearSVC(dual="auto")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Número de puntos mal etiquetados de un total de %d puntos : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print(classification_report(y_test, y_pred))
print("accuracy = " + str(accuracy_score(y_test, y_pred)))