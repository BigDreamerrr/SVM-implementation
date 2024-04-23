import numpy as np
from SVM import SVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qpsolvers import solve_qp

f = open('./iris.data')
label = { 
    'Iris-setosa' : 1, 
    'Iris-versicolor' : -1,
    'Iris-virginica' : -1
}

X = np.zeros((150, 4))
Y = np.zeros((150, 1))

for index in range(150):
    line = f.readline()
    if line == '':
        break

    line = line.replace('\n', '')

    data = line.split(',')
    X[index] = [float(x) for x in data[0:len(data)-1]]
    Y[index] = label[data[-1]]

f.close()
X_train, X_test, y_train, y_test = train_test_split(
     X, Y, test_size=0.33, random_state=42, stratify=Y)

model = SVM()

print("Solved!" if model.fit(X_train, y_train, 1.0, 3) else "Wrong!")
pred = model.predict(X_test)

print(accuracy_score(y_train, model.predict(X_train)))

print(len(y_test))
print(accuracy_score(y_test, pred))

pass

# from sklearn.svm import SVC

# model = SVC(kernel='poly')
# model.fit(X_train, y_train)

# pred = model.predict(X_test)

# print(accuracy_score(pred, y_test))