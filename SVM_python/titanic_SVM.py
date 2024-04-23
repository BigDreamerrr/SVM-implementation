import pandas as pd
import numpy as np
from SVM import SVM
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'train.csv')

df = df.dropna()

X = np.zeros((712, 7))
Y = np.zeros((712, 1))

col_index = -1
tables = {}

for (columnName, columnData) in df.items():
    if columnName == 'Survived':
        for index, data in enumerate(columnData):
            Y[index] = 1 if data == 1 else -1
        continue
    col_index += 1

    table = []

    for row_id, data in enumerate(columnData):
        if type(data) == int or type(data) == float:
            X[row_id][col_index] = data
            continue

        if data not in table:
            table.append(data)

        X[row_id][col_index] = table.index(data)

    tables[columnName] = table

np.save(r'data/ti_X', X)
np.save(r'data/ti_Y', Y)

model = SVM()
standard_model = SVC()
# model.fit(X, Y, 0.1, 1)
standard_model.fit(X, Y)

# pred = model.predict(X)
pred2 = standard_model.predict(X)

print(accuracy_score(pred2, Y))
# print(accuracy_score(pred, Y))

df = pd.read_csv(r'test.csv').fillna(0)
X_test = np.zeros((418, 7))

col_index = -1
for (columnName, columnData) in df.items():
    col_index += 1

    for row_id, data in enumerate(columnData):
        if type(data) == int or type(data) == float:
            X_test[row_id][col_index] = data
            continue

        X_test[row_id][col_index] = tables[columnName].index(data)

Y_test = model.predict(X_test)

ans_dict = { 'PassengerId' : [], 'Survived' : [] }

for i in range(0, 418):
    ans_dict['PassengerId'].append(892 + i)
    ans_dict['Survived'].append((0 if Y_test[i] < 0 else 1))

df2 = pd.DataFrame.from_dict(ans_dict)
df2.to_csv(r'ans.csv', index=False)