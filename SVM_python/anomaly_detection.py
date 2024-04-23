import pandas as pd
import numpy as np
from SVM import OneClassSVM
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

frame = pd.read_csv(r'conn250K.csv', header=None)

X = frame.to_numpy().astype(np.float64)
X[:, 0] /= (X[:, 1] + X[:, 2]) # how many seconds to trafer a byte?
X[:, 0] = np.nan_to_num(X[:, 0], posinf=0, neginf=0)

scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# fig = plt.figure(figsize=(12, 12))
# ax = fig.add_subplot(projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2])
# plt.show()

model = OneClassSVM()
model.fit(X[:1000], 1, lamb=0.1)

pass