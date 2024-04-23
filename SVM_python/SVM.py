import numpy as np
from LP_solver import Solver
from qpsolvers import solve_qp

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

class KernelTrick:
    def poly_trick(X_i, X_j, d):
        # since X_j is a 1d vector. Reshaping it helps us do np.multiply

        return (X_i.dot(X_j.reshape(X_j.shape[0], 1)))**d # (N x fet) * (fet, 1)

class SVM:
    def __init__(self, trick_func = KernelTrick.poly_trick):
        self.trick_func = trick_func
        self.params = None

    def fit(self, X, Y, lamb=0.1, *args):
        vars_cnt = X.shape[0]

        C = -np.ones((vars_cnt))
        Q = np.empty((vars_cnt, vars_cnt))

        # X.shape[0] = number of data points
        # Q(i, j) = k(x_i, x_j)*y_i*y_j
        for i in range(vars_cnt):
            for j in range(i, vars_cnt):
                Q[i][j] = self.trick_func(X[i], X[j], *args) * Y[i][0] * Y[j][0]
                Q[j][i] = Q[i][j]

        # Y's shape: (var_count, 1)
        M = Y.T
        N = np.zeros((1))

        A = np.identity(vars_cnt)
        B = np.ones(X.shape[0]) / (2 * vars_cnt *  lamb)

        self.params = [0] * vars_cnt
        self.X = X
        self.Y = Y
        self.args = args

        # correct = solve_qp(Q, C, A, B, M, N, solver="scs", lb=np.zeros(vars_cnt))
        _, optimal = Solver.min_quadratic_solver(Q, A, B, C, M, N, optimized_X=self.params)
        self.params = np.array(self.params)

        margin_point_index = -1
        max_coeff = -float('inf')
        
        for index, param in enumerate(self.params):
            if abs(param) < 1e-10 or abs(param - 1 / (2 * X.shape[0] * lamb)) < 1e-10:
                continue

            if param > 0 and param < 1 / (2 * X.shape[0] * lamb) and max_coeff < param:
                margin_point_index = index
                max_coeff = max(param, max_coeff)

                if max_coeff > 1e-3:
                    break # if large enough, stop searching
        
        sum = np.sum(np.multiply(
            np.multiply(self.params.reshape((vars_cnt, 1)), Y), 
            self.trick_func(X, X[margin_point_index], *args)))

        self.b = sum - Y[margin_point_index]

        return optimal
    
    def predict(self, X):
        vars_cnt = self.params.shape[0]
        preds = np.empty((X.shape[0]))

        for i in range(len(preds)):
            preds[i] = np.sign(np.sum(
                np.multiply(
                    np.multiply(self.params.reshape((vars_cnt, 1)), self.Y), 
                    self.trick_func(self.X, X[i], *self.args))) - self.b)[0]

        return preds
    
class OneClassSVM:
    def __init__(self, trick_func = KernelTrick.poly_trick):
        self.trick_func = trick_func
        self.params = None

    def get_dist_from_center(self, Z):
        kerel_trick_sum_on_Z = 2 * \
            np.sum(np.multiply(
                    self.params.reshape(self.params.shape[0], 1), 
                    self.trick_func(self.X, Z, *self.args)))

        return \
            self.trick_func(Z, Z, self.args) + self.constant_term - kerel_trick_sum_on_Z

    def fit(self, X, *args, lamb=1.0):
        n = X.shape[0]
        Q = np.empty((n, n))

        # Q is the kernel table * 2
        for i in range(n):
            Q[i:, i] =  (self.trick_func(X[i:], X[i], *args) * 2).reshape((n - i,))
            Q[i, i:] = Q[i:, i]

        C = -np.diagonal(Q) / 2 # since Q is double the kernel table

        M = np.ones((1, n))
        N = np.ones((1))

        A = np.identity(n)
        B = np.ones(n) / (lamb * n)

        full_params = [0] * n

        full_params = solve_qp(Q, C, A, B, M, N, solver="scs", lb=np.zeros(n))
        # Solver.min_quadratic_solver(Q, A, B, C, M, N, full_params)
        full_params = np.array(full_params)

        I_sv = []
        for index, param in enumerate(full_params):
            if abs(param) < 1e-4 or abs(param - 1 / (lamb * n)) < 1e-4:
                continue # too close to 0 or 1/vn

            if param < (1 / (lamb * n)):
                I_sv.append(index)

        self.X = X[I_sv]
        self.params = full_params[I_sv]
        # precalculate constant term in predictor formula

        n = self.params.shape[0]

        kernel_table = np.empty((n, n))
        for i in range(n):
            for j in range(i, n):
                kernel_table[i][j] = Q[I_sv[i]][I_sv[j]] / 2
                kernel_table[j][i] = kernel_table[i][j]

        self.constant_term = np.sum(
            np.multiply(
                self.params.reshape(n, 1).dot(self.params.reshape(1, n)),
                kernel_table))
        
        self.squared_r = float('inf')
        self.args = args

        for i in range(len(self.X)):
            self.squared_r = min(self.squared_r, self.get_dist_from_center(self.X[i]))

    def predict(self, Z):
        preds = np.empty((Z.shape[0]))

        for index in range(Z.shape[0]):
            diff = self.squared_r - self.get_dist_from_center(Z[index])
            preds[index] = (1 if diff >= 0 else -1)

        return preds
    
def draw_points(mu, var, num_points=100):
    X = np.empty((num_points,))

    for i in range(num_points):
        X[i] = np.random.normal(loc=mu, scale=np.sqrt(var))

    return X

# mu = 1.2
# var = 0.01

# X = draw_points(mu, var)
# X = X.reshape(X.shape[0], 1)

# import matplotlib.pyplot as plt
# plt.plot(X.reshape(X.shape[0]), np.zeros(X.shape[0]), marker='o', markerfacecolor='blue', markersize=12)
# plt.show()

# model = OneClassSVM()

# model.fit(X, 2, lamb=0.01)
# labels = model.predict(X)

# print(np.sqrt(model.squared_r))
# pass