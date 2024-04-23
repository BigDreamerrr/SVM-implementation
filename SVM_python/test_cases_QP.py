import numpy as np
from scipy import optimize
from LP_solver import Solver
from qpsolvers import solve_qp

def semi_definite(X):
    return np.all(np.linalg.eigvals(X) >= 0)

def evaluate(num_tests=1):
    wrong = 0
    for _ in range(num_tests):
        N = np.random.randint(2, 100)
        M = np.random.randint(2, 100)
        
        A = np.random.uniform(low=-300, high=400, size=(M, N))
        x = np.random.uniform(low=0.0, high=400, size=(N, 1))
        Q = np.random.uniform(low=-300, high=400, size=(N, N))
        Q  = Q.dot(Q.T)
        
        B = A.dot(x).reshape((M,)) + np.random.randint(40, 100)
        B = B.astype(np.float32)
        C = np.random.uniform(low=-300, high=400, size=(N))

        opt_X = solve_qp(Q, C, A, B, solver="scs", lb=np.zeros(N))
        if opt_X is None:
            continue
        my_opt_X = [0] * N

        real_ans = 0.5 * opt_X.T.dot(Q).dot(opt_X) + C.T.dot(opt_X)
        my_ans, optimal = Solver.min_quadratic_solver(Q, A, B, C, 
                                                      optimized_X=my_opt_X)

        if np.linalg.norm(my_opt_X - opt_X) > 2.5:
            np.save('Q', Q)
            np.save('A', A)
            np.save('B', B)
            np.save('C', C)
            wrong += 1

    print(wrong)
# evaluate(num_tests=6_000)

# Q = np.load('Q.npy')
# A = np.load('A.npy') 
# B = np.load('B.npy')
# C = np.load('C.npy')

# opt_X = solve_qp(Q, C, A, B, solver="scs", lb=np.zeros(A.shape[0]))
# my_opt_X = [0] * A.shape[0]

# print(opt_X)

# real_ans = 0.5 * opt_X.T.dot(Q).dot(opt_X) + C.T.dot(opt_X)
# my_ans, optimal = Solver.min_quadratic_solver(Q, A, B, C, 
#                                                       optimized_X=my_opt_X)

# pass

Q = np.array([
    [1, -1],
    [-1, 2]
])

A = np.array([
    [1, 1],
    [-2, -3]
])

B = np.array([3, -6])
C = np.array([-1, -1])

optimized = [0] * 2
ans = Solver.min_quadratic_solver(Q, A, B, C, optimized_X=optimized)

print(ans)