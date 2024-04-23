import numpy as np


class KnapsackSolver:
    def solve(values, weights, W, X=None):
        n = len(values)

        C = np.array(values)
        B = np.empty((n + 1))
        B[0] = W
        B[1:] = 1

        A = np.empty((n + 1, n))
        A[0, :] = weights
        A[1:, :] = np.identity(n)

        return Solver.linear_solver(A, B, C, X)

class Item:
    def __init__(self,val,w):
        self.value = val
        self.weight = w

def fractionalknapsack(W, arr, n):
    # Sorting Item on basis of ratio
    arr.sort(key=lambda x: (x.value/x.weight), reverse=True)    

    # Result(value in Knapsack)
    finalvalue = 0.0

    # Looping through all Items
    for item in arr:

        # If adding Item won't overflow, 
        # add it completely
        if item.weight <= W:
            W -= item.weight
            finalvalue += item.value

        # If we can't add current Item, 
        # add fractional part of it
        else:
            finalvalue += item.value * W / item.weight
            break
     
    # Returning final value
    return finalvalue

def evaluate(num_tests=400):
    wrong = 0
    for i in range(num_tests):
        n = np.random.randint(3, 40)

        values = np.array(
            [np.random.randint(200, 3000) for _ in range(n)])
        weights =  np.array(
            [np.random.randint(20, 100) for _ in range(n)])
        W = np.random.randint(10, 300)

        X = [0] * n

        my_ans = KnapsackSolver.solve(values, weights, W, X=X)
        real_ans = fractionalknapsack(
            W, [Item(values[_], weights[_]) for _ in range(n)], n)
        
        if abs(my_ans - real_ans) > 1e-8:
            wrong += 1

    print(f'wrong: {wrong}')
        
# evaluate(num_tests=20_000)

# Q = np.array([
#     [1, -1],
#     [-1, 2]
# ], dtype=np.float64)

# A = np.array([
#     [1, 1],
#     [-2, -3]
# ], dtype=np.float64)

# B = np.array([3, -6], dtype=np.float64)
# C = np.array([-1, -1], dtype=np.float64)

# X = np.array([[2, -1]], dtype=np.float64) # X should be mat
# Y = np.array([1], dtype=np.float64)

# optimized = [0] * 2
# ans = Solver.min_quadratic_solver(Q, A, B, C, X, Y, optimized_X=optimized)

# print(ans)
# print(optimized)

# from qpsolvers import solve_qp

# opt_X = solve_qp(Q, C, A, B, X, Y, solver="scs", lb=np.zeros(2))

# optimized = np.array(optimized)
# pass

A = np.array([
    [2, 1, 1],
    [1, 2, 3],
    [2, 2, 1]
])

B  = np.array([ 2, 4, 8 ])
C = np.array([ 4, 1, 4 ])

X = [0] * 3
ans = Solver.linear_solver(A, B, C, optimized_X=X)
pass