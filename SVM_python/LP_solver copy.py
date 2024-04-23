from typing import List
import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def linear_solver(A, B, C, optimized_X = None):
        M, N = A.shape

        # construct solving mat
        # (M + 1) equations, including M equations from A, last equation involving non-basic vars
        # (M + N + 1) coefficients, including b entries in the last column

        solving_mat = np.zeros((M + 1, M + N + 1))

        solving_mat[:M, :N] = A
        
        for i in range(M):
            solving_mat[i, N + i] = 1
            solving_mat[i, -1] = B[i]

        solving_mat[-1, :N] = -C[:]

        # solving...
        while True:
            min_none_basic_coeff = solving_mat[-1].min()

            if min_none_basic_coeff >= 0:
                break

            min_pos = solving_mat[-1].argmin()
            
            amount_upper_bound_pos = -1
            amount_upper_bound = float('inf')

            for i in range(M):
                B_i = solving_mat[i, -1]

                if solving_mat[i, min_pos] <= 0:
                    continue # this variable does not contribute to final answer

                if B_i / solving_mat[i, min_pos] < amount_upper_bound:
                    amount_upper_bound = B_i / solving_mat[i, min_pos]
                    amount_upper_bound_pos = i

            if amount_upper_bound == float('inf'):
                break

            solving_mat[amount_upper_bound_pos, :] /= \
                solving_mat[amount_upper_bound_pos, min_pos]
            
            # do row transformation to move chosen non-basic variable to basis

            for i in range(M + 1):
                if i == amount_upper_bound_pos:
                    continue # ignore row with definition of newly allocated basic variable
                
                factor = solving_mat[i, min_pos]
                solving_mat[i, :] = \
                    solving_mat[i, :] - solving_mat[amount_upper_bound_pos, :] * factor

        if optimized_X != None:
            for j in range(N):
                if solving_mat[-1, j] == 0: # climb up vertically to find its value, in basis
                    non_zero_row = np.where(solving_mat[:, j] == 1)[0]

                    if len(non_zero_row) != 0:
                        optimized_X[j] = solving_mat[non_zero_row[0], -1]
                    else:
                        optimized_X[j] = 0
        
        return solving_mat[-1][-1] # max value of the problem

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

class Solution:    
    #Function to get the maximum total value in the knapsack.
    def fractionalknapsack(self, W, arr, n):
        weights = [arr[i].weight for i in range(n)]
        values = [arr[i].value for i in range(n)]

        return KnapsackSolver.solve(values, weights, W)

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

        values = [np.random.randint(200, 3000) for _ in range(n)]
        weights =  [np.random.randint(20, 100) for _ in range(n)]
        W = np.random.randint(10, 300)
        X = [0] * n

        my_ans = KnapsackSolver.solve(values, weights, W, X=X)
        real_ans = fractionalknapsack(
            W, [Item(values[_], weights[_]) for _ in range(n)], n)
        
        if abs(my_ans - real_ans) > 1e-8:
            wrong += 1

    print(f'wrong: {wrong}')
        

# evaluate(num_tests=20_000)

values = np.load('val.npy')
weights = np.load('weights.npy')
W = np.load('W.npy')

print(KnapsackSolver.solve(values, weights, W))

# n = np.random.randint(3, 5)

# values = [np.random.randint(200, 3000) for _ in range(n)]
# weights =  [np.random.randint(20, 100) for _ in range(n)]
# W = np.random.randint(10, 300)
# X = [0] * n
# # my_ans = KnapsackSolver.solve(values, weights, W, X=X)