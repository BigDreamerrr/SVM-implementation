from ortools.linear_solver import pywraplp
from LP_solver import Solver
import numpy as np

def or_tools_linear_solves(A, B, C, optimized_X=None):
    """Stores the data for the problem."""
    data = {}
    data["constraint_coeffs"] = A
    data["bounds"] = B
    data["obj_coeffs"] = C
    data["num_vars"] = len(A[0])
    data["num_constraints"] = len(A)

    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        return None

    infinity = solver.infinity()
    x = {}
    for j in range(data["num_vars"]):
        x[j] = solver.NumVar(0, infinity, "x[%i]" % j)

    for i in range(data["num_constraints"]):
        constraint = solver.RowConstraint(-infinity, data["bounds"][i], "")
        for j in range(data["num_vars"]):
            constraint.SetCoefficient(x[j], data["constraint_coeffs"][i][j])

    objective = solver.Objective()
    for j in range(data["num_vars"]):
        objective.SetCoefficient(x[j], data["obj_coeffs"][j])
    objective.SetMaximization()
    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        return None

    if optimized_X != None:
        for j in range(data["num_vars"]):
            optimized_X[j] = x[j].solution_value()

    return solver.Objective().Value()

def evaluate(num_tests=1):
    wrong = 0
    valid_sol = True

    for _ in range(num_tests):
        N = np.random.randint(3, 400)
        M = np.random.randint(3, 400)

        # A = np.random.uniform(low=-3, high=4, size=(M, N))
        # x = np.random.uniform(low=0.0, high=4, size=(N, 1))
        
        A = np.random.uniform(low=-300, high=300, size=(M, N))
        x = np.random.uniform(low=0.0, high=300, size=(N, 1))
        
        B = A.dot(x).reshape((M,)) + np.random.uniform(40, 5_000)
        C = np.random.uniform(low=-300, high=300, size=(N))

        ops_X = [0] * N
        my_ans = Solver.linear_solver(A, B, C, optimized_X=ops_X)
        optimal_ans = or_tools_linear_solves(A.tolist(), B.tolist(), C.tolist())

        # check my answer conditions
        dot_result = A.dot(ops_X)
        
        valid = True
        for j in range(len(dot_result)):
            if not np.isclose(dot_result[j], B[j]) and dot_result[j] > B[j]:
                valid = False
                break
        
        valid_sol = valid and np.isclose(C.T.dot(ops_X), my_ans)

        if optimal_ans == None:
            continue # ignore

        if abs(my_ans - optimal_ans) > 1e-2:
            wrong += 1

    print(f'wrong: {wrong}')
    print(valid_sol)

# evaluate(num_tests=4_000)

A = [
        [5, 7, 9, 2, 1],
        [18, 4, -9, 10, 12],
        [4, 7, 3, 8, 5],
        [5, 13, 16, 3, -7],
    ]

B = [250, 285, 211, 315]
C = [7, 8, 2, 9, 6]

print(or_tools_linear_solves(A, B, C))
print(Solver.linear_solver(np.array(A), np.array(B), np.array(C)))