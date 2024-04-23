import numpy as np
from ortools.linear_solver import pywraplp

class Solver:
    def solve(
            solving_mat,
            def_pos,
            ignore_next_to_last=False,
            conditions_on_entering_basis=None,
            conditions_on_exiting_basis=None,
            patience_level=10):
        
        row_len = solving_mat.shape[0]
        col_len = solving_mat.shape[1]

        max_cost = -float('inf')
        last_repeat = -1
        patience = 0

         # solving...
        while True:
            min_pos = None

            # min of all variable coeffs
            if conditions_on_entering_basis == None:
                min_pos = solving_mat[-1, :-1].argmin()
            else:
                min_coeff = float('inf')
                for j in range(col_len - 1): # ignore b-column
                    if def_pos[j] == -1 and \
                        conditions_on_entering_basis(j, solving_mat, def_pos):
                        if min_coeff > solving_mat[-1, j]:
                            min_coeff = solving_mat[-1, j]
                            min_pos = j

            min_coeff = solving_mat[-1, :-1].min()

            if abs(min_coeff) < 1e-5 or min_coeff > 0:
                break
            
            amount_upper_bound_pos = -1
            amount_upper_bound = float('inf')

            for i in range(row_len - (2 if ignore_next_to_last else 1)):
                B_i = solving_mat[i, -1]
                row_val = solving_mat[i, min_pos]

                if abs(row_val) < 1e-5 or row_val < 0 \
                or (conditions_on_exiting_basis != None and \
                     conditions_on_exiting_basis(i, solving_mat)):
                    continue # this variable does not contribute to final answer

                if B_i / row_val < amount_upper_bound:
                    amount_upper_bound = B_i / row_val
                    amount_upper_bound_pos = i

            if amount_upper_bound == float('inf'):
                break

            for i in range(col_len - 1):
                if def_pos[i] == amount_upper_bound_pos:
                    def_pos[i] = -1
                    break # basic's definition in this row is cleared

            def_pos[min_pos] = amount_upper_bound_pos # chosen has definition in this row

            solving_mat[amount_upper_bound_pos, :] /= \
                solving_mat[amount_upper_bound_pos, min_pos]
            
            # do row transformation to move chosen non-basic variable to basis

            for i in range(row_len):
                if i == amount_upper_bound_pos:
                    continue # ignore row with definition of newly allocated basic variable
                
                factor = solving_mat[i, min_pos]
                solving_mat[i, :] = \
                    solving_mat[i, :] - solving_mat[amount_upper_bound_pos, :] * factor
            
            if abs(max_cost - solving_mat[-1, -1]) < 1e-5:
                if abs(solving_mat[-1, -1] - last_repeat) < 1e-5:
                    patience += 1
                else:
                    last_repeat = solving_mat[-1, -1]
                    patience = 0

            max_cost = max(max_cost, solving_mat[-1, -1])
            if patience == patience_level:
                break

    def linear_solver(
            A, 
            B, 
            C,
            optimized_X = None, 
            use_phase_2=True,
            conditions_entering_basis=None,
            conditions_exiting_basis=None):
        M, N = A.shape

        # construct solving mat
        # (M + 2) equations, including M equations from A, 2 last equation involving 
        # non-basic vars and detailed sum of artificial vars
        # (M + N + x + 1) coefficients, including b entries in the last column, given, slacks and artificials
        artificial_cnt = np.sum(B < 0)
        solving_mat = np.zeros(
            (M + (2 if use_phase_2 else 1), M + N + artificial_cnt + 1))

        solving_mat[:M, :N] = A
        
        def_pos = [-1] * (M + N + artificial_cnt)

        artificial_index = 0
        for i in range(M):
            solving_mat[i, N + i] = 1
            solving_mat[i, -1] = B[i]

            if B[i] < 0:
                solving_mat[i, :] = -solving_mat[i, :]
                solving_mat[i, M + N + artificial_index] = 1

                solving_mat[-1, :N] += -solving_mat[i, :N]
                solving_mat[-1, N + i] = 1
                solving_mat[-1, -1] += -solving_mat[i, -1]

                def_pos[M + N + artificial_index] = i # artificial is now the basic
                artificial_index += 1
            else:
                def_pos[N + i] = i # slack is now the basic

        if use_phase_2:
            solving_mat[-2, :N] = -C[:]
            Solver.solve(
                solving_mat, 
                def_pos,
                ignore_next_to_last=True,
                conditions_on_entering_basis=conditions_entering_basis,
                conditions_on_exiting_basis=conditions_exiting_basis)
        else:
            Solver.solve(
                solving_mat, 
                def_pos,
                ignore_next_to_last=False,
                conditions_on_entering_basis=conditions_entering_basis,
                conditions_on_exiting_basis=conditions_exiting_basis)            

        if use_phase_2:
            solving_mat = np.delete(
                solving_mat,
                [(-2 - _) for _ in range(artificial_cnt)], 
                axis=1)[:-1]
            Solver.solve(
                solving_mat, 
                def_pos,
                conditions_on_entering_basis=conditions_entering_basis,
                conditions_on_exiting_basis=conditions_exiting_basis)

        if optimized_X != None:
            for j in range(N):
                # climb up vertically to find its value, in basis
                def_pos_j = def_pos[j]
                if def_pos_j == -1:
                    optimized_X[j] = 0
                else:
                    optimized_X[j] = solving_mat[def_pos_j, -1]
        
        return solving_mat[-1][-1] # max value of the problem
    
    def min_quadratic_solver(Q, A, B, C, X, Y, optimized_X = None):
        A = np.concatenate((A, X, -X))
        B = np.concatenate((B, Y, -Y))

        m, n = A.shape

        linear_A = np.concatenate(
            (np.concatenate((-Q, A)),
            np.concatenate((-A.T, np.zeros((m, m))))), 
            axis=-1)
        linear_B = np.concatenate((C, B), axis=-1)

        extended_X = [0] * (m + n)

        def conditions_on_entering_basis(col_index, mat, def_pos):
            partner = -1
            if col_index < m + n:
                partner = col_index + (m + n)
            elif col_index < 2 * (m + n):
                partner = col_index - (m + n)
            else:
                return True

            _def_pos = def_pos[partner]
            if _def_pos == -1: # outside basis
                return True
            
            return False
        
            # col_len = len(mat[:, col_index])
            # min_ratio = float('inf')
            # min_ratio_pos = -1

            # for j in range(col_len - 1):
            #     if mat[j, col_index] <= 0:
            #         continue
                
            #     ratio = mat[j, -1] / mat[j, col_index]
            #     if min_ratio > ratio:
            #         min_ratio = ratio
            #         min_ratio_pos = j

            # # other case is when it goes to basis and its parter goes out
            # return min_ratio_pos == _def_pos

        sum = Solver.linear_solver(
            linear_A,
            linear_B, 
            None,
            optimized_X=extended_X,
            use_phase_2=False,
            conditions_entering_basis=conditions_on_entering_basis)

        if optimized_X != None:
            optimized_X[:] = extended_X[:n]

        found_X = np.array(extended_X[:n])

        return \
            (0.5 * found_X.T.dot(Q).dot(found_X) + C.T.dot(found_X)),\
            (np.isclose(sum, 0))