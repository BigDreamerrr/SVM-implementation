#include"SVM.h"

Solver* create_solver(int row_len, int col_len) {
    Solver* solver = (Solver*)malloc(sizeof(Solver));
    solver->row_len = row_len;
    solver->col_len = col_len;
    solver->solving_mat = (double**)malloc(row_len * sizeof(double*));
    for (int i = 0; i < row_len; i++) {
        solver->solving_mat[i] = (double*)malloc(col_len * sizeof(double));
        memset(solver->solving_mat[i], 0, col_len * sizeof(double));
    }

    return solver;
}

void destroy_solver(Solver* solver) {
    for (int i = 0; i < solver->row_len; i++) {
        free(solver->solving_mat[i]);
    }
    free(solver->solving_mat);
    free(solver);
}

void print_mat(Solver* solver) {
    printf("\n\n");
    for (int i = 0; i < solver->row_len; i++) {
        for (int j = 0; j < solver->col_len; j++) {
            printf("%f ", solver->solving_mat[i][j]);
        }

        printf("\n");
    }
}

void solve(
    Solver* solver,
    int* def_pos,
    int ignore_next_to_last,
    int (*conditions_on_entering_basis)(int, double**, int*),
    int (*conditions_on_exiting_basis)(int, double**),
    int patience_level) {

    int row_len = solver->row_len;
    int col_len = solver->col_len;

    double max_cost = -INFINITY;
    double last_repeat = -1;
    int patience = 0;

    while (1) {
        int min_pos = -1;

        if (conditions_on_entering_basis == NULL) {
            min_pos = 0;
            for (int j = 1; j < col_len - 1; j++) {
                if (solver->solving_mat[row_len - 1][j] < solver->solving_mat[row_len - 1][min_pos]) {
                    min_pos = j;
                }
            }
        }
        else {
            double min_coeff = INFINITY;
            for (int j = 0; j < col_len - 1; j++) {
                if (def_pos[j] == -1 && conditions_on_entering_basis(j, solver->solving_mat, def_pos)) {
                    if (solver->solving_mat[row_len - 1][j] < min_coeff) {
                        min_coeff = solver->solving_mat[row_len - 1][j];
                        min_pos = j;
                    }
                }
            }
        }

        double min_coeff = INFINITY;

        for (int i = 0; i < col_len; i++) {
            min_coeff = min(min_coeff, solver->solving_mat[row_len - 1][i]);
        }

        if (fabs(min_coeff) < 1e-5 || min_coeff > 0) {
            break;
        }

        int amount_upper_bound_pos = -1;
        double amount_upper_bound = INFINITY;

        for (int i = 0; i < row_len - (2 * ignore_next_to_last); i++) {
            double B_i = solver->solving_mat[i][col_len - 1];
            double row_val = solver->solving_mat[i][min_pos];

            if (fabs(row_val) < 1e-5 || row_val < 0 || (conditions_on_exiting_basis != NULL && conditions_on_exiting_basis(i, solver->solving_mat))) {
                continue;
            }

            if (B_i / row_val < amount_upper_bound) {
                amount_upper_bound = B_i / row_val;
                amount_upper_bound_pos = i;
            }
        }

        if (amount_upper_bound == INFINITY) {
            break;
        }

        for (int i = 0; i < col_len - 1; i++) {
            if (def_pos[i] == amount_upper_bound_pos) {
                def_pos[i] = -1;
                break;
            }
        }

        def_pos[min_pos] = amount_upper_bound_pos;

        double scale = solver->solving_mat[amount_upper_bound_pos][min_pos];

        for (int j = 0; j < col_len; j++) {
            solver->solving_mat[amount_upper_bound_pos][j] /= scale;
        }

        for (int i = 0; i < row_len; i++) {
            if (i == amount_upper_bound_pos) {
                continue;
            }

            double factor = solver->solving_mat[i][min_pos];
            for (int j = 0; j < col_len; j++) {
                solver->solving_mat[i][j] -= solver->solving_mat[amount_upper_bound_pos][j] * factor;
            }
        }

        if (fabs(max_cost - solver->solving_mat[row_len - 1][col_len - 1]) < 1e-5) {
            if (fabs(solver->solving_mat[row_len - 1][col_len - 1] - last_repeat) < 1e-5) {
                patience++;
            }
            else {
                last_repeat = solver->solving_mat[row_len - 1][col_len - 1];
                patience = 0;
            }
        }

        max_cost = fmax(max_cost, solver->solving_mat[row_len - 1][col_len - 1]);
        if (patience == patience_level) {
            break;
        }
    }
}

double linear_solver(
    int M,
    int N,
    double** A,
    double* B,
    double* C,
    double* optimized_X,
    int use_phase_2,
    int (*conditions_entering_basis)(int, double**, int*),
    int (*conditions_exiting_basis)(int, double**)) {
    int artificial_cnt = 0;
    for (int i = 0; i < M; i++) {
        if (B[i] < 0) {
            artificial_cnt++;
        }
    }

    int col_len = M + N + artificial_cnt + 1;
    int row_len = M + (use_phase_2 ? 2 : 1);

    Solver* solver = create_solver(row_len, col_len);

    int* def_pos = malloc(sizeof(int) * (M + N + artificial_cnt));
    int artificial_index = 0;

    for (int i = 0; i < M + N + artificial_cnt; i++) {
        def_pos[i] = -1;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            solver->solving_mat[i][j] = A[i][j];
        }

        solver->solving_mat[i][N + i] = 1;
        solver->solving_mat[i][col_len - 1] = B[i];

        if (B[i] < 0) {
            for (int j = 0; j < col_len; j++) {
                solver->solving_mat[i][j] = -solver->solving_mat[i][j];
            }
            solver->solving_mat[i][M + N + artificial_cnt] = 1;

            for (int j = 0; j < N; j++) {
                solver->solving_mat[row_len - 1][j] += -solver->solving_mat[i][j];
            }
            solver->solving_mat[row_len - 1][N + i] = 1;
            solver->solving_mat[row_len - 1][col_len - 1] += -solver->solving_mat[i][col_len - 1];

            def_pos[M + N + artificial_index] = i;
            artificial_index++;
        }
        else {
            def_pos[N + i] = i;
        }
    }

    if (use_phase_2) {
        for (int j = 0; j < N; j++) {
            solver->solving_mat[row_len - 2][j] = -C[j];
        }
        solve(
            solver,
            def_pos,
            1,
            conditions_entering_basis,
            conditions_exiting_basis, 10);
    }
    else {
        solve(
            solver,
            def_pos,
            0,
            conditions_entering_basis,
            conditions_exiting_basis, 10);
    }

    if (use_phase_2) {
        // row_len - 2 -> row_len - 1 - artificial_cnt
        int last_col_containing_artificial = col_len - 1 - artificial_cnt;

        // delete all artificals and last row
        for (int i = 0; i < solver->row_len; i++) {
            double temp = solver->solving_mat[i][col_len - 1];
            solver->solving_mat[i][col_len - 1] = solver->solving_mat[i][last_col_containing_artificial];
            solver->solving_mat[i][last_col_containing_artificial] = temp;
        }

        solver->col_len -= artificial_cnt;
        free(solver->solving_mat[row_len - 1]);
        solver->row_len--;

        solve(solver, def_pos, false, conditions_entering_basis, conditions_exiting_basis, 10);
    }

    if (optimized_X != NULL) {
        for (int j = 0; j < N; j++) {
            int def_pos_j = def_pos[j];
            if (def_pos_j == -1) {
                optimized_X[j] = 0;
            }
            else {
                optimized_X[j] = solver->solving_mat[def_pos_j][col_len - 1];
            }
        }
    }

    double result = solver->solving_mat[solver->row_len - 1][solver->col_len - 1];

    destroy_solver(solver);
    free(def_pos);

    return result;
}

int quadra_m;
int quadra_n;

int quadra_conditions_on_entering_basis(int col_index, double** mat, int* def_pos) {
    int partner = -1;
    if (col_index < quadra_m + quadra_n) {
        partner = col_index + (quadra_m + quadra_n);
    }
    else if (col_index < 2 * (quadra_m + quadra_n)) {
        partner = col_index - (quadra_m + quadra_n);
    }
    else {
        return 1;
    }

    int def_pos_partner = def_pos[partner];
    if (def_pos_partner == -1) {
        return 1;
    }

    return 0;
}

double min_quadratic_solver(
    int A_shape_0,
    int A_shape_1,
    int X_shape_0,
    double** Q,
    double** A,
    double* B,
    double* C,
    double** X,
    double* Y,
    double* optimized_X) {
    quadra_m = A_shape_0;
    quadra_n = A_shape_1;

    double** new_A = NULL, * new_B = NULL;
    if (X != NULL) {
        int old_row_num = quadra_m;
        quadra_m += 2 * X_shape_0;

        // stack A with X, -X
        // stack B with Y, -Y

        new_A = malloc(quadra_m * sizeof(double*));
        new_B = malloc(quadra_m * sizeof(double));

        memcpy(new_B, B, old_row_num * sizeof(double));
        memcpy(new_B + old_row_num, Y, X_shape_0 * sizeof(double));

        for (int i = old_row_num + X_shape_0; i < old_row_num + 2 * X_shape_0; i++) {
            new_B[i] = -Y[i - old_row_num - X_shape_0]; // in this area, new_B = -Y
        }

        for (int i = 0; i < quadra_m; i++) {
            new_A[i] = malloc(quadra_n * sizeof(double));

            if (i < old_row_num) {
                memcpy(new_A[i], A[i], quadra_n * sizeof(double));
            }
            else if (i < old_row_num + X_shape_0) {
                memcpy(new_A[i], X[i - old_row_num], quadra_n * sizeof(double));
            }
            else {
                for (int j = 0; j < quadra_n; j++) {
                    new_A[i][j] = -X[i - old_row_num - X_shape_0][j];
                }
            }
        }

        A = new_A;
        B = new_B;
    }

    double** linear_A = (double**)malloc((quadra_m + quadra_n) * sizeof(double*));
    for (int i = 0; i < quadra_m + quadra_n; i++) {
        linear_A[i] = (double*)malloc((quadra_m + quadra_n) * sizeof(double));
    }

    double* linear_B = (double*)malloc((quadra_m + quadra_n) * sizeof(double));

    /* N    |     M
    * =============
    *N|
    * |
    *=|
    * |
    *M|============
    */

    for (int i = 0; i < quadra_m + quadra_n; i++) {
        for (int j = 0; j < quadra_m + quadra_n; j++) {
            if (i < A_shape_1) {
                if (j < A_shape_1) {
                    linear_A[i][j] = -Q[i][j];
                }
                else {
                    linear_A[i][j] = -A[j - A_shape_1][i];
                }
            }
            else if (j < A_shape_1) {
                linear_A[i][j] = A[i - A_shape_1][j];
            }
            else {
                linear_A[i][j] = 0; // fill 0 in this area
            }
        }
        if (i < A_shape_1) {
            linear_B[i] = C[i];
        }
        else {
            linear_B[i] = B[i - A_shape_1];
        }
    }

    double* extended_X = (double*)malloc((quadra_m + quadra_n) * sizeof(double));

    linear_solver(
        (quadra_m + quadra_n),
        (quadra_m + quadra_n),
        linear_A,
        linear_B,
        NULL,
        extended_X,
        0,
        quadra_conditions_on_entering_basis,
        NULL);

    if (optimized_X != NULL) {
        for (int i = 0; i < quadra_n; i++) {
            optimized_X[i] = extended_X[i];
        }
    }

    double* found_X = (double*)malloc(quadra_n * sizeof(double));
    for (int i = 0; i < quadra_n; i++) {
        found_X[i] = extended_X[i];
    }

    //double result = 0.5 * found_X.T.dot(Q).dot(found_X) + C.T.dot(found_X);
    // (1, n) x (n x n) x (n x 1) + (1, n) x (n, 1)

    double result = 0;

    for (int j = 0; j < quadra_n; j++) {
        double vec_term = 0;
        for (int i = 0; i < quadra_n; i++) {
            vec_term += found_X[i] * Q[i][j];
        }

        result += found_X[j] * (0.5 * vec_term + C[j]);
    }

    for (int i = 0; i < quadra_m + quadra_n; i++) {
        free(linear_A[i]);
    }

    free(linear_A);
    free(linear_B);
    free(extended_X);
    free(found_X);

    if (new_A != NULL) {
        for (int i = 0; i < quadra_m; i++) {
            free(new_A[i]); // free each row
        }

        free(new_A);
        free(new_B);
    }

    return result;
}

double Py_LPSolver(PyObject* A, PyObject* B, PyObject* C, PyObject* X) {
    double** A_arr, * B_arr, * C_arr;

    PyArray_AsCArray(&A, &A_arr, PyArray_DIMS(A), 2, NULL);
    PyArray_AsCArray(&B, &B_arr, PyArray_DIMS(B), 1, NULL);
    PyArray_AsCArray(&C, &C_arr, PyArray_DIMS(C), 1, NULL);

    double* C_X = NULL;
    if (X != Py_None) {
        C_X = malloc(PyArray_DIMS(X)[0] * sizeof(double));
    }

    double ans = linear_solver(
        (int)PyArray_DIMS(A)[0],
        (int)PyArray_DIMS(A)[1],
        A_arr,
        B_arr,
        C_arr,
        C_X,
        true,
        NULL,
        NULL);

    printf("ans: %f\n", ans);

    PyArray_Free(A, A_arr);
    PyArray_Free(B, B_arr);
    PyArray_Free(C, C_arr);

    if (X != Py_None) {
        for (int i = 0; i < PyArray_DIMS(X)[0]; i++) {
            PyArray_SETITEM(X, PyArray_GETPTR1(X, i), PyFloat_FromDouble(C_X[i]));
        }
    }

    free(C_X);

    return ans;
}

double Py_QPSolver(PyObject* Q, PyObject* A, PyObject* B, PyObject* C, PyObject* X, PyObject* Y, PyObject* Opti) {
    double** Q_arr, ** A_arr, * C_arr, * B_arr, ** X_arr, * Y_arr;

    npy_intp* A_dims = PyArray_DIMS(A);

    PyArray_AsCArray(&Q, &Q_arr, PyArray_DIMS(Q), 2, NULL);
    PyArray_AsCArray(&A, &A_arr, PyArray_DIMS(A), 2, NULL);
    PyArray_AsCArray(&C, &C_arr, PyArray_DIMS(C), 1, NULL);
    PyArray_AsCArray(&B, &B_arr, PyArray_DIMS(B), 1, NULL);

    PyArray_AsCArray(&X, &X_arr, PyArray_DIMS(X), 2, NULL);
    PyArray_AsCArray(&Y, &Y_arr, PyArray_DIMS(Y), 1, NULL);

    double* opti = NULL;

    if (Opti != Py_None) {
        opti = malloc(PyArray_DIMS(Opti)[0] * sizeof(double));
    }

    double ans =
        min_quadratic_solver(
            (int)A_dims[0], 
            (int)A_dims[1], 
            (int)PyArray_DIMS(X)[0], Q_arr, A_arr, B_arr, C_arr, X_arr, Y_arr, opti);

    if (Opti != Py_None) {
        for (int i = 0; i < PyArray_DIMS(Opti)[0]; i++) {
            PyArray_SETITEM(Opti, PyArray_GETPTR1(Opti, i), PyFloat_FromDouble(opti[i]));
        }
    }

    free(opti);

    PyArray_Free(Q, Q_arr);
    PyArray_Free(A, A_arr);
    PyArray_Free(B, B_arr);
    PyArray_Free(C, C_arr);
    PyArray_Free(X, X_arr);
    PyArray_Free(Y, Y_arr);

    return ans;
}