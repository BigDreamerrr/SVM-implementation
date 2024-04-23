#pragma once
#include"Python.h"
#include"arrayobject.h"
#include<stdbool.h>

typedef struct {
    int row_len;
    int col_len;
    double** solving_mat;
} Solver;

__declspec(dllexport) double Py_LPSolver(PyObject* A, PyObject* B, PyObject* C, PyObject* X);
__declspec(dllexport) double Py_QPSolver(PyObject* Q, PyObject* A, PyObject* B, PyObject* C, PyObject* X, PyObject* Y, PyObject* Opti);