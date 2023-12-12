import itertools
import logging

import numpy as np
import pytest

from block_matrix import BlockMatrix
from experiments_lu import f_func, u_func
from linear_solvers import solve_lu
from poisson_problem import compute_error, rhs

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def u_func2(x):
    return sum([xl**3 for xl in x])

def f_func2(x):
    return sum([6*xl for xl in x])

N_LIST2 = [2, 4]
D_LIST2 = [1, 1]
U_LIST2 = [u_func2, u_func2]
F_LIST2 = [f_func2, f_func2]
B_LIST2 = [[0.75], [0.09375, 0.1875, 0.28125]]
HAT_U_LIST2 = [[0.375], [0.234375, 0.375, 0.328125]]
ERROR_LIST2 = [0.34375, 0.3671875]

# @pytest.mark.parametrize("n, d, u, f, b_input, hat_u_input, error_input", itertools.product(N_LIST2, D_LIST2, U_LIST2, F_LIST2, B_LIST2, HAT_U_LIST2, ERROR_LIST2))
# def test_experiment_error(n, d, u, f, b_input, hat_u_input, error_input):
#     b = rhs(d, n, f)
#     logger.debug(f"b = {b}")
#     assert b == b_input, "b was calculated wrong"

#     mat = BlockMatrix(d, n)
#     p_mat, l_mat, u_mat = mat.get_lu()
#     hat_u = solve_lu(p_mat, l_mat, u_mat, b)
#     logger.debug(f"hat_u = {hat_u}")
#     assert hat_u == hat_u_input, "b was calculated wrong"

#     error = compute_error(d, n, hat_u, u)
#     logger.debug(f"error = {error}")

def approximate_laplacian(u, x, h=1e-5):
    d = len(x)
    laplacian = 0
    for i in range(d):
        x_plus_h = np.copy(x)
        x_plus_h[i] += h
        x_minus_h = np.copy(x)
        x_minus_h[i] -= h
        laplacian += (u(x_plus_h) - 2*u(x) + u(x_minus_h)) / h**2
    return laplacian

def test_poisson_solution():
    test_points = [np.random.rand(3) for _ in range(10)]  # Generate random test points

    for point in test_points:
        laplacian_u = approximate_laplacian(u_func, point)
        f_val = f_func(point)
        np.testing.assert_almost_equal(laplacian_u, f_val, decimal=5)
