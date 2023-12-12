import itertools

import numpy as np
import pytest

from block_matrix import BlockMatrix
from linear_solvers import solve_lu

D_LIST = [1]*4 + [2]*4 + [3]*4
N_LIST = [2,3,4,5]*3

@pytest.mark.parametrize("n, d", itertools.product(N_LIST, D_LIST))
def test_solve_lu(d, n):
    # Setup test data
    mat = BlockMatrix(d, n)
    A = mat.get_sparse().toarray()
    p, l, u = mat.get_lu()
    
    # with random vectors try 100 times
    for _ in range(100):
        b = np.random.rand((n-1)**d)

        # Expected output
        expected_x = np.linalg.solve(A, b)

        # Call the solve_lu function
        calculated_x = solve_lu(p, l, u, b)

        # Assert that the calculated solution is close to the expected solution
        np.testing.assert_allclose(calculated_x, expected_x, rtol=1e-13)
