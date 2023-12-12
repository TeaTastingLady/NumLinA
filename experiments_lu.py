from typing import Callable

import numpy as np

from block_matrix import BlockMatrix
from linear_solvers import solve_lu
from poisson_problem import error_plot, rhs

N_LIST = list(range(2, 20))
D_LIST = list(range(1,5))
N_FIXED = 5
D_FIXED = 3
KAPPA = 3

def u_func(x: np.ndarray, kappa: int = KAPPA) -> float:
    d = len(x)
    return np.prod([x[l] * np.sin(kappa * np.pi * x[l]) for l in range(d)])

def term_1(x: float, kappa: int = KAPPA) -> float:
    return 2 * kappa * np.pi * np.cos(kappa * np.pi * x) - kappa**2 * np.pi**2 * x * np.sin(kappa * np.pi * x)
    
def f_func(x: np.ndarray) -> float:
    sum = 0
    for i in range(len(x)):
        if i == len(x)-1:
            sum += term_1(x[i]) * u_func(x[:i])
        elif i == 0:
            sum += term_1(x[i]) * u_func(x[i+1:])
        else:
            sum += term_1(x[i]) * u_func(np.concatenate((x[:i],x[i+1:])))
    return sum

def experiment_error(d_list: list[list], n_list: list[list], y_log_scale: bool = True):
    hat_u_list_total = []
    for set_idx in range(len(d_list)):
        hat_u_list = []
        for idx in range(len(d_list[set_idx])):
            b = rhs(d_list[set_idx][idx], n_list[set_idx][idx], f_func)
            # print(f"{idx+1} b calculated")
            mat = BlockMatrix(d_list[set_idx][idx], n_list[set_idx][idx])
            # print(f"{idx+1} mat calculated")
            p, l, u = mat.get_lu()
            # print(f"{idx+1} plu calculated")
            hat_u = solve_lu(p, l, u, b)
            # print(f"{idx+1} hat_u calculated")
            hat_u_list.append(hat_u)
        hat_u_list_total.append(hat_u_list)
        print(f"completed set {set_idx + 1}")

    error_plot(d_list, n_list, hat_u_list_total, u_func, y_log_scale=y_log_scale)

if __name__ == "__main__":
    experiment_error([D_LIST], [[N_FIXED] * len(D_LIST)])
    experiment_error([[1] * len(N_LIST), [2] * len(N_LIST), [3] * len(N_LIST)], [N_LIST] * 3)
