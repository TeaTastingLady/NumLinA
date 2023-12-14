import numpy as np
from numpy import linalg as LA
from typing import Callable
from poisson_problem import error_plot, rhs
from linear_solvers import solve_lu
from block_matrix import BlockMatrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from linear_solvers import solve_lu

N_LIST = list(range(2, 20))
D_LIST = list(range(1,5))
N_FIXED = 5
D_FIXED = 3
KAPPA = 1.5

def u_func(x: np.ndarray, kappa: int = KAPPA) -> float:
    d = len(x)
    return np.prod([x[l] * np.sin(kappa * np.pi * x[l]) for l in range(d)])

def term_1(x: float, kappa: int = KAPPA) -> float:
    return -2 * kappa * np.pi * np.cos(kappa * np.pi * x) + kappa**2 * np.pi**2 * x * np.sin(kappa * np.pi * x)
    
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

def matrix_condition(d: int, n: int) -> float:
    A = BlockMatrix(d,n).get_sparse().toarray()
    return LA.cond(A)

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

def experiment_condition(d_list,n_list):
    cond_values = [matrix_condition(d_list[idx], n_list[idx]) for idx in range(len(d_list))]
    print(cond_values)
    N_values = [(n_list[idx] - 1) ** d_list[idx] for idx in range(len(d_list))]
    print(N_values)
    plt.plot(
        N_values,
        cond_values,
        marker="o",
        linestyle="-",
        markersize=4,
        color="red",
    )
    plt.title("matrix condition")
    plt.xlabel("N values")
    plt.ylabel("matrix condition")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    plt.show()

def get_solutions(n, d):
    mat = BlockMatrix(d,n).get_lu()
    b = rhs(d,n,f_func)
    hat_u = solve_lu(*mat, b)
    u_vec = rhs(d,n, u_func)
    # print(f"b = {b}")
    # print(f"A*u = {BlockMatrix(d,n).get_sparse().toarray().dot(u_vec)}")
    # print(f"hat_u = {hat_u}")
    # print(f"u_vec = {u_vec}")
    return u_vec, hat_u

def experiment_solution(n):
    # reshape vectors to matrix (n-1) x (n-1)
    # matrix to heatmap
    d = 2
    u_vec, hat_u = get_solutions(n, d)
    u_vec_mat = u_vec.reshape((n-1, n-1))
    hat_u_mat = hat_u.reshape((n-1, n-1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(u_vec_mat, cmap='viridis', interpolation='nearest')
    ax1.set_title('Exact solution')
    fig.colorbar(im1, ax=ax1, orientation='vertical')
    im2 = ax2.imshow(hat_u_mat, cmap='plasma', interpolation='nearest')
    ax2.set_title('Approximation')
    fig.colorbar(im2, ax=ax2, orientation='vertical')

    plt.show()


def experiment_solution2(n):
    d = 2
    u_vec, hat_u = get_solutions(n, d)
    u_vec_mat = u_vec.reshape((n-1,n-1))
    hat_u_mat = hat_u.reshape((n-1,n-1))

    # Create x and y coordinates for u_vec_mat
    x1 = np.arange(u_vec_mat.shape[1])
    y1 = np.arange(u_vec_mat.shape[0])
    x1, y1 = np.meshgrid(x1, y1)

    # Create x and y coordinates for hat_u_mat
    x2 = np.arange(hat_u_mat.shape[1])
    y2 = np.arange(hat_u_mat.shape[0])
    x2, y2 = np.meshgrid(x2, y2)

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 6))


    # Plot for u_vec_mat
    ax1 = fig.add_subplot(121, projection='3d')  # 1 row, 2 columns, 1st subplot
    ax1.plot_surface(x1, y1, u_vec_mat, cmap='viridis')
    ax1.set_title('Exact solution')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')
    ax1.set_zlabel('Z Axis')

    # Plot for matrix2
    ax2 = fig.add_subplot(122, projection='3d')  # 1 row, 2 columns, 2nd subplot
    ax2.plot_surface(x2, y2, hat_u_mat, cmap='plasma')
    ax2.set_title('Approximation')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Y Axis')
    ax2.set_zlabel('Z Axis')

    # Show plot
    plt.show()

if __name__ == "__main__":
    #experiment_error([D_LIST], [[N_FIXED] * len(D_LIST)])
    # experiment_error([[1] * len(N_LIST), [2] * len(N_LIST), [3] * len(N_LIST)], [N_LIST] * 3)
    # print(matrix_condition(2,4))
    # print(matrix_condition(3,4))
    # print(matrix_condition(2,5))
    # print(matrix_condition(2,6))
    #experiment_condition([1]*10,list(range(2,12)))
    #experiment_condition([2]*10,list(range(2,12)))
    #experiment_condition([3]*10,list(range(2,12)))
    #experiment_condition([1,2,3,4],[5,5,5,5])
    # mat = BlockMatrix(2,4).get_lu()
    # b = rhs(2,4,f_func)
    # print(solve_lu(*mat, b))
    experiment_solution(20)
    experiment_solution2(20)
