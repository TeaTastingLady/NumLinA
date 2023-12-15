from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from block_matrix import BlockMatrix
from linear_solvers import solve_lu
from poisson_problem import compute_error, error_plot, rhs

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

def experiment_error(n_list: list[list]):
    start_experiment = time() # stop time for calculations

    error_values_list = []
    for d in range(1,4):
        error_values = []
        for n in n_list[d-1]:
            b = rhs(d, n, f_func)
            # print(f"{idx+1} b calculated")
            mat = BlockMatrix(d, n)
            # print(f"{idx+1} mat calculated")
            p, l, u = mat.get_lu()
            # print(f"{idx+1} plu calculated")
            hat_u = solve_lu(p, l, u, b)
            # print(f"{idx+1} hat_u calculated")
            error_values.append(compute_error(d, n, hat_u, u_func))

        error_values_list.append(error_values)
        print(f"completed dimension set {d}")
    
    end_experiment = time() # stop time for calculations
    print(f"error experiment calculation time: {end_experiment-start_experiment}s")

    error_plot(n_list, error_values_list)

def experiment_condition(n_list: list[list]):
    start_experiment = time() # stop time for calculations

    plot_colors = ["blue", "green", "red", "magenta", "cyan"]
    for d in range(1, 4):
        cond_values = [matrix_condition(d, n_list[d-1][idx]) for idx in range(len(n_list[d-1]))]
        N_values = [(n_list[d-1][idx] - 1) ** d for idx in range(len(n_list[d-1]))]
        plt.plot(
            N_values,
            cond_values,
            marker="o",
            linestyle="-",
            markersize=4,
            color=plot_colors[d-1],
            label=f"Dimension {d}",
        )
        print(f"completed dimension set {d}")

    end_experiment = time() # stop time for calculations
    print(f"condition experiment calculation time: {end_experiment-start_experiment}s")

    plt.title("matrix condition")
    plt.xlabel("N values depending on n for fixed dimensions")
    plt.ylabel("matrix condition")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
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

def experiment_solution_heatmap(n):
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


def experiment_solution_plot(n):
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
    # n values for dimension 1 to 3
    MAX_N3_error = 22 # max value for dimension 3 and automatically adjusting lists for dimension 1 and 2
    N_LIST1_error = list(range(2, MAX_N3_error)) + list(np.unique(np.round(np.logspace(np.log10(MAX_N3_error),np.log10(MAX_N3_error**3), num=6)).astype(int)))[1:]
    print(f"N_LIST1_error: {N_LIST1_error}")
    N_LIST2_error = list(range(2, MAX_N3_error)) + list(np.unique(np.round(np.logspace(np.log10(MAX_N3_error),np.log10(np.sqrt(MAX_N3_error**3)), num=6)).astype(int)))[1:]
    print(f"N_LIST2_error: {N_LIST2_error}")
    N_LIST3_error = list(range(2, MAX_N3_error))
    print(f"N_LIST3_error: {N_LIST3_error}")

    # error plot
    experiment_error([N_LIST1_error, N_LIST2_error, N_LIST3_error])

    # n values for dimension 1 to 3
    MAX_N3_condition = 16 # max value for dimension 3 and automatically adjusting lists for dimension 1 and 2
    N_LIST1_condition = list(range(2, MAX_N3_condition)) + list(np.unique(np.round(np.logspace(np.log10(MAX_N3_condition),np.log10(MAX_N3_condition**3), num=6)).astype(int)))[1:]
    print(f"N_LIST1_condition: {N_LIST1_condition}")
    N_LIST2_condition = list(range(2, MAX_N3_condition)) + list(np.unique(np.round(np.logspace(np.log10(MAX_N3_condition),np.log10(np.sqrt(MAX_N3_condition**3)), num=6)).astype(int)))[1:]
    print(f"N_LIST2_condition: {N_LIST2_condition}")
    N_LIST3_condition = list(range(2, MAX_N3_condition))
    print(f"N_LIST3_condition: {N_LIST3_condition}")

    # condition plot
    # Note: Die Dimension 1 muss am schnellsten steigen, da immer gilt,
    # dass für das gleich n und d beliebig die Kondition fast gleich ist.
    # Da wir jedoch auf der x-Achse N haben, steigt die Kondition bei d=1 am schnellsten.
    experiment_condition([N_LIST1_condition, N_LIST2_condition, N_LIST3_condition])

    # solution plots
    experiment_solution_plot(20)
    experiment_solution_heatmap(20)
