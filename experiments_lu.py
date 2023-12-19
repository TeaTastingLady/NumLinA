"""
Author: Karla Menze, Laura Faustmann
Date: 20/12/2023
"""
from time import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

from block_matrix import BlockMatrix
from linear_solvers import solve_lu
from poisson_problem import compute_error, error_plot, func_to_vec, rhs

KAPPA = 1


def u_func(x: np.ndarray, kappa: int = KAPPA) -> float:  # pylint: disable=invalid-name
    """
    example solution function

    Parameters
    ----------
    x : np.ndarray
        x point coordinates
    kappa : int
        flexible parameter of example function

    Returns
    -------
    float
        function value of coordinates
    """
    return np.prod([x[l] * np.sin(kappa * np.pi * x[l]) for l in range(len(x))])


def term_1(x: float, kappa: int = KAPPA) -> float:  # pylint: disable=invalid-name
    """
    helper term for ``f_func``

    Parameters
    ----------
    x : np.ndarray
        x point coordinates
    kappa : int
        flexible parameter of example function

    Returns
    -------
    float
        function value of coordinates
    """
    return -2 * kappa * np.pi * np.cos(
        kappa * np.pi * x
    ) + kappa**2 * np.pi**2 * x * np.sin(kappa * np.pi * x)


def f_func(x: np.ndarray) -> float:  # pylint: disable=invalid-name
    """
    right side for example ``u_func`` of poisson problem

    Parameters
    ----------
    x : np.ndarray
        x point coordinates

    Returns
    -------
    float
        function value of coordinates
    """
    sum_value = 0
    for i in range(len(x)):  # pylint: disable=consider-using-enumerate
        if i == len(x) - 1:
            sum_value += term_1(x[i]) * u_func(x[:i])
        elif i == 0:
            sum_value += term_1(x[i]) * u_func(x[i + 1 :])
        else:
            sum_value += term_1(x[i]) * u_func(np.concatenate((x[:i], x[i + 1 :])))
    return sum_value


def matrix_condition(d: int, n: int) -> float:  # pylint: disable=invalid-name
    """
    Function to calculate matrix condition of laplace operator approximation A

    Parameters
    ----------
    d : int
        dimsension of poisson problem
    n : int
        number of discrete points per dimension

    Returns
    -------
    float
        matrix condition of A
    """
    A_mat = BlockMatrix(d, n).get_sparse().toarray()  # pylint: disable=invalid-name
    return LA.cond(A_mat)


def experiment_error(n_list: list[list]):
    """
    Experiment function to generate error plot

    Parameters
    ----------
    n_list : list[list]
        list with three lists that contain the n values for the dimension d = 1 to 3

    Returns
    -------
    shows plot
    """
    start_experiment = time()  # stop time for calculations

    error_values_list = []
    for d in range(1, 4):  # pylint: disable=invalid-name
        error_values = []
        for n in n_list[d - 1]:  # pylint: disable=invalid-name
            b_vec = rhs(d, n, f_func)
            # print(f"{idx+1} b calculated")
            mat = BlockMatrix(d, n)
            # print(f"{idx+1} mat calculated")
            p_mat, l_mat, u_mat = mat.get_lu() # pylint: disable=unbalanced-tuple-unpacking
            # print(f"{idx+1} plu calculated")
            hat_u = solve_lu(p_mat, l_mat, u_mat, b_vec)
            # print(f"{idx+1} hat_u calculated")
            error_values.append(compute_error(d, n, hat_u, u_func))

        error_values_list.append(error_values)
        print(f"completed dimension set {d}")

    end_experiment = time()  # stop time for calculations
    print(f"error experiment calculation time: {end_experiment-start_experiment}s")

    error_plot(n_list, error_values_list)


def experiment_condition(n_list: list[list]):
    """
    Experiment function to generate matrix condition plot

    Parameters
    ----------
    n_list : list[list]
        list with three lists that contain the n values for the dimension d = 1 to 3

    Returns
    -------
    shows plot
    """
    start_experiment = time()  # stop time for calculations

    plot_colors = ["blue", "green", "red", "magenta", "cyan"]
    for d in range(1, 4):  # pylint: disable=invalid-name
        cond_values = [
            matrix_condition(d, n_list[d - 1][idx]) for idx in range(len(n_list[d - 1]))
        ]
        N_values = [  # pylint: disable=invalid-name
            (n_list[d - 1][idx] - 1) ** d for idx in range(len(n_list[d - 1]))
        ]
        plt.plot(
            N_values,
            cond_values,
            marker="o",
            linestyle="-",
            markersize=4,
            color=plot_colors[d - 1],
            label=f"Dimension {d}",
        )
        print(f"completed dimension set {d}")

    end_experiment = time()  # stop time for calculations
    print(f"condition experiment calculation time: {end_experiment-start_experiment}s")

    plt.title("matrix condition")
    plt.xlabel("N values depending on n for fixed dimensions")
    plt.ylabel("matrix condition")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_solutions(
    n: int, d: int  # pylint: disable=invalid-name
) -> tuple[np.ndarray, np.ndarray]:
    """
    Function to return the approximated and true solution

    Parameters
    ----------
    d : int
        dimsension of poisson problem
    n : int
        number of discrete points per dimension

    Returns
    -------
    u_vec : np.ndarray
        true solution vector
    hat_u : np.ndarray
        approximated solution vector
    """
    p_mat, l_mat, u_mat = BlockMatrix(d, n).get_lu() # pylint: disable=unbalanced-tuple-unpacking
    b_vec = rhs(d, n, f_func)
    hat_u = solve_lu(p_mat, l_mat, u_mat, b_vec)
    u_vec = func_to_vec(d, n, u_func)
    # print(f"b = {b}")
    # print(f"A*u = {mat.get_sparse().toarray().dot(u_vec)}")
    # print(f"hat_u = {hat_u}")
    # print(f"u_vec = {u_vec}")
    return u_vec, hat_u


def experiment_solution_heatmap(n: int):  # pylint: disable=invalid-name
    """
    Function to generate heatmap of true and approximate solution for given n and d=2

    Parameters
    ----------
    n : int
        number of discrete points per dimension

    Returns
    -------
    shows heatmap plot
    """
    d = 2  # pylint: disable=invalid-name
    u_vec, hat_u = get_solutions(n, d)
    u_vec_mat = u_vec.reshape((n - 1, n - 1))
    hat_u_mat = hat_u.reshape((n - 1, n - 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    im1 = ax1.imshow(u_vec_mat, cmap="viridis", interpolation="nearest")
    ax1.set_title("Exact solution")
    fig.colorbar(im1, ax=ax1, orientation="vertical")
    im2 = ax2.imshow(hat_u_mat, cmap="plasma", interpolation="nearest")
    ax2.set_title("Approximation")
    fig.colorbar(im2, ax=ax2, orientation="vertical")

    plt.show()


def experiment_solution_plot(n: int):  # pylint: disable=invalid-name
    """
    Function to generate 3d plot of true and approximate solution for given n and d=2

    Parameters
    ----------
    n : int
        number of discrete points per dimension

    Returns
    -------
    shows 3d plot
    """
    d = 2  # pylint: disable=invalid-name
    u_vec, hat_u = get_solutions(n, d)
    u_vec_mat = u_vec.reshape((n - 1, n - 1))
    hat_u_mat = hat_u.reshape((n - 1, n - 1))

    # Create x and y coordinates for u_vec_mat
    x_one = np.arange(u_vec_mat.shape[1])
    y_one = np.arange(u_vec_mat.shape[0])
    x_one, y_one = np.meshgrid(x_one, y_one)

    # Create x and y coordinates for hat_u_mat
    x_two = np.arange(hat_u_mat.shape[1])
    y_two = np.arange(hat_u_mat.shape[0])
    x_two, y_two = np.meshgrid(x_two, y_two)

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(8, 6))

    # Plot for u_vec_mat
    ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 columns, 1st subplot
    ax1.plot_surface(x_one, y_one, u_vec_mat, cmap="viridis")
    ax1.set_title("Exact solution")
    ax1.set_xlabel("X Axis")
    ax1.set_ylabel("Y Axis")
    ax1.set_zlabel("Z Axis")

    # Plot for matrix2
    ax2 = fig.add_subplot(122, projection="3d")  # 1 row, 2 columns, 2nd subplot
    ax2.plot_surface(x_two, y_two, hat_u_mat, cmap="plasma")
    ax2.set_title("Approximation")
    ax2.set_xlabel("X Axis")
    ax2.set_ylabel("Y Axis")
    ax2.set_zlabel("Z Axis")

    # Show plot
    plt.show()


def main():
    """main method of program that runs
    the different experiments"""
    # n values for dimension 1 to 3
    # max value for dimension 3 and
    # automatically adjusting lists
    # for dimension 1 and 2 (~15 min)
    MAX_N3_error = 30 # pylint: disable=invalid-name
    N_LIST1_error = ( # pylint: disable=invalid-name
        list(range(2, MAX_N3_error))
        + list(
            np.unique(
                np.round(
                    np.logspace(
                        np.log10(MAX_N3_error), np.log10(MAX_N3_error**3), num=6
                    )
                ).astype(int)
            )
        )[1:]
    )
    print(f"N_LIST1_error: {N_LIST1_error}")
    N_LIST2_error = ( # pylint: disable=invalid-name
        list(range(2, MAX_N3_error))
        + list(
            np.unique(
                np.round(
                    np.logspace(
                        np.log10(MAX_N3_error),
                        np.log10(np.sqrt(MAX_N3_error**3)),
                        num=6,
                    )
                ).astype(int)
            )
        )[1:]
    )
    print(f"N_LIST2_error: {N_LIST2_error}")
    N_LIST3_error = list(range(2, MAX_N3_error)) # pylint: disable=invalid-name
    print(f"N_LIST3_error: {N_LIST3_error}")

    # error plot
    experiment_error([N_LIST1_error, N_LIST2_error, N_LIST3_error])

    # n values for dimension 1 to 3
    # max value for dimension 3 and
    # automatically adjusting lists
    # for dimension 1 and 2
    MAX_N3_condition = 16 # pylint: disable=invalid-name
    N_LIST1_condition = ( # pylint: disable=invalid-name
        list(range(2, MAX_N3_condition))
        + list(
            np.unique(
                np.round(
                    np.logspace(
                        np.log10(MAX_N3_condition),
                        np.log10(MAX_N3_condition**3),
                        num=6,
                    )
                ).astype(int)
            )
        )[1:]
    )
    print(f"N_LIST1_condition: {N_LIST1_condition}")
    N_LIST2_condition = ( # pylint: disable=invalid-name
        list(range(2, MAX_N3_condition))
        + list(
            np.unique(
                np.round(
                    np.logspace(
                        np.log10(MAX_N3_condition),
                        np.log10(np.sqrt(MAX_N3_condition**3)),
                        num=6,
                    )
                ).astype(int)
            )
        )[1:]
    )
    print(f"N_LIST2_condition: {N_LIST2_condition}")
    N_LIST3_condition = list(range(2, MAX_N3_condition)) # pylint: disable=invalid-name
    print(f"N_LIST3_condition: {N_LIST3_condition}")

    # condition plot
    # Note: Die Dimension 1 muss am schnellsten steigen, da immer gilt,
    # dass für das gleich n und d beliebig die Kondition fast gleich ist.
    # Da wir jedoch auf der x-Achse N haben, steigt die Kondition bei d=1 am schnellsten.
    experiment_condition([N_LIST1_condition, N_LIST2_condition, N_LIST3_condition])

    # solution plots
    experiment_solution_plot(20)
    experiment_solution_heatmap(20)


if __name__ == "__main__":
    main()
