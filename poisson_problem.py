"""

Author: Laura Faustmann
Date: 28/11/2023
"""
import numpy as np
import matplotlib.pyplot as plt


def rhs(d, n, f): # pylint: disable=invalid-name
    """Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """
    if d < 1 or n < 2:
        raise ValueError(
            f"there was a wrong input parameter: d > 0 (BUT: {d} > 0) or n > 1 (BUT: {n} > 1)"
        )

    b = np.array([f([coord * 1/n for coord in inv_idx(m, d, n)]) * 1/n**2 for m in range(1, (n - 1) ** d + 1)]) # pylint: disable=invalid-name

    return b


def idx(nx, n): # pylint: disable=invalid-name
    """Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.

    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    d = len(nx) # pylint: disable=invalid-name
    sum_iterable = 1
    for i in range(d):
        sum_iterable += (nx[i] - 1) * (n - 1) ** i
    return sum_iterable


def inv_idx(m, d, n): # pylint: disable=invalid-name
    """Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.

    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    nx = [] # pylint: disable=invalid-name
    m_variable = m
    for i in range(d - 1):
        x_d = np.ceil(m_variable / (n - 1) ** (d - 1 - i))
        nx = [x_d] + nx # pylint: disable=invalid-name
        m_variable -= (x_d - 1) * (n - 1) ** (d - 1 - i)
    nx = [m_variable] + nx # pylint: disable=invalid-name
    return nx


def compute_error(d, n, hat_u, u): # pylint: disable=invalid-name, unused-argument
    """Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """
    u_vec = rhs(d,n,u)
    return np.max(np.abs(u_vec - hat_u))

def error_plot(d_list: list[list], n_list: list[list], hat_u_list: list[list], u, y_log_scale: bool):
    """Plots the error of the discrete solution hat_u compared
    to the exact solution u dependent on n and d.

    Parameters
    ----------
    d_list: list
    list of dimensions
    n_list: list
    list of number of partial intervals
    hat_u_list: list
    discrete solution
    u: callable
    exact solution

    Return
    ------
    shows plot
    """
    plot_colors = ["blue", "green", "red", "magenta", "cyan"]
    for set_idx in range(len(d_list)):
        error_values = [compute_error(d_list[set_idx][idx], n_list[set_idx][idx], hat_u_list[set_idx][idx], u) for idx in range(len(d_list[set_idx]))]
        N_values = [(n_list[set_idx][idx] - 1) ** d_list[set_idx][idx] for idx in range(len(d_list[set_idx]))]
        plt.plot(
            N_values,
            error_values,
            marker="o",
            linestyle="-",
            markersize=4,
            color=plot_colors[set_idx],
            label=f"plot {set_idx + 1}"
        )

    plt.title(f"error depending on N")

    plt.xlabel("N values")
    plt.ylabel("error")
    if y_log_scale:
        plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Y = [[1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 1, 2], [1, 2, 2], [1, 3, 2], [1, 1, 3], [1, 2, 3]]
    for coord in Y:
        print(coord, idx(coord, 4))
