"""
NumLinA Blatt1
PPI 08
Maximilian Krienitz ...
Laura Faustmann ...
"""
import matplotlib.pyplot as plt
import numpy as np


class FiniteDifference:
    """Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.
    Parameters
    ----------
    h : float
    Step size of the approximation.
    f : callable
    Function to approximate the derivatives of. The calling signature is
    ‘f(x)‘. Here ‘x‘ is a scalar or array_like of ‘numpy‘. The return
    value is of the same type as ‘x‘.
    d_f : callable, optional
    The analytic first derivative of ‘f‘ with the same signature.
    dd_f : callable, optional
    The analytic second derivative of ‘f‘ with the same signature..
    Attributes
    ----------
    h : float
    Step size of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):  # pylint: disable=invalid-name
        if h <= 0:  # pylint: disable=invalid-name
            raise ValueError("h muss größer 0 sein")
        self.h = h  # pylint: disable=invalid-name
        self.f = f  # pylint: disable=invalid-name
        self.d_f = d_f
        self.dd_f = dd_f

    def compute_dh_f(self):
        """Calculates the approximation for the first derivative of the f with step size h.
        Parameters
        ----------
        -
        Return
        ------
        callable
        Calculates the approximation of the first derivative for a given x.
        """

        def dh_f(x_value):
            x_p = x_value + self.h
            dh_f_x = (self.f(x_p) - self.f(x_value)) / self.h
            return dh_f_x

        return dh_f

    def compute_ddh_f(self):
        """Calculates the approximation for the second derivative of f with step size h.
        Parameters
        ----------
        -
        Return
        ------
        callable
        Calculates the approximation of the first derivative for a given x.
        """

        def ddh_f(x_value):
            x_p = x_value + self.h
            x_m = x_value - self.h
            ddh_f_x = (self.f(x_p) - 2 * self.f(x_value) + self.f(x_m)) / self.h**2
            return ddh_f_x

        return ddh_f

    def compute_errors(self, a, b, p):  # pylint: disable=invalid-name
        """Calculates the error of the first and second derivative of the f with step size h
        and gives it back as tuple.
        Parameters
        ----------
        a: float
        begin of the interval
        b: float
        end of the interval
        p: integer
        number of evaluation points used for the approximation
        Return
        ------
        callable
        """
        fehler1 = []
        dh_f = self.compute_dh_f()
        for i in range(p + 1):
            x_i = a + i * abs(b - a) / p
            dist = abs(self.d_f(x_i) - dh_f(x_i))
            fehler1.append(dist)

        e1_value = max(fehler1)

        fehler2 = []
        ddh_f = self.compute_ddh_f()
        for i in range(p + 1):
            x_i = a + i * abs(b - a) / p
            dist = abs(self.dd_f(x_i) - ddh_f(x_i))
            fehler2.append(dist)

        e2_value = max(fehler2)

        return (e1_value, e2_value)

    def func_plotten(self, a, b, p):  # pylint: disable=invalid-name
        """Plots f with its exact first and second derivative as well as the approximation of the
        first and second derivative on a given Interval [a,b].
        Parameters
        ----------
        a: float
        begin of the interval
        b: float
        end of the interval
        p: integer
        number of evaluation points used for the approximation
        Return
        ------
        callable
        """
        dh_f = self.compute_dh_f()
        ddh_f = self.compute_ddh_f()

        # berechne x_i
        # berechne f(x_i)
        werte_x_i = []
        werte_f = []
        werte_d_f = []
        werte_dd_f = []
        werte_dh_f = []
        werte_ddh_f = []
        for i in range(p + 1):
            x_i = a + i * abs(b - a) / p
            werte_x_i.append(x_i)
            werte_f.append(self.f(x_i))
            werte_d_f.append(self.d_f(x_i))
            werte_dd_f.append(self.dd_f(x_i))
            werte_dh_f.append(dh_f(x_i))
            werte_ddh_f.append(ddh_f(x_i))

        func_werte = []
        for i, _ in enumerate(werte_f):
            func_werte.append(
                [
                    werte_f[i],
                    werte_d_f[i],
                    werte_dd_f[i],
                    werte_dh_f[i],
                    werte_ddh_f[i],
                ]  # pylint: disable=unnecessary-list-index-lookup
            )

        plt.plot(werte_x_i, werte_f, "-", markersize=1, color="black", label="f(x_i)")
        plt.plot(
            werte_x_i, werte_d_f, "-", markersize=1, color="blue", label="d_f(x_i)"
        )
        plt.plot(
            werte_x_i, werte_dd_f, "-", markersize=1, color="darkred", label="dd_f(x_i)"
        )
        plt.plot(
            werte_x_i, werte_dh_f, "-", markersize=1, color="cyan", label="dh_f(x_i)"
        )
        plt.plot(
            werte_x_i, werte_ddh_f, "-", markersize=1, color="red", label="ddh_f(x_i)"
        )

        plt.legend()
        plt.show()


def error_plot(
    func_list, h_list, error_parameter, title="", h_line_on=True
):  # pylint: disable=invalid-name
    """Plots error of the approximation of the first and second derivative for a given collection of step sizes.
    Parameters
    ----------
    func_list: list
    list consisting of the f with its exact first and second derivative as well as its
    approximated first and second derivative
    h_list: list
    three evaluation points
    error_parameter: tuple
    parameters for error approximation (a, b, p)
    title: string
    title of the plot
    h_line_on: boolean
    plots h, h¹, h²
    Return
    ------
    callable
    """
    e1_colors = ["steelblue", "yellowgreen", "lightcoral", "plum", "paleturquoise"]
    e2_colors = ["blue", "green", "red", "magenta", "cyan"]

    werte_h2 = [h**2 for h in h_list]
    werte_h3 = [h**3 for h in h_list]
    if h_line_on:
        plt.plot(h_list, h_list, "-", markersize=4, color="black", label="h^1")
        plt.plot(h_list, werte_h2, "-", markersize=4, color="gray", label="h^2")
        plt.plot(h_list, werte_h3, "-", markersize=4, color="lightgray", label="h^3")

    for idx, func in enumerate(func_list):
        # func format: (f, d_f, dd_f, name)
        werte_e1 = []
        werte_e2 = []
        for h in h_list:  # pylint: disable=invalid-name
            fehler = FiniteDifference(h, *func[:3]).compute_errors(*error_parameter)
            werte_e1.append(fehler[0])
            werte_e2.append(fehler[1])

        plt.plot(
            h_list,
            werte_e1,
            "-",
            markersize=4,
            color=e1_colors[idx],
            label=f"e1 ({func[3]})",
        )
        plt.plot(
            h_list,
            werte_e2,
            "-",
            markersize=4,
            color=e2_colors[idx],
            label=f"e2 ({func[3]})",
        )

    plt.yscale("log")
    plt.xscale("log")
    plt.title(title)
    plt.legend()
    plt.show()


####### testing #######


def g1(x):  # pylint: disable=invalid-name
    """example function"""
    return np.sin(x) / x


def dg1(x):  # pylint: disable=invalid-name
    """example function first derivative"""
    return (x * np.cos(x) - np.sin(x)) / x**2


def ddg1(x):  # pylint: disable=invalid-name
    """example function second derivative"""
    return -((x**2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / x**3


def gk_creator(k):  # pylint: disable=invalid-name
    """example function creator depending on kappa"""

    def gk(x):  # pylint: disable=invalid-name
        return np.sin(k * x) / x

    return gk


def dgk_creator(k):  # pylint: disable=invalid-name
    """example function creator first derivative depending on kappa"""

    def dgk(x):  # pylint: disable=invalid-name
        return (k * np.cos(k * x)) / x - (np.sin(k * x)) / x**2

    return dgk


def ddgk_creator(k):  # pylint: disable=invalid-name
    """example function creator second derivative depending on kappa"""

    def ddgk(x):  # pylint: disable=invalid-name
        return (
            -((k**2 * x**2 - 2) * np.sin(k * x) + 2 * k * x * np.cos(k * x))
            / x**3
        )

    return ddgk


def main():
    """main function"""
    ### Aufgabe 1.1 ###
    print("### Aufgabe 1.1 ###")
    fin_diff = FiniteDifference(0.01, g1, dg1, ddg1)

    # showing compute_errors which uses compute_dh_f and compute_ddh_f
    fehler = fin_diff.compute_errors(np.pi, 3 * np.pi, 1000)
    print("Approximationerror (e1, e2): ", fehler)

    # showing plot of f, d_f, dd_f, dhf, and ddhf
    fin_diff.func_plotten(np.pi, 3 * np.pi, 1000)

    # showing errors for different h values
    error_plot(
        [(g1, dg1, ddg1, "g1")],
        [0.05, 0.25, 1.25],
        (np.pi, 3 * np.pi, 1000),
        title="Errorplot depending on h",
    )

    ### Aufgabe 1.2 ###
    print("### Aufgabe 1.2 ###")
    error_plot(
        [(g1, dg1, ddg1, "g1")],
        [np.pi / 3, np.pi / 4, np.pi / 5, np.pi / 10],
        (np.pi, 3 * np.pi, 1000),
        title="""Comparison of approximated and exact
        first and second derivative of sin(x)/x by error""",
    )

    # errorplots for h depending on l from 1 till 10^-l
    l = 5  # pylint: disable=invalid-name
    h_steps = [10 ** (-i) for i in range(l + 1)]
    print(f"h steps: 1, ..., 10^-{l}")
    error_plot(
        [(g1, dg1, ddg1, "g1")],
        h_steps,
        (np.pi, 3 * np.pi, 1000),
        h_line_on=True,
        title="errorplot h=1,...,10^-l",
    )

    ### Aufgabe 1.3 ###
    print("### Aufgabe 1.3 ###")
    kappa_list = [1, 0.001, 5]
    kappa_func_list = [
        (gk_creator(k), dgk_creator(k), ddgk_creator(k), f"g{k}") for k in kappa_list
    ]
    error_plot(
        kappa_func_list,
        h_steps,
        (np.pi, 3 * np.pi, 1000),
        h_line_on=True,
        title="kappa errorplot",
    )


if __name__ == "__main__":
    main()
