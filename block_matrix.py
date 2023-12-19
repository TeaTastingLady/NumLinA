"""
Author: Karla Menze, Laura Faustmann
Date: 20/12/2023
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu
from scipy.sparse import bmat, diags


class BlockMatrix:
    """Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """

    def __init__(self, d, n):  # pylint: disable=invalid-name
        if d < 1 or n < 2:
            raise ValueError(
                f"there was a wrong input parameter: d > 0 (BUT: {d} > 0) or n > 1 (BUT: {n} > 1)"
            )
        self.d = d  # pylint: disable=invalid-name
        self.n = n  # pylint: disable=invalid-name

    def get_sparse(self):
        """Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """
        diagonals = np.full(self.n - 1, 2 * self.d)
        off_diagonals = np.full(self.n - 2, -1)
        A = diags( # pylint: disable=invalid-name
            [diagonals, off_diagonals, off_diagonals], [0, -1, 1], format="csr"
        )

        for l in range(2, self.d + 1):  # pylint: disable=invalid-name
            identity = -np.eye((self.n - 1) ** (l - 1))
            block_rows = []
            for i in range(self.n - 1):
                block_row = []
                for j in range(self.n - 1):
                    if i == j:
                        block_row.append(A)
                    elif abs(j - i) == 1:
                        block_row.append(identity)
                    else:
                        block_row.append(None)
                block_rows.append(block_row)
            A = bmat(block_rows, format="csr")  # pylint: disable=invalid-name

        return A

    def eval_sparsity(self):
        """Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        A = self.get_sparse()  # pylint: disable=invalid-name
        non_zero = A.count_nonzero()
        return non_zero, non_zero / (self.n - 1) ** (2 * self.d)

    def get_lu(self):
        """Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray0
            upper triangular matrix of LU-decomposition
        """
        A = self.get_sparse() # pylint: disable=invalid-name
        return lu(A.toarray())

    def eval_sparsity_lu(self):
        """Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        _, l, u = self.get_lu() # pylint: disable=invalid-name, unbalanced-tuple-unpacking
        nnz_l = np.count_nonzero(l)
        nnz_u = np.count_nonzero(u)
        nnz_lu = nnz_l + nnz_u - len(l)  # minus ones on diagonal of l
        return nnz_lu, nnz_lu / (self.n - 1) ** (2 * self.d)


def nz_plots_n(d, n_list):  # pylint: disable=invalid-name
    """Plots the storage requirement of the sparse Matrix A^d and a
    full populated Matrix depending on n.
    Parameters
    ----------
    d: int
    dimension of Matrix
    n_list: list
    list of number of partial intervals
    Return
    ------
    callable
    """
    nz_values_sparse = [BlockMatrix(d, n).eval_sparsity()[0] for n in n_list]
    nz_values_full = [(n - 1) ** (2 * d) for n in n_list]
    plt.plot(
        n_list,
        nz_values_sparse,
        marker="o",
        linestyle="-",
        markersize=4,
        color="blue",
        label="sparse",
    )
    plt.plot(
        n_list,
        nz_values_full,
        marker="o",
        linestyle="-",
        markersize=4,
        color="red",
        label="full",
    )
    plt.title(f"non-zero values depending on n with d={d}")
    plt.xlabel("n values")
    plt.ylabel("non-zero values")
    plt.legend()
    plt.tight_layout()
    plt.show()


def nz_plots_d(n, d_list):  # pylint: disable=invalid-name
    """Plots the storage requirement of the sparse Matrix A^d
    and a full populated Matrix depending on d.
    Parameters
    ----------
    n: int
    number of partial intervals
    d_list: list
    list of dimensions
    Return
    ------
    callable
    """
    nz_values_sparse = [BlockMatrix(d, n).eval_sparsity()[0] for d in d_list]
    nz_values_full = [(n - 1) ** (2 * d) for d in d_list]
    plt.plot(
        d_list,
        nz_values_sparse,
        marker="o",
        linestyle="-",
        markersize=4,
        color="blue",
        label="sparse",
    )
    plt.plot(
        d_list,
        nz_values_full,
        marker="o",
        linestyle="-",
        markersize=4,
        color="red",
        label="full",
    )
    plt.title(f"non-zero values depending on d with n={n}")
    plt.xlabel("d values")
    plt.ylabel("non-zero values")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


def nz_plots_N(N_list):  # pylint: disable=invalid-name
    """Plots the storage requirement of the sparse Matrix A^d
    and a full populated Matrix depending on N.
    Parameters
    ----------
    N_list: list
    list of discretization points
    Return
    ------
    callable
    """
    nz_values_full = [N**2 for N in N_list]
    plt.plot(
        N_list,
        N_list,
        marker="o",
        linestyle="-",
        markersize=4,
        color="blue",
        label="sparse",
    )
    plt.plot(
        N_list,
        nz_values_full,
        marker="o",
        linestyle="-",
        markersize=4,
        color="red",
        label="full",
    )
    plt.title("non-zero values depending on N")
    plt.xlabel("N values")
    plt.ylabel("non-zero values")
    plt.legend()
    plt.tight_layout()
    plt.show()


def nz_plots_A_lu(d, n_list):  # pylint: disable=invalid-name
    """Plots the storage requirement of the sparse Matrix A^d and a
    full populated Matrix depending on n.
    Parameters
    ----------
    d: int
    dimension of Matrix
    n_list: list
    list of number of partial intervals
    Return
    ------
    callable
    """
    nz_values_lu = [BlockMatrix(d, n).eval_sparsity_lu()[0] for n in n_list]
    nz_values_A = [BlockMatrix(d, n).eval_sparsity()[0] for n in n_list] # pylint: disable=invalid-name
    N_values = [(n - 1) ** d for n in n_list] # pylint: disable=invalid-name
    plt.plot(
        N_values,
        nz_values_lu,
        marker="o",
        linestyle="-",
        markersize=4,
        color="blue",
        label="lu",
    )
    plt.plot(
        N_values,
        nz_values_A,
        marker="o",
        linestyle="-",
        markersize=4,
        color="red",
        label="A (sparse)",
    )
    plt.title(f"non-zero values of lu and A depending on N with d={d}")
    plt.xlabel("N values")
    plt.ylabel("non-zero values")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """main function"""
    n = 3  # pylint: disable=invalid-name
    d = 3  # pylint: disable=invalid-name
    mat = BlockMatrix(d, n)
    print(f"A_d matrix for n={n} and d={d}:")
    print(mat.get_sparse().toarray())
    print()
    print("non-zero values of sparse (absolute, relative):")
    print(mat.eval_sparsity())
    print()
    print("non-zero values of lu (absolute, relative):")
    print(mat.eval_sparsity_lu())


if __name__ == "__main__":
    main()
    # nz_plots_n(3, list(range(2, 10)))
    # nz_plots_d(5, list(range(1, 6)))
    # nz_plots_N(list(range(1,25)))
    nz_plots_A_lu(1, list(range(2, 10)))
    nz_plots_A_lu(2, list(range(2, 10)))
    nz_plots_A_lu(3, list(range(2, 10)))
