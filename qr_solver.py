import numpy as np
from scipy.linalg import qr, solve_triangular, norm
from numpy import linalg as LA

def make_b(data: np.ndarray) -> np.ndarray:
    return np.square(data[:, 0])

def make_A(data: np.ndarray) -> np.ndarray:
    col_1 = np.square(data[:, 1])
    col_3 = data[:, 0]
    col_4 = data[:, 1]
    col_2 = np.multiply(col_3, col_4)
    col_5 = np.ones(len(col_3))
    A_mat = np.column_stack((col_1, col_2, col_3, col_4, col_5))
    return A_mat

def read_data(path: str, collection: list = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",")
    if collection is not None:
        if max(collection) >= len(data):
            raise ValueError(f"In collection is at least one element out of bounds (length of data: {len(data)}).")
        if len(collection) < 5:
            raise ValueError("Collection needs at least 5 elements.")
        data = data[collection]
    print("### Daten ###")
    print(data)
    b_vec = make_b(data)
    A_mat = make_A(data)
    return A_mat, b_vec

def make_qr(A_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_mat, r_mat = qr(A_mat)
    return q_mat, r_mat

def check_rank(r_mat: np.ndarray) -> bool:
    _, n_dim = r_mat.shape
    for i in range(n_dim):
        # check eigenvalues
        if abs(r_mat[i][i]) < 1e-7:
            print(abs(r_mat[i][i]))
            return False

    return True

def solve_lgs(q: np.ndarray, r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    _, n = r.shape
    q_t = np.transpose(q)
    z_vector = np.matmul(q_t, b)
    z_1, z_2 = z_vector[:n], z_vector[n:]
    r_1 = r[:n]
    # TO-DO: check if solution is unique and implement other case
    if check_rank(r):
        x_vec = solve_triangular(r_1, z_1, lower=False)
    else:
        x_vec = None
    return x_vec, norm(z_2)

def compare_cond(A_mat: np.ndarray) -> tuple[float, float]:
    return LA.cond(A_mat), LA.cond(np.matmul(np.transpose(A_mat), A_mat))

import matplotlib.pyplot as plt
def plot_sol(x_vec, path, collection):
    a, b, c, d, e = x_vec
    # TODO: Grenzen von Daten abhängig wählen!
    # for now use the first entry of the data collection
    x_points, y_points = np.genfromtxt(path, delimiter=",", unpack = True)
    x_data = np.linspace(x_points[0] - 1, x_points[0]+5)
    y_data = np.linspace(y_points[0]-5, y_points[0]+1)

    x_data, y_data = np.meshgrid(x_data, y_data)
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e

    plt.contour(x_data, y_data, function, [0])
    plt.scatter(x_points, y_points, linewidth=0.2)

    plt.show()

def solve(path, collection):
    A_mat, b_vec = read_data(path, collection)
    q_mat, r_mat = make_qr(A_mat)
    x_vec, _ = solve_lgs(q_mat, r_mat, b_vec)

    return x_vec

def plot_ellipsis(letter):
    # assuming we have a file with all data and with rounded data for each letter
    print()
    print("#####################")
    print(f"### file {letter} ###")
    print("#####################")
    print()
    

    x_points, y_points = np.genfromtxt(f"qr_data\ellipse_{letter}.txt", delimiter=",", unpack = True)
    x_data = np.linspace(x_points[0] - 1, x_points[0]+5)
    y_data = np.linspace(y_points[0]-5, y_points[0]+1)

    x_data, y_data = np.meshgrid(x_data, y_data)

    collection=[0, 1, 2, 3, 4]

    # the given points
    points = plt.scatter(x_points, y_points, linewidths=0.9, color="blue",
                         marker=".", zorder=2.5)
    selected_points = plt.scatter(x_points[collection], y_points[collection], linewidths=0.9, color="black",
                         marker=".", zorder=3)

    # get ellipsis for all points:
    a, b, c, d, e = solve(f"qr_data\ellipse_{letter}.txt", collection=None)
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="red", zorder=2.1)

    # get ellipsis for rounded points
    a, b, c, d, e = solve(f"qr_data\ellipse_{letter}_rd.txt", collection=None)
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="green")

    # if more than 5 points: we can omit some oBdA the last (or try something else)
    a, b, c, d, e = solve(f"qr_data\ellipse_{letter}.txt", collection=collection)
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="purple")

    # we could also try solving for points that are more regularly distributed?

    # für die Legende (contour hat kein "label"-Attribut)
    from matplotlib.lines import Line2D # (move to the beginning)
    solution = Line2D([0], [0], label='solution', color='red')
    rounded = Line2D([0], [0], label='rounded', color='green')
    selection = Line2D([0], [0], label='selection', color='purple')
    
    plt.legend([points, selected_points, solution, rounded, selection],
               ["points", "selected points", "solution", "rounded", "selection"])

    plt.show()

    # we should also look at the points more closely, how far are they off? (for c, d)

if __name__ == "__main__":
    # TO-DO: Decide which plots (Aufgabe: 4.3)
    for letter in ["a", "b", "c", "d"]:
        print()
        print("#####################")
        print(f"### file {letter} ###")
        print("#####################")
        print()
        A_mat, b_vec = read_data(f"qr_data/ellipse_{letter}.txt", collection=[0, 1, 2, 3, 4])
        print("### A Matrix ###")
        print(A_mat)
        print("### b Vektor ###")
        print(b_vec)
        q_mat, r_mat = make_qr(A_mat)
        print("### Q Matrix ###")
        print(q_mat)
        print("### R Matrix ###")
        print(r_mat)
        print("### A voller Spaltenrang? ###")
        print(check_rank(r_mat))
        if check_rank(r_mat):
            plot_ellipsis(letter)
        x_vec, norm_z2 = solve_lgs(q_mat, r_mat, b_vec)
        print("### x Vektor ###")
        print(x_vec)
        print("### Norm von z2 ###")
        print(norm_z2)
        A_cond, AtA_cond = compare_cond(A_mat)
        print("### Kondition von A ###")
        print(A_cond)
        print("### Kondition von AtA ###")
        print(AtA_cond)


