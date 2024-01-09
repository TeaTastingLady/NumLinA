import numpy as np
from scipy.linalg import qr, solve_triangular, norm

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

def read_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",")
    print("Daten")
    print(data)
    b_vec = make_b(data)
    print("b Vektor")
    print(b_vec)
    A_mat = make_A(data)
    print("A Matrix")
    print(A_mat)
    return A_mat, b_vec

def make_qr(A_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_mat, r_mat = qr(A_mat)
    return q_mat, r_mat

def check_rank(r_mat: np.ndarray) -> bool:
    _, n_dim = r_mat.shape
    for i in range(n_dim):
        # check eigenvalues
        if r_mat[i][i] == 0:
            return False

    return True

def solve_lgs(q: np.ndarray, r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    _, n = r.shape
    q_t = np.transpose(q)
    z_vector = np.matmul(q_t, b)
    z_1, z_2 = z_vector[:n], z_vector[n:]
    r_1 = r[:n]
    # TO-DO: check if solution is unique and implement other case
    x_vec = solve_triangular(r_1, z_1, lower=False)
    return x_vec, norm(z_2)

if __name__ == "__main__":
    A_mat, b_vec = read_data("qr_data/ellipse_d.txt")
    q_mat, r_mat = make_qr(A_mat)
    print("### Q Matrix ###")
    print(q_mat)
    print("### R Matrix ###")
    print(r_mat)
    print("### A voller Spaltenrang? ###")
    print(check_rank(r_mat))
    x_vec, norm_z2 = solve_lgs(q_mat, r_mat, b_vec)
    print("### x Vektor ###")
    print(x_vec)
    print("### Norm von z2 ###")
    print(norm_z2)



