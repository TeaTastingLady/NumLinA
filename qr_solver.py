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

def read_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",")
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

if __name__ == "__main__":
    # TO-DO: Decide which plots (Aufgabe: 4.3)
    for letter in ["a", "b", "c", "d"]:
        print()
        print("#####################")
        print(f"### file {letter} ###")
        print("#####################")
        print()
        A_mat, b_vec = read_data(f"qr_data/ellipse_{letter}.txt")
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



