import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from numpy import linalg as LA
from scipy.linalg import norm, qr, solve_triangular


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

def read_data(path: str, collection: list = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",")
    if collection is not None:
        if max(collection) >= len(data):
            raise ValueError(f"In collection is at least one element out of bounds (length of data: {len(data)}).")
        if len(collection) < 5:
            raise ValueError("Collection needs at least 5 elements.")
        data = data[collection]

    b_vec = make_b(data)
    A_mat = make_A(data)
    return A_mat, b_vec, data

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

def plot_ellipsis(data_none: np.ndarray, x_vec: np.ndarray, x_vec_rd: np.ndarray, x_vec_none: np.ndarray,  collection: list):
    # assuming we have a file with all data and with rounded data for each letter
    x_points, y_points = data_none[:, 0], data_none[:, 1]
    x_data = np.linspace(x_points[0] - 1, x_points[0] + 5)
    y_data = np.linspace(y_points[0] - 5, y_points[0] + 1)

    x_data, y_data = np.meshgrid(x_data, y_data)

    # the given points
    points = plt.scatter(x_points, y_points, linewidths=0.9, color="blue",
                         marker=".", zorder=2.5)
    selected_points = plt.scatter(x_points[collection], y_points[collection], linewidths=0.9, color="black",
                         marker=".", zorder=3)

    # get ellipsis for all points:
    a, b, c, d, e = x_vec_none
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="red", zorder=2.1)

    # get ellipsis for rounded points
    a, b, c, d, e = x_vec_rd
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="green")

    # if more than 5 points: we can omit some oBdA the last (or try something else)
    a, b, c, d, e = x_vec
    function = - x_data**2 + a*y_data**2 + b*x_data*y_data + c*x_data + d*y_data + e
    plt.contour(x_data, y_data, function, [0], colors="purple")

    # we could also try solving for points that are more regularly distributed?

    # für die Legende (contour hat kein "label"-Attribut)
    solution = Line2D([0], [0], label='solution', color='red')
    rounded = Line2D([0], [0], label='rounded', color='green')
    selection = Line2D([0], [0], label='selection', color='purple')
    
    plt.legend([points, selected_points, solution, rounded, selection],
               ["points", "selected points", "solution", "rounded", "selection"])

    plt.show()
    # we should also look at the points more closely, how far are they off? (for c, d)

def plot_cond(A_mat_conds: list):
    # Transpose the data to separate columns
    columns = list(zip(*A_mat_conds))

    # Number of bars
    num_bars = len(columns)

    # Improved plot aesthetics
    plt.style.use('seaborn-darkgrid')

    # New figure and axes
    _, ax = plt.subplots(figsize=(12, 8))

    # Bar width and spacing
    bar_width = 0.2
    group_spacing = 0.5

    # Calculate new indices with spacing for each group
    new_indices = []
    base = 0
    for _ in range(len(A_mat_conds)):
        new_indices.append([base + j * bar_width for j in range(num_bars)])
        base += num_bars * bar_width + group_spacing

    # Colors for the bars
    colors = ['skyblue', 'steelblue', 'lightgreen', 'forestgreen', 'lightcoral', 'darkred']

    col_names = ["all", "selected", "rounded"]            
    # Plotting each pair of columns with connections
    for i in range(0, num_bars, 2):
        ax.bar(new_indices[i::num_bars], columns[i], bar_width, color=colors[i], label=f'A cond {col_names[i]}')
        ax.bar(new_indices[i+1::num_bars], columns[i+1], bar_width, color=colors[i+1], label=f'AtA cond {col_names[i]}')
        
        # Drawing lines to connect the bars
        for j in range(len(data)):
            ax.plot([new_indices[i::num_bars][j] + bar_width/2, new_indices[i+1::num_bars][j] + bar_width/2], 
                    [columns[i][j], columns[i+1][j]], color='black', linestyle='-', linewidth=1)



    # Enhancing the plot with labels, title, and legend
    ax.set_xlabel('Data Points', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Professional Bar Plot of Given Data', fontsize=14)
    ax.set_xticks(new_indices + bar_width * (num_bars - 1) / 2)
    ax.set_xticklabels([["a","b","c","d"][i] for i in range(len(A_mat_conds))])
    ax.legend(fontsize=10)

    # Using log scale for y-axis
    plt.yscale('log')

    # Show the plot
    plt.show()

def plot_cond(A_mat_conds: list):
    # Transpose the data to separate columns
    columns = list(zip(*A_mat_conds))

    # Number of bars
    num_bars = len(columns)

    # Improved plot aesthetics
    plt.style.use('seaborn-v0_8-darkgrid')

    # New figure and axes
    _, ax = plt.subplots(figsize=(12, 8))

    # Bar width and spacing
    bar_width = 0.2
    group_spacing = 0.5

    # Calculate new indices with spacing for each group
    new_indices = []
    base = 0
    for _ in range(len(A_mat_conds)):
        group_indices = [base + j * bar_width for j in range(num_bars)]
        new_indices.append(group_indices)
        base += num_bars * bar_width + group_spacing

    # Colors for the bars
    colors = ['skyblue', 'steelblue', 'lightgreen', 'forestgreen', 'lightcoral', 'darkred']

    col_names = ["all", "selected", "rounded"]
    example_names = ["a", "b","c","d"]
    # Plotting each pair of columns with connections
    for i in range(0, num_bars, 2):
        for j in range(len(A_mat_conds)):
            ax.bar(new_indices[j][i], columns[i][j], bar_width, color=colors[i], label=f'A cond ({col_names[i//2]})' if j == 0 else "")
            ax.bar(new_indices[j][i+1], columns[i+1][j], bar_width, color=colors[i+1], label=f'AtA cond ({col_names[i//2]})' if j == 0 else "")

    # Enhancing the plot with labels, title, and legend
    ax.set_xlabel('Examples', fontsize=12)
    ax.set_ylabel('Kondition', fontsize=12)
    ax.set_title('Konditionen für unterschiedliche Daten', fontsize=14)
    ax.set_xticks([np.mean(ni) for ni in new_indices])
    ax.set_xticklabels([f'Example {i}' for i in example_names])
    ax.legend(fontsize=10)

    # Using log scale for y-axis
    plt.yscale('log')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # TO-DO: Decide which plots (Aufgabe: 4.3)
    COLLECTION = [0, 1, 2, 3, 4]
    A_mat_conds = [] # condition of A and AtA for rounded, selected, and all as one tuple
    for letter in ["a", "b", "c", "d"]:
        print()
        print("#####################")
        print(f"### file {letter} ###")
        print("#####################")
        print()
        A_mat, b_vec, data = read_data(f"qr_data/ellipse_{letter}.txt", collection=COLLECTION)
        print("### Daten (selected) ###")
        print(data)
        print("### A Matrix (selected) ###")
        print(A_mat)
        print("### b Vektor (selected) ###")
        print(b_vec)
        q_mat, r_mat = make_qr(A_mat)
        print("### Q Matrix (selected) ###")
        print(q_mat)
        print("### R Matrix (selected) ###")
        print(r_mat)
        print("### A voller Spaltenrang? (selected) ###")
        print(check_rank(r_mat))
        x_vec, norm_z2 = solve_lgs(q_mat, r_mat, b_vec)
        print("### x Vektor (selected) ###")
        print(x_vec)
        print("### Norm von z2 (selected) ###")
        print(norm_z2)

        # calculate rd data
        A_mat_rd, b_vec_rd, data_rd = read_data(f"qr_data/ellipse_{letter}_rd.txt", collection=None)
        q_mat_rd, r_mat_rd = make_qr(A_mat_rd)
        x_vec_rd, norm_z2_rd = solve_lgs(q_mat_rd, r_mat_rd, b_vec_rd)            
        print("### Norm von z2 (rounded) ###")
        print(norm_z2_rd)

        # calculate for full data if not already
        if COLLECTION is not None:
            A_mat_none, b_vec_none, data_none = read_data(f"qr_data/ellipse_{letter}.txt", collection=None)
            q_mat_none, r_mat_none = make_qr(A_mat_none)
            x_vec_none, norm_z2_none = solve_lgs(q_mat_none, r_mat_none, b_vec_none)
            print("### Norm von z2 (all) ###")
            print(norm_z2_none)
        else:
            data_none = data.copy()
            A_mat_none = A_mat.copy()
            x_vec_none = x_vec.copy()

        if check_rank(r_mat):
            print("### Plotten ###")
            print("plot ellipsis")
            plot_ellipsis(data_none, x_vec, x_vec_rd, x_vec_none, COLLECTION)
            # Observation: if residuum is greater, than the ellipsis is bigger

        A_cond, AtA_cond = compare_cond(A_mat)
        print("### Kondition von A (selected) ###")
        print(A_cond)
        print("### Kondition von AtA (selected) ###")
        print(AtA_cond)
        if COLLECTION is not None:
            A_cond_none, AtA_cond_none = compare_cond(A_mat_none)
            print("### Kondition von A (all) ###")
            print(A_cond_none)
            print("### Kondition von AtA (all) ###")
            print(AtA_cond_none)

        A_cond_rd, AtA_cond_rd = compare_cond(A_mat_rd)
        print("### Kondition von A (rounded) ###")
        print(A_cond_rd)
        print("### Kondition von AtA (rounded) ###")
        print(AtA_cond_rd)

        # if letter != "b":
        A_mat_conds.append((A_cond_none, AtA_cond_none, A_cond, AtA_cond, A_cond_rd, AtA_cond_rd))

    plot_cond(A_mat_conds)


