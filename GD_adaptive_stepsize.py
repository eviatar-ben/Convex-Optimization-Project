import numpy
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

np.random.seed(0)

step = 0.0001  # 0.00005
epsilon = 0.0001

step_sizes = (10 * np.ones(7)) ** np.array(np.arange(7) - 10)


def score(X, A):
  scores = np.einsum('...i,ij,...j->...', A, np.linalg.inv(X), A)
  _, logdet = np.linalg.slogdet(X)
  return logdet, np.all(scores <= 1.)


def inv(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    return eigenvectors @ np.diag(1 / eigenvalues) @ eigenvectors.T


def objective(X):
    # Objective function: negative logarithm of the determinant of X
    return np.log(np.linalg.det(X))


def objective_with_barrier(X, A, alpha=1.0):
    # Objective function: negative logarithm of the determinant of X
    return np.log(np.linalg.det(X)) - np.sum([np.log(1 - a.T @ X @ a) for a in A]) * alpha


def objective_var_change(C):
    # Objective function: negative logarithm of the determinant of X
    return np.log(np.linalg.det(inv(C)))


def objective_var_changeC(C):
    # Objective function: negative logarithm of the determinant of X
    return -np.log(np.linalg.det(C))


def is_pd(X):
    return np.all(np.linalg.eigvalsh(X) > 0)


def is_sym(X):
    return np.all(X - X.T <= epsilon)


def constraint(X, a):
    # Constraint function: a_i^T * X^(-1) * a_i <= 1
    return np.dot(a.T, np.dot(X, a))


def print_constraints(C, A):
    print([constraint(C, a) for a in A])


def projected_grad(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    eigenvalues[eigenvalues <= 0] = epsilon
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T  # + epsilon * np.eye(n)


def check_constraint(X, A):
    return all([constraint(X, a) <= 1 for a in A])


def check_constraint_list(listC, A):
    return np.array([check_constraint(C, A) for C in listC])


def generate_orthogonal_basis(vector):
    n = len(vector)
    basis = [vector / np.linalg.norm(vector)]  # Normalize the input vector to create the first basis vector

    for i in range(1, n):
        orthogonal_vector = np.random.rand(n)  # Generate a random vector
        for j in range(i):
            orthogonal_vector -= np.dot(basis[j], orthogonal_vector) * basis[
                j]  # Remove projections onto previous basis vectors
        orthogonal_vector /= np.linalg.norm(orthogonal_vector)  # Normalize the orthogonal vector
        basis.append(orthogonal_vector)

    for b1 in basis:
        for b2 in basis:
            if b1 is not b2:
                assert np.abs(np.dot(b1, b2)) < epsilon
            else:
                assert np.abs(np.dot(b1, b2) - 1) < epsilon
    return np.array(basis).T


def generate_initializer(A):
    a_vectors = [a for a in A]
    i_max_norm = np.argmax([np.linalg.norm(a) for a in A])
    a_max_norm = a_vectors[i_max_norm]
    max_norm2 = np.linalg.norm(a_max_norm, ord=2) ** 2
    D = np.diag((max_norm2 + 0.05) * np.ones(len(a_max_norm)))
    U = generate_orthogonal_basis(a_max_norm)
    X0 = U @ D @ U.T
    return X0


def handle_edge_cases(A):
    # check if A is all zero
    if np.all([np.all(a == 0) for a in A]):
        X0 = np.zeros((len(A[0]), len(A[0]))) + epsilon * np.eye(len(A[0]))
        assert is_pd(X0)
        return X0


def filter_unnecessary_constraints(A):
    # get rid of rows that are all zeros
    A = [a for a in A if not np.all(a == 0)]
    return A


def solve_optimization(A, name):
    X0 = handle_edge_cases(A)
    if X0 is not None:
        return X0
    # A = filter_unnecessary_constraints(A)
    # Generate a random positive definite symmetric matrix as the initial guess
    X0 = generate_initializer(A)

    # Define the optimization parameters
    max_iter = 15  # Maximum number of iterations

    # Perform the optimization using gradient descent
    objective_score = []
    objective_with_barrier_score = []
    best_step_size = []
    alphas = []
    C = inv(X0)
    constraints_list = {i: [] for i in range(len(A))}
    constraints_flag = True
    for i in range(max_iter):
        for j in range(len(A)):
            constraints_list[j].append(constraint(C, A[j]))
        if not check_constraint(C, A):
            print_constraints(C, A)
            raise ValueError(f"Start of the loop... Constr. violated")
        C_inv = inv(C)
        t = i + 1
        alpha = 1 / t
        # if np.max([constraint(C, a) for a in A]) > 0.99:
        #     alpha = t
        # equality_constraint = - np.outer(A[-1], A[-1]) / (A[-1].T @ C @ A[-1])
        log_barrier = alpha * np.sum([np.outer(a, a) / (1 - a.T @ C @ a - epsilon) for a in A], axis=0)
        grad = -C_inv + log_barrier  # + equality_constraint
        # calculate the hessian
        possible_C_next_list = []
        for step in step_sizes:
            possible_C_next_list.append(C - step * grad)
        possible_C_next = np.array(possible_C_next_list)
        valid_C_next = possible_C_next[check_constraint_list(possible_C_next, A)]
        if len(valid_C_next) > 0:
            C_best_next = valid_C_next[np.nanargmin([objective_var_changeC(C) for C in valid_C_next])]
            curr_step_size = step_sizes[np.where(possible_C_next == C_best_next)[0][0]]
        else:
            print_constraints(possible_C_next[0], A)
            # raise ValueError(f"Constr. violated")
            break

        # if not check_constraint(C_next,A):
        #     print("constraint violated")
        #     constraints_flag = False
        #     break
        C = projected_grad(C_best_next)
        if not check_constraint(C, A):
            print("constraint violated")
            constraints_flag = False

        assert is_pd(C)
        assert check_constraint(C, A)

        if np.linalg.norm(grad) < epsilon:
            break
        if i % 50 == 0:
            alphas.append(alpha)
            best_step_size.append(curr_step_size)
            objective_score.append(objective_var_change(C))
            # objective_with_barrier_score.append(objective_with_barrier(C, A, alpha=alpha))
            print(f"Iteration {i}, objective:{objective_var_change(C)},"
                  # f" objective with barrier: {objective_with_barrier(C, A, alpha=alpha)},"
                  f" max constraint: {np.max([constraint(C, a) for a in A])}")

    # # plot the objective function
    # plt.figure()
    # plt.plot(objective_score, label="objective")
    # plt.legend()
    # plt.savefig(f"./figures/{name}_objective.png")
    # # plot the constraints
    # plt.figure()
    # for i in range(len(A)):
    #     plt.plot(constraints_list[i], label=f"constraint {i}")
    # plt.legend()
    # plt.savefig(f"./figures/{name}_constraints({constraints_flag}).png")
    #
    # plt.figure()
    # plt.semilogy(best_step_size)
    # plt.savefig(f"./figures/{name}_stepsize.png")
    # # plt.show()
    assert check_constraint(C, A)
    return inv(C)


if __name__ == "__main__":
    # A = np.random.rand(M, n)
    # get all the file names in examples file:
    listdir = [l.split(".npy")[0] for l in os.listdir("./examples")]

    all_files = ['blobs.100.10', 'blobs.1000.100', 'blobs.1000.101', 'checkerboard.50.4', 'checkerboard.500.40',
                 'gaussian.10.10', 'gaussian.1000.1000', 'gaussian.2.5', 'gaussian.5.2', 'moons.1000.2', 'moons.50.2',
                 'sparse.1000.10', 'sparse.100000.100', 'spiral.1000.20', 'spiral.10000.200', 'uniform.10.10',
                 'uniform.1000.1000', 'uniform.2.5', 'uniform.5.2', 'wave.50.4', 'wave.500.20', 'wave.5000.100']

    passed = ['blobs.100.10', 'blobs.1000.100', 'blobs.1000.101',
              'checkerboard.50.4', 'checkerboard.500.40',
              "gaussian.2.5", "gaussian.5.2", "gaussian.10.10",
              "moons.50.2", "moons.1000.2",
              # "spiral.1000.20",
              "uniform.2.5", "uniform.5.2", "uniform.10.10",
              "wave.50.4", "wave.500.20", "wave.5000.100"]

    to_check = ["wave.5000.100"]

    take_too_long = ["gaussian.1000.1000", "sparse.1000.10", "sparse.100000.100", "spiral.10000.200", "wave.5000.100"]

    hist = {}
    for path in glob("./examples/*"):
        # load npy file:
        name = os.path.basename(path).split(".npy")[0]
        if name not in passed:
            continue

        print(f"processing {name}")
        A = np.load(path)
        A = A[np.argsort([np.linalg.norm(a) for a in A])]
        X_optimal = solve_optimization(A, name)
        X_optimal_score, constraints = score(X_optimal, A)
        print(f"score: {X_optimal_score}")
        print(f"constraints: {constraints}")
        assert constraints == 1.0
        hist[name] = X_optimal_score
        break

    plt.figure()
    plt.bar(range(len(hist)), hist.values(), tick_label=hist.keys())
    plt.xticks(rotation=90)
    plt.savefig("./figures/histogram.png")
    plt.show()

