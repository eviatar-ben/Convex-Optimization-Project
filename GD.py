import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

np.random.seed(0)

step = 0.000001
epsilon = 0.0001



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


def is_pd(X):
    return np.all(np.linalg.eigvalsh(X) > 0)


def is_sym(X):
    return np.all(X - X.T <= epsilon)


def constraint(X, a):
    # Constraint function: a_i^T * X^(-1) * a_i <= 1
    return np.dot(a.T, np.dot(X, a))


def projected_grad(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    eigenvalues[eigenvalues <= 0] = epsilon
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T #+ epsilon * np.eye(n)


def check_constraint(X, A):
    return all([constraint(X, a) <= 1 + epsilon for a in A])

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
    D = np.diag((max_norm2) * np.ones(len(a_max_norm)))
    U = generate_orthogonal_basis(a_max_norm)
    X0 = U @ D @ U.T
    return X0


def solve_optimization(A, name):
    # Generate a random positive definite symmetric matrix as the initial guess
    X0 = generate_initializer(A)

    # Define the optimization parameters
    max_iter = 1000  # Maximum number of iterations

    # Perform the optimization using gradient descent
    objective_score = []
    objective_with_barrier_score = []
    alphas = []
    C = inv(X0)
    constraints_list = {i:[] for i in range(len(A))}
    constraints_flag = True
    for i in range(max_iter):
        for j in range(len(A)):
            constraints_list[j].append(constraint(C, A[j]))
        C_inv = inv(C)
        t = i + 1
        alpha = 1 / t
        # if np.max([constraint(C, a) for a in A]) > 0.99:
        #     alpha = t
        equality_constraint = (1/t) * np.outer(A[-1], A[-1]) / (1 - A[-1].T @ C @ A[-1] + epsilon) - np.outer(A[-1], A[-1]) / (A[-1].T @ C @ A[-1])
        log_barrier = alpha * np.sum([np.outer(a, a) / (1 - a.T @ C @ a) for a in A[:-1]], axis=0)
        grad = -C_inv + log_barrier + equality_constraint
        # calculate the hessian
        C_next = C - step * grad
        if not check_constraint(C_next,A):
            print("constraint violated")
            constraints_flag = False
            break
        C = projected_grad(C_next)

        assert is_pd(C)
        assert check_constraint(C, A)

        if np.linalg.norm(grad) < epsilon:
            break
        if i % 50 == 0:
            alphas.append(alpha)

            objective_score.append(objective_var_change(C))
            # objective_with_barrier_score.append(objective_with_barrier(C, A, alpha=alpha))
            print(f"Iteration {i}, objective:{objective_var_change(C)},"
                  f" objective with barrier: {objective_with_barrier(C, A, alpha=alpha)},"
                  f" max constraint: {np.max([constraint(C, a) for a in A])}")

    # plot the objective function
    plt.plot(objective_score, label="objective")
    plt.legend()
    plt.savefig(f"figures/{name}_objective.png")
    # plot the constraints
    for i in range(len(A)):
        plt.plot(constraints_list[i], label=f"constraint {i}")
    plt.legend()
    plt.savefig(f"figures/{name}_constraints_({constraints_flag}).png")
    return inv(C)

if __name__ == "__main__":
    # A = np.random.rand(M, n)
    for path in glob(".\examples\*"):
        # load npy file:
        name = os.path.basename(path).split(".npy")[0]
        print(f"processing {name}")
        A = np.load(path)
        A = A[np.argsort([np.linalg.norm(a) for a in A])]
        X_optimal = solve_optimization(A, name)
        break
