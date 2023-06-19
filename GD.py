import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
M = 30  # Number of vectors a
n = 30  # Dimension of the vectors a
step = 0.00000001
epsilon = 0.0001

def inv(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    return eigenvectors @ np.diag(1/eigenvalues) @ eigenvectors.T


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
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T + epsilon*np.eye(n)

def check_constraint(X, A):
    return all([constraint(X, a)<=1.001 for a in A])

def generate_random_X():
    # Generate a random lower triangular matrix
    # lower_triangular = np.random.uniform(0, 1, size=(n, n))
    # lower_triangular[np.triu_indices(n, 1)] = 0
    #
    # # Construct a symmetric matrix
    # symmetric_matrix = lower_triangular @ lower_triangular.T
    root_matrix = np.random.rand(n, n)
    psd = root_matrix @ root_matrix.T
    # Add a small positive constant to ensure positive definiteness
    positive_definite_matrix = psd + np.eye(n) * epsilon
    return positive_definite_matrix

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

    return np.array(basis).T

def generate_initializer(A):
    a_vectors = [a for a in A]
    i_max_norm = np.argmax([np.linalg.norm(a) for a in A])
    a_max_norm = a_vectors[i_max_norm]
    max_norm2 = np.linalg.norm(a_max_norm, ord=2)**2
    D = np.diag(max_norm2*np.ones(n)) * 2
    # U = generate_orthogonal_basis(a_max_norm)

    X0 = D
    #X0 = U @ D @ U.T
    return X0


def solve_optimization(A):
    # Generate a random positive definite symmetric matrix as the initial guess
    X0 = generate_initializer(A)

    # Define the optimization parameters
    max_iter = 10000  # Maximum number of iterations

    # Perform the optimization using gradient descent
    objective_score = []
    objective_with_barrier_score = []
    C = inv(X0)
    for i in range(max_iter):
        C_inv = inv(C)
        t = i + 1
        alpha = 1 / t
        log_barrier = alpha * np.sum([np.outer(a, a) / (1 - a.T @ C @ a) for a in A], axis=0)
        grad = -C_inv + log_barrier
        C_next = C - step * grad
        C = projected_grad(C_next)


        assert check_constraint(C,A)

        if np.linalg.norm(grad) < epsilon:
            break
        objective_score.append(objective_var_change(C))
        objective_with_barrier_score.append(objective_with_barrier(C, A, alpha=alpha))
        if i % 50 == 0:
            print(f"Iteration {i}, objective:{objective_var_change(C)},"
                  f" objective with barrier: {objective_with_barrier(C, A, alpha=alpha)},"
                  f" max constraint: {np.max([constraint(C, a) for a in A])}")

    # plot the objective function and the objective function with barrier

    plt.plot(objective_score, label="objective")
    plt.plot(objective_with_barrier_score, label="objective with barrier")
    plt.legend()
    plt.show()

    return inv(C)


# Example usage:


# Generate random vectors a

# for i in range(1000):
#     A = np.random.rand(M, n)
#     X = generate_initializer(A)
#     assert check_constraint(X, A)
#     assert is_pd(X)

A = np.random.rand(M, n)
# Solve the optimization problem
X_optimal = solve_optimization(A)
#print("Optimal X:")
#print(X_optimal)