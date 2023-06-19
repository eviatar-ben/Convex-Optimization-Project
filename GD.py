import numpy as np

M = 10  # Number of vectors a
n = 40  # Dimension of the vectors a
step = 0.000001
epsilon = 0.0001


def inv(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    return eigenvectors @ np.diag(1 / eigenvalues) @ eigenvectors.T


def objective(X):
    # Objective function: negative logarithm of the determinant of X
    return np.log(np.linalg.det(X))


def is_pd(X):
    return np.all(np.linalg.eigvalsh(X) > 0)


def is_sym(X):
    return np.all(X - X.T <= epsilon)


def constraint(X, a):
    # Constraint function: a_i^T * X^(-1) * a_i <= 1
    X_inv = np.linalg.inv(X)
    return np.dot(a.T, np.dot(X_inv, a))


def projected_grad(X):
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T + epsilon * np.eye(n)


def check_constraint(X, A):
    return all([constraint(X, a) <= 1.001 for a in A])


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
    max_norm2 = np.linalg.norm(a_max_norm, ord=2) ** 2
    D = np.diag(max_norm2 * np.ones(n))
    # U = generate_orthogonal_basis(a_max_norm)

    X0 = D
    # X0 = U @ D @ U.T
    return X0


def solve_optimization(A):
    # Generate a random positive definite symmetric matrix as the initial guess
    # X0 = generate_random_X()
    X0 = generate_initializer(A)

    # Define the optimization parameters
    max_iter = 1000  # Maximum number of iterations

    # Perform the optimization using gradient descent
    X = X0.copy()
    for i in range(max_iter):
        X_inv = inv(X)
        # alpha = a.T @ X_inv @ a
        # alpha_list = [a.T @ X_inv @ a for a in A]
        # alpha_mat_list = [np.diag(a) @ X_inv @ np.diag(a) for a in A]
        t = i + 1
        grad = X_inv - 1 / t * np.sum(
            [(np.diag(a) @ X_inv @ X_inv @ np.diag(a)) / (1 - a.T @ X_inv @ a - epsilon) for a in A], axis=0)
        # grad = +X_inv @ np.sum([1 - 1/t*(alpha_mat_list[j]/(1-alpha_list[j])) for j in range(M)], axis=0)

        X_new = X - step * grad
        X = projected_grad(X_new)

        # assert check_constraint(X,A)

        if np.linalg.norm(grad) < epsilon:
            break

        if i % 50 == 0:
            print(objective(X))

    print([constraint(X, a) for a in A])
    # print(f"Check constraint: {check_constraint(X, A)}")
    print(f"{objective(X)}")

    return X


# Example usage:


# Generate random vectors a

# for i in range(1000):
#     A = np.random.rand(M, n)
#     X = generate_initializer(A)
#     assert check_constraint(X, A)
#     assert is_pd(X)

A = np.random.randint(low=-10, high=10, size=(M, n))
# Solve the optimization problem
X_optimal = solve_optimization(A)
# print("Optimal X:")
# print(X_optimal)
