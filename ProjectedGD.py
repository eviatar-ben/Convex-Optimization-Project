import numpy as np
from scipy.linalg import eigh


# Given a1...am
# we want to minimize the following function:
# f(x) = log(det(X)) where X is PSD matrix
# such that ai.T*X^-1*ai <=1 ,  ai for all i


def project_psd(X):
    eigenvalues, eigenvectors = eigh(X)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def objective(X):
    return np.log(np.linalg.det(X))


def gradient(X):
    # gradient of log(det(x))
    return np.linalg.inv(X)


def constraint(X, A):
    return np.all([ai.T @ np.linalg.inv(X) @ ai <= 1 for ai in A])


def projected_gradient_descent(X_init, learning_rate, num_iterations):
    X = X_init.copy()

    for i in range(num_iterations):
        grad = gradient(X)
        X -= learning_rate * grad
        X = project_psd(X)

    return X


def main():
    # Example usage
    n = 40  # Dimension of the matrix
    X_init = np.random.rand(n, n)  # Initialize a random matrix
    X_init = X_init @ X_init.T  # Ensure symmetry
    # todo take into account the constraints "such that ai.T*X^-1*ai <=1 ,  ai for all i"  by lagrange multipliers

    learning_rate = 0.01
    num_iterations = 10000

    optimal_X = projected_gradient_descent(X_init, learning_rate, num_iterations)
    print(optimal_X)
    print(np.log(np.linalg.det(optimal_X)))


if __name__ == '__main__':
    main()
