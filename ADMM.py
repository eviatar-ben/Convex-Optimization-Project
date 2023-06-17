# Given a1...am
# we want to minimize the following function:
# f(x) = log(det(X)) where X is PSD matrix
# such that ai.T*X^-1*ai <=1 ,  ai for all i

import numpy as np


def optimize_function(X, rho, max_iter=1000, tol=1e-6):
    """
    Solves the optimization problem min [-log(det(X))] subject to X being a PSD matrix.

    Parameters:
        X (numpy.ndarray): Initial matrix X.
        rho (float): Penalty parameter for ADMM.
        max_iter (int): Maximum number of iterations. Defaults to 1000.
        tol (float): Tolerance for convergence. Defaults to 1e-6.

    Returns:
        X_optimal (numpy.ndarray): Optimal solution for X.
    """
    # Get the dimensions of X
    n = X.shape[0]

    # Initialize variables
    Z = np.copy(X)
    U = np.zeros_like(X)

    # ADMM iterations
    for iteration in range(max_iter):
        # Update X
        X_prev = np.copy(X)
        X = update_X(X, Z, U, rho)

        # Update Z
        Z = update_Z(X, Z, U, rho)

        # Update U
        U = U + X - Z

        # Check convergence
        norm_residual = np.linalg.norm(X - Z)
        norm_X = np.linalg.norm(X)
        if norm_residual < tol * norm_X:
            break

    return X


def update_X(X, Z, U, rho):
    """
    Updates X using closed-form solution.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(Z - U)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensures eigenvalues are non-negative
    X = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T + U - Z / rho
    return X


def update_Z(X, Z, U, rho):
    """
    Updates Z using the soft-thresholding operator.
    """
    Z = X + U
    eigenvalues, eigenvectors = np.linalg.eigh(Z)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensures eigenvalues are non-negative
    Z = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T - U
    return Z


# Example usage
n = 3  # Dimension of the matrix X
X = np.eye(n)  # Initial matrix X
rho = 1.0  # Penalty parameter

X_optimal = optimize_function(X, rho)