import numpy as np
from scipy.linalg import eigh


# Given a1...am
# we want to minimize the following function:
# f(x) = log(det(X)) where X is PSD matrix
# such that ai.T*X^-1*ai <=1 ,  ai for all i


# Example usage
n = 4  # Dimension of the matrix
epsilon = 0.001

def project_pd(X):
    # X is symmetric, not necessarily PD:
    eigenvalues, eigenvectors = eigh(X)
    eigenvalues[eigenvalues < 0] = 0
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T + epsilon * np.eye(n)


def project_constraints(X, constraints):
    # Project X onto the feasible set defined by the constraints

    for ai in constraints:
        # Check if the constraint ai.T * X^(-1) * ai > 1 is violated
        X_inv = np.linalg.inv(X)
        while np.dot(ai.T, X_inv).dot(ai) > 1:
            # Adjust X to satisfy the constraint
            scaling_factor = np.sqrt(np.dot(ai.T, np.linalg.inv(X)).dot(ai))
            X /= scaling_factor
            X_inv = np.linalg.inv(X)  # Recompute X_inv to ensure positive definiteness

    return X


def objective(X):
    return np.log(np.linalg.det(X))


def gradient(X):
    # gradient of log(det(x))
    return np.linalg.inv(X)


def constraint(X, A):
    return np.all([ai.T @ np.linalg.inv(X) @ ai <= 1 for ai in A])


def projected_gradient_descent(X_init, constraints, learning_rate, num_iterations):
    X = X_init.copy()

    for i in range(num_iterations):
<<<<<<< HEAD
        grad = gradient(X)
        X -= learning_rate * grad
        X = project_psd(X)
        X = project_constraints(X, constraints)
        # print(X)
=======
        grad = gradient(X) # is PD
        X_new = X - learning_rate * grad # X^(t+1) = X^(t) - eta * grad{X^(t)}
        # X_new is symmetric (sum of symmetric matrices) but not necessarlity PD:
        X = project_pd(X_new)
>>>>>>> 9ddd85417751df943f083cf85c77d602adbe009e

    return X


def main():
<<<<<<< HEAD
    # Example usage
    n = 50  # Dimension of the matrix
=======
>>>>>>> 9ddd85417751df943f083cf85c77d602adbe009e
    X_init = np.random.rand(n, n)  # Initialize a random matrix
    X_init = X_init @ X_init.T + epsilon*np.eye(n)  # Ensure symmetry and PD.


    # todo take into account the constraints "such that ai.T*X^-1*ai <=1 ,  ai for all i"  by lagrange multipliers

    learning_rate = 0.01
    num_iterations = 10000
    constraints = [np.random.rand(n) for _ in range(10)]
    optimal_X = projected_gradient_descent(X_init, constraints, learning_rate, num_iterations)
    print(optimal_X)
    print(np.log(np.linalg.det(optimal_X)))


if __name__ == '__main__':
    main()
