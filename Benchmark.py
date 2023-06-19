import cvxpy as cp
import numpy as np


def score(X, A):
    scores = np.einsum('...i,ij,...j->...', A, X, A)
    return np.linalg.det(X), np.mean(scores <= 1. + 1e-8)  # industrial solvers always miss





def solve_cvx(A):
    n, d = A.shape

    # use root instead of covariance matrix
    R = cp.Variable(shape=(d, d), PSD=True)

    # objective and constraints
    obj = cp.Minimize(-cp.log_det(R))
    constraints = [cp.SOC(1., R @ A[i]) for i in range(n)]
    prob = cp.Problem(obj, constraints)

    # solve
    prob.solve(solver=cp.SCS)
    if prob.status != cp.OPTIMAL:
        raise Exception('CVXPY Error')

    # fixing the result and projection
    X = R.value.T @ R.value
    X /= np.max(np.einsum('...i,ij,...j->...', A, X, A))

    return X


def optimize(A):
    X_cvx = solve_cvx(A)
    return X_cvx, score(X_cvx, A)


if __name__ == '__main__':
    n, d = 100, 3
    np.random.seed(0)
    A = np.random.randn(n, d) * (np.arange(d) + 1.)
    print(optimize(A)[1])
