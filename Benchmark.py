import cvxpy as cp
import numpy as np

import numpy as np


def score(X, A):
    scores = np.einsum('...i,ij,...j->...', A, np.linalg.inv(X), A)
    _, logdet = np.linalg.slogdet(X)
    return logdet, np.all(scores <= 1.)


n, d = 100, 3
np.random.seed(0)
A = np.random.randn(n, d) * (np.arange(d) + 1.)



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
    return np.linalg.inv(X) * np.max(np.einsum('...i,ij,...j->...', A, X, A))


X_cvx = solve_cvx(A)
score(X_cvx, A)


def optimize(A):
    X_cvx = solve_cvx(A)
    return X_cvx, score(X_cvx, A)


if __name__ == '__main__':
    n, d = 100, 3
    M = 4  # Number of vectors a
    n = 5  # Dimension of the vectors a
    np.random.seed(0)
    A = np.random.rand(M, n)
    # sort the vectors in A by their norm
    A = A[np.argsort([np.linalg.norm(a) for a in A])]
    # A = np.random.randn(n, d) * (np.arange(d) + 1.)
    print(optimize(A))
