import numpy as np
import pytest
from RobotArmIRB120 import pseudoInverseMat


N_TESTS = 100


def test_pseudoInverseMat_numpy(tol=1e-1):
    print(f"Testing pseudoInverseMat numpy ...", end=" ", flush=True)
    mean_ = 0.0
    for _ in range(N_TESTS):
        m = np.random.randint(2, 10)
        n = np.random.randint(2, 10)
        lam = 1e-3*np.random.rand()

        A = np.random.rand(m,n)

        pinvA = pseudoInverseMat(A, lam)
        invA = np.linalg.pinv(A)

        diff_ = np.linalg.norm(pinvA - invA)
        mean_ += diff_

    mean_ = mean_ / N_TESTS
    assert mean_  <= tol, f"Error: pseudo-inverse and numpy different. ({mean_})"

    print(f"passed. ({mean_})")

def test_pseudoInverseMat_id(tol=1e-3):
    print(f"Testing pseudoInverseMat Id ...", end=" ", flush=True)
    mean_ = 0.0
    for _ in range(N_TESTS):
        m = np.random.randint(2, 10)
        n = np.random.randint(2, 10)
        lam = 1e-3*np.random.rand()

        A = np.random.rand(m,n)

        pinvA = pseudoInverseMat(A, lam)

        if m >= n:
            I_res = pinvA @ A
            I_sol = np.eye(n)
        else:
            I_res = A @ pinvA
            I_sol = np.eye(m)


        diff_ = np.linalg.norm(I_sol - I_res)
        mean_ += diff_

    mean_ = mean_ / N_TESTS
    assert mean_  <= tol, f"Error: pseudo-inverse does not make identity matrix. ({mean_})"

    print(f"passed. ({mean_})")


test_pseudoInverseMat_numpy()
test_pseudoInverseMat_id()