import numpy as np
import pytest
from RobotArmIRB120 import jointToPosition, jointToRotMat, jointToPosJac, jointToRotJac


N_TESTS = 1000


@pytest.mark.parametrize("func", [jointToPosJac, jointToRotJac])
def test_getTransform_form(func,n=6):
    print(f"Testing {func.__name__} shape ...", end=" ")
    q = np.random.rand(n)
    J_ = func(q)
    assert J_.shape == (3,n), f"Error: Jacobian matrix has wrong shape! ({J_.shape})"
    print("passed.")

def test_jointToPosJac_numerical(epsilon=1e-6, tol=1e-8):
    print(f"Testing J_p numerical ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        n = len(q)

        J = np.zeros((3, n))

        for i in range(n):

            q_plus = q.copy()
            q_minus = q.copy()

            q_plus[i] += epsilon
            q_minus[i] -= epsilon

            p_plus = jointToPosition(q_plus)
            p_minus = jointToPosition(q_minus)

            J[:, i] = (p_plus - p_minus) / (2 * epsilon)

        J_analytic = jointToPosJac(q)
        J_numeric = J

        diff = np.linalg.norm(J_analytic - J_numeric)
        assert diff <= tol, f"Error: numerical jacobian does not match ({diff})"

    print("passed.")

def test_jointToPosJac_velocity():
    print(f"Testing J_p velocity ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        q_dot = np.random.rand(6)

        J_p = jointToPosJac(q)
        v_predicted = J_p @ q_dot

        dt = 1e-6
        p_plus = jointToPosition(q + q_dot * dt)
        p_minus = jointToPosition(q - q_dot * dt)
        v_numeric = (p_plus - p_minus) / (2 * dt)

        assert np.allclose(v_predicted, v_numeric, atol=1e-6)

    print("passed.")

def test_jointToRotJac_numerical(dt=1e-6, tol=1e-8):
    print(f"Testing J_R numerical ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        q_dot = np.random.rand(6)        
        J_R = jointToRotJac(q)

        omega_jacobian = J_R @ q_dot

        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        R_plus = jointToRotMat(q_plus)
        R_minus = jointToRotMat(q_minus)

        R_delta = R_plus @ R_minus.T

        omega_numeric = np.array([
                R_delta[2, 1] - R_delta[1, 2],
                R_delta[0, 2] - R_delta[2, 0],
                R_delta[1, 0] - R_delta[0, 1]
            ]) / (4 * dt)

        diff = np.linalg.norm(omega_jacobian - omega_numeric)
        assert diff <= tol, f"Error: numerical rotational jacobian does not match ({diff})"

    print("passed.")

def test_jointToRotJac_velocity(dt=1e-6):
    print(f"Testing J_R velocity ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        q_dot = np.random.rand(6)
        J_R = jointToRotJac(q)
        
        q_plus = q + q_dot * dt
        q_minus = q - q_dot * dt

        R_plus = jointToRotMat(q_plus)
        R_minus = jointToRotMat(q_minus)

        R_dot_numeric = (R_plus - R_minus) / (2 * dt)

        omega = J_R @ q_dot

        def skew(v):
            x, y, z = v
            return np.array([
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0]
            ])

        R = jointToRotMat(q)
        R_dot_jacobian = skew(omega) @ R

        assert np.allclose(R_dot_jacobian, R_dot_numeric, atol=1e-6)

    print("passed.")


for func in [jointToPosJac, jointToRotJac]:
    test_getTransform_form(func)

test_jointToPosJac_numerical(epsilon=1e-6)
test_jointToPosJac_velocity()
test_jointToRotJac_numerical()
test_jointToRotJac_velocity()