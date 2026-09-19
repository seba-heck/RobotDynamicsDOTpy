import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest
from RobotArmIRB120 import jointToPosition,jointToRotMat,rotMatToQuat,quatToRotMat, quatMult, rotVecWithQuat, rotMatToRotVec


N_TESTS = 1000

get_quat = lambda r: np.hstack([r.as_quat()[-1], r.as_quat()[0:3]])


@pytest.mark.parametrize("q, r_des", [(np.ones(6)*(np.pi/6.0), np.array([0.29, 0.19, 0.23]))])
def test_jointToPosition(q, r_des):
    r_ = jointToPosition(q)
    diff_ = np.linalg.norm(r_ - r_des)
    assert diff_ <= 1e-2, "Error: wrong end-effector position!"
    print(f"Test position passed: \n q = {q} -> r = {r_} ({diff_})")

def test_quatToRotMat(tol=1e-8):
    print(f"Testing rotation transformation ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        R = jointToRotMat(q)
        quat_IE = rotMatToQuat(R)
        R_IE = quatToRotMat(quat_IE)

        assert np.linalg.norm(R - R_IE) <= tol, "Error: rotation matrices are different."

    print("passed.")

def test_rotMatToQuat(tol=1e-8):
    print(f"Testing quaternion transformation ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        r = R.from_euler('xyz', np.random.rand(3), degrees=True)
        mat_ = r.as_matrix()
        quat_res = rotMatToQuat(mat_)
        quat_sol = get_quat(r)

        diff = np.linalg.norm(quat_sol - quat_res)
        assert diff <= tol, f"Error: quaternions are different."

    print("passed.")

def test_quatMult(tol=1e-8):
    print(f"Testing quaternion multiplication ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        q_1 = R.random()
        q_2 = R.random()

        q_sol = get_quat( q_1 * q_2 )
        q_res = quatMult( get_quat(q_1), get_quat(q_2) )

        diff = np.linalg.norm(q_sol - q_res)
        assert diff <= tol, "Error: multiplication failed."

    print("passed.")

def test_rotMatToRotVec(tol=1e-8):
    print(f"Testing rotation matrix to rotation vector ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        R_ = R.random()

        q_sol = R_.as_rotvec()
        q_res = rotMatToRotVec( R_.as_matrix() ).flatten()

        diff = np.linalg.norm(q_sol - q_res)
        assert diff <= tol, f"Error: transformation rot mat to rot vec failed. ({diff})"

    print("passed.")

def test_rotVecWithQuat(tol=1e-8):
    print(f"Testing quaternion rotation ...", end=" ", flush=True)
    for _ in range(N_TESTS):
        R_ = R.random()
        vec_ = np.random.rand(3)

        r_sol = R_.apply(vec_)
        r_res = rotVecWithQuat(get_quat(R_), vec_)

        diff = np.linalg.norm(r_sol - r_res)
        assert diff <= tol, "Error: rotation failed."

    print("passed.")

test_jointToPosition(np.ones(6)*(np.pi/6.0), np.array([0.29, 0.19, 0.23]))
test_quatToRotMat()
test_rotMatToQuat()
test_quatMult()
test_rotVecWithQuat()
test_rotMatToRotVec()
