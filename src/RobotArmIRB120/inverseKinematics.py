"""
INVERSE KINEMATICS
Calculates the inverse kinematics for the ABB robot arm.
A iterative algorithm is used to find the joint angle to the
desired position.

Example:
    Example of the 'inverseKinematics' function:
    $ uv run src/RobotArmIRB120/inverseKinematics.py

Author: Sebastian Heckers
Based on: 
"""
import numpy as np
from RobotArmIRB120 import jointToPosJac, jointToRotJac, jointToPosition, jointToRotMat, rotMatToRotVec, pseudoInverseMat, viz_hist


def inverseKinematics(I_r_IE_des, C_IE_des, q_0, tol):
    """
    Input: I_r_IE_des - desired end-effector position, 
            C_IE_des - desired end-effector orientation (rotation matrix)
            q_0 - initial joint angles, tol - stopping threshold
    Output: q - joint angle for desired end-effector position and orientation
            hist_ - list of q (joint angles) from each iteration
    """
    it_idx = 0
    it_max = 100  # maximum number of iterations
    lam = 0.001  # damping factor (for pseudo-inverse)
    alpha = 0.5  # update rate

    diff_r = lambda r_,q_: r_ - jointToPosition(q_).reshape((3,1))
    diff_ph = lambda C_,q_: rotMatToRotVec(C_ @ jointToRotMat(q_).T)

    q = q_0
    hist_ = [(q_0, diff_r(I_r_IE_des, q), diff_ph(C_IE_des, q))]

    # iterate until desired joint angle
    while it_idx == 0 or (np.linalg.norm(dxe) > tol and it_idx < it_max):
        # Jacobian for current q
        J_P = jointToPosJac(q)
        J_R = jointToRotJac(q)
        I_J = np.vstack([J_P, J_R])

        # get new pseudo-inverse matrix
        I_J_pinv = pseudoInverseMat(I_J, lam)

        # find error vector (difference between current and desired pose)
        dr = diff_r(I_r_IE_des, q)
        dph = diff_ph(C_IE_des, q)
        dxe = np.vstack([dr, dph])

        # update
        q = ( q.reshape((6,1)) + alpha*I_J_pinv@dxe ).flatten()

        it_idx += 1
        hist_.append((q, diff_r(I_r_IE_des, q), diff_ph(C_IE_des, q)))

    dr = diff_r(I_r_IE_des, q)
    dph = diff_ph(C_IE_des, q)

    print(f"Inverse Kinematics terminated after {it_idx} iterations.\n Position error: {np.linalg.norm(dr)}.\n Attitude error: {np.linalg.norm(dph)}.\n Final q: {q}")

    return q, hist_


if __name__ == "__main__":
    I_r_IE_des = np.array([0.3, 0.4, 0.25]).reshape((3,1))
    C_IE_des = np.eye(3)
    q_0 = np.zeros(6)

    q, hist_ = inverseKinematics(I_r_IE_des, C_IE_des, q_0, 1e-6)

    print("Iterations:")
    for i,h_ in enumerate(hist_): print(f" it {i}: q = {h_[0]}, dxe = {np.linalg.norm(np.vstack([h_[1],h_[2]]))} ")
    viz_hist([h_ for h_,dr_,dph_ in hist_], dt=0.5)