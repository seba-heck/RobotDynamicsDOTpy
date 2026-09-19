"""
FUNCTIONS of MODEL-BASED CONTROL
Various functions (and helper) for implementations of model-based control schemes:
 - control_pd_g: joint-space control
 - control_inv_dyn: operational-space inverse dynamics control
 - control_op_space_hybrid: hybrid force and motion control

Author: Sebastian Heckers
Based on: Exercise 2b from 'Robot Dynamics, ETHZ
"""
import os,sys
import numpy as np
from RobotArmIRB120 import jointToPosition,jointToRotMat,eulAngXYZToRotMat,rotMatToRotVec,pseudoInverseMat

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def _control_pose_err(q, I_r_IE_des, eul_IE_des):
    I_r_IE = jointToPosition(q).reshape((3,1))
    C_IE = jointToRotMat(q)

    # orientation error
    C_IE_des = eulAngXYZToRotMat(eul_IE_des)
    C_err = C_IE_des @ C_IE.T
    orientation_error = rotMatToRotVec(C_err)

    # pose error
    chi_err = np.vstack([I_r_IE_des - I_r_IE, orientation_error])

    return chi_err

def control_pd_g(q_des, q, dq, g):
    """Joint-space PD controller with gravity compensation."""
    # Gains: mainly inertia dependent so they are tuned joint-wise
    kp = 10.0
    kd = 2.0
    kpMat = kp * np.diag([5000, 3000, 5, 1, 0.5, 0.01])
    kdMat = kd * np.diag([5000, 3000, 5, 1, 0.5, 0.01])

    tau = kpMat@(q_des - q).reshape((6,1)) - kdMat@dq.reshape((6,1)) + g

    return tau

def control_inv_dyn(I_r_IE_des, eul_IE_des, q, dq, M, b, g, I_Je, I_dJe):
    """Operational-space inverse dynamics controller with PD stabilizing feedback term."""

    kp = 10.0
    kd = 6.0
    kpMat = kp * np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    kdMat = kd * np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # pose error
    chi_err = _control_pose_err(q, I_r_IE_des, eul_IE_des)

    w_des = kpMat@chi_err - kdMat@(I_Je@dq).reshape((6,1))
    ddq = pseudoInverseMat(I_Je, 0.1) @ (w_des - I_dJe@dq.reshape((6,1)))

    # Inverse dynamics torque
    tau = M @ ddq + b + g

    return tau

def control_op_space_hybrid(I_r_IE_des, eul_IE_des, q, dq, I_F_Ex, M_i, b_i, g_i, J_i, dJ_i):
    """Operational-space inverse dynamics controller with a PD stabilizing feedback term and a desired end-effector force."""

    kp = 50.0
    kd = 14.0
    kpMat = kp * np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    kdMat = kd * np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    I_F_E = np.vstack([I_F_Ex, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # pose error
    chi_err = _control_pose_err(q, I_r_IE_des, eul_IE_des)

    M_inv = np.linalg.inv( M_i )
    lam = pseudoInverseMat(J_i @ M_inv @ J_i.T, 0.01)
    mu = lam @ J_i @ M_inv @ b_i - lam @ dJ_i @ dq.reshape((6,1))
    p = lam @ J_i @ M_inv @ g_i

    Sm = np.diag([0, 1, 1, 1, 1, 1])
    Sf = np.eye(6) - Sm

    w_e = kpMat@chi_err - kdMat@(J_i@dq.reshape((6,1)))

    tau = J_i.T @ (lam@Sm@w_e + Sf@I_F_E + mu + p)

    return tau