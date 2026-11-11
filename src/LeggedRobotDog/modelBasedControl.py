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
from LeggedRobotDog import get_M,get_b,get_g,get_J,get_dJ,get_T

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# Reference joint configurations and velocities
PD_G_Q_STAR_J = np.array([
    0.9, -1.5,
    0.9, -1.5,
    0.5, 0.8, 0.6
]).reshape((7,1))
PD_G_DQ_STAR_J = np.zeros((7,1))

def control_pd_g(q,dq,g,S,q_des=PD_G_Q_STAR_J,dq_des=PD_G_DQ_STAR_J):
    """Joint-space PD controller with gravity compensation on zero velocity references."""

    # Joint configurations and velocities
    q_j = q[3:10].reshape(-1,1)
    dq_j = dq[3:10]

    # PD gains
    Kp = np.concatenate([ 150.0 * np.ones((4, 1)), 20.0 * np.ones((3, 1)) ])
    Kd = np.concatenate([ 2.0 * np.ones((4, 1)), 0.2 * np.ones((3, 1)) ])

    # print(f"Kp: {Kp.shape} {Kp}")
    # print(f"Kd: {Kd.shape} {Kd}")
    # print(f"q_des: {q_des.shape} {q_des}")
    # print(f"dq_des: {dq_des.shape} {dq_des}")
    # print(f"q_j: {q_j.shape} {q_j}")
    # print(f"dq_j: {dq_j.shape} {dq_j}")

    # Joint torque command
    tau = np.multiply( Kp, q_des - q_j) + np.multiply( Kd, dq_des - dq_j)

    # Gravity compensation
    tau += S @ g

    return tau


def control_floating_base(model,data,q_des, q, dq, g):
    """Joint-space PD controller with gravity compensation."""
    
    # Evaluate robot dynamics
    M_i = np.asarray(get_M(model,data,q), dtype=float)
    b_i = np.asarray(get_b(model,data,q,dq), dtype=float).reshape(-1, 1)
    g_i = np.asarray(get_g(model,data,q), dtype=float).reshape(-1, 1)

    I_J_Ff,I_J_Hf,I_J_EE = get_J(model,data,q)
    I_Jd_Ff,I_Jd_Hf = get_dJ(model,data,q,dq)
    I_T_Ff,I_T_Hf = get_T(model,data,q,dq)
    
    # Extract x and z coordinates
    I_p_Ff = I_T_Ff[[0, 2], 3]
    I_p_Hf = I_T_Hf[[0, 2], 3]


    return tau
