"""
FUNCTIONS for TIME-STEPPING AND SIMULATING the IRB120
These are functions for implementing and running the simulation loop.
It has various time-stepping methods (euler, semi-implicit, rk4 and adaptive methods), helper functions
and dynamics functions describing the simulated dynamics.

Author: Sebastian Heckers
Based on: Exercise 3 from 'Robot Dynamics, ETHZ
"""
import os,sys
import numpy as np
import pinocchio as pin

from .generateEOM import get_M,get_b,get_g,get_J,get_dJ,get_T


# def dynamics(model, data, q, dq, tau, F_EE, kp = 10.0, I_p_Ff0 = None, I_p_Hf0 = None):
def dynamics(eom, q, dq, tau, F_EE, kp = 10.0, I_p_Ff0 = None, I_p_Hf0 = None):
    kd = 2*np.sqrt(kp)

    M_i = eom['M_i']
    b_i = eom['b_i']
    g_i = eom['g_i']
    I_J_Ff = eom['I_J_Ff']
    I_J_Hf = eom['I_J_Hf']
    I_J_EE = eom['I_J_EE']
    I_Jd_Ff = eom['I_dJ_Ff']
    I_Jd_Hf = eom['I_dJ_Hf']
    I_T_Ff = eom['I_T_Ff']
    I_T_Hf = eom['I_T_Hf']
    I_T_EE = eom['I_T_EE']

    # # Evaluate robot dynamics
    # M_i = np.asarray(get_M(model,data,q), dtype=float)
    # b_i = np.asarray(get_b(model,data,q,dq), dtype=float).reshape(-1, 1)
    # g_i = np.asarray(get_g(model,data,q), dtype=float).reshape(-1, 1)

    # _,I_J_Ff,I_J_Hf,I_J_EE = get_J(model,data,q)
    # _,I_Jd_Ff,I_Jd_Hf,_ = get_dJ(model,data,q,dq)
    # _,I_T_Ff,I_T_Hf,I_T_EE = get_T(model,data,q,dq)

    # Extract x and z coordinates
    I_p_Ff = I_T_Ff[[0, 2], 3]
    I_p_Hf = I_T_Hf[[0, 2], 3]

    # Initialize reference foot positions
    if I_p_Ff0 is None:
        I_p_Ff0 = I_p_Ff.copy()
        I_p_Hf0 = I_p_Hf.copy()

    # Assemble foot constraints -> no linear velocity
    I_J_c = np.vstack([
        I_J_Ff[0:2, :],
        I_J_Hf[0:2, :]
    ])
    I_Jd_c = np.vstack([
        I_Jd_Ff[0:2, :],
        I_Jd_Hf[0:2, :]
    ])
    num_constr = I_J_c.shape[0]

    # Selection matrix
    S = np.hstack([
        np.zeros((7, 3)),
        np.eye(7)
    ])

    # Drift correction
    wd_drift = np.concatenate([
        kp * (I_p_Ff0 - I_p_Ff),
        kp * (I_p_Hf0 - I_p_Hf)
    ]).reshape((4,1)) - kd * I_J_c @ dq

    # Define matrix and vector of the linear system
    A = np.block([
        [M_i, -I_J_c.T],
        [-I_J_c, np.zeros((num_constr, num_constr))]
    ])

    b = np.concatenate([
        S.T @ tau + I_J_EE.T @ F_EE - b_i - g_i,
        (I_Jd_c @ dq - wd_drift).reshape((4,1))
    ])

    # Damped least-squares regularization
    damping = 1e-12
    A = A + damping * np.eye(A.shape[0])

    # Solve for acceleration and constraint forces simultaneously
    xd = np.linalg.solve(A, b)

    ddq = xd[:10].reshape((10,1))
    F_C = xd[10:14]

    return ddq, F_C
