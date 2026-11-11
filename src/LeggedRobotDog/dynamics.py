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


def dynamics(model, data, q, dq, tau, F_EE, kp = 10.0, I_p_Ff0 = None, I_p_Hf0 = None):
    kd = 2*np.sqrt(kp)

    # Evaluate robot dynamics
    M_i = np.asarray(get_M(model,data,q), dtype=float)
    b_i = np.asarray(get_b(model,data,q,dq), dtype=float).reshape(-1, 1)
    g_i = np.asarray(get_g(model,data,q), dtype=float).reshape(-1, 1)

    I_J_Ff,I_J_Hf,I_J_EE = get_J(model,data,q)
    I_Jd_Ff,I_Jd_Hf = get_dJ(model,data,q,dq)
    I_T_Ff,I_T_Hf,I_T_EE = get_T(model,data,q,dq)

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

    # print(f"A: {A.shape}")
    # print(f"S: {S.shape}")
    # print(f"1: {(S.T @ tau + I_J_EE.T @ F_EE - b_i - g_i).shape}")
    # print(f"2: {(I_Jd_c @ dq - wd_drift).shape}")
    # print(f"wd_drift: {wd_drift.shape}")
    # print(f"3: {(kp * (I_p_Ff0 - I_p_Ff)).shape}")
    # print(f"4: {(I_p_Ff0 - I_p_Ff).shape}")

    b = np.concatenate([
        S.T @ tau + I_J_EE.T @ F_EE - b_i - g_i,
        (I_Jd_c @ dq - wd_drift).reshape((4,1))
    ])

    # print(f"b: {b.shape}")

    # # Check dimensions
    # print("M_d shape:", M_i.shape)
    # print("I_J_c shape:", I_J_c.shape)
    # print("A shape:", A.shape)

    # # Check mass matrix
    # print("\nMass matrix:")
    # print("Rank:", np.linalg.matrix_rank(M_i))
    # print("Condition number:", np.linalg.cond(M_i))
    # print("Eigenvalues:", np.linalg.eigvalsh(0.5 * (M_i + M_i.T)))

    # # Check constraint Jacobian
    # print("\nConstraint Jacobian:")
    # print("Rank:", np.linalg.matrix_rank(I_J_c))
    # print("Expected rank:", I_J_c.shape[0])
    # print("Singular values:", np.linalg.svd(I_J_c, compute_uv=False))

    # # Check complete system matrix
    # print("\nSystem matrix:")
    # print("Rank:", np.linalg.matrix_rank(A))
    # print("Expected rank:", A.shape[0])
    # print("Singular values:", np.linalg.svd(A, compute_uv=False))

    # Damped least-squares regularization
    damping = 1e-12
    A = A + damping * np.eye(A.shape[0])

    # Solve for acceleration and constraint forces simultaneously
    xd = np.linalg.solve(A, b)

    ddq = xd[:10].reshape((10,1))
    F_C = xd[10:14]

    return ddq, F_C


def step_euler(q,dq,dt, dynamics):
    """Explicit Euler step."""
    ddq = dynamics(q,dq)

    q_new = q + dt*dq
    dq_new = dq + dt*ddq

    return q_new,dq_new,ddq,dt

def step_semi_implicit_euler(q,dq,dt,dynamics):
    """Semi-implicit / sympletic Euler step."""
    ddq = dynamics(q,dq)

    dq_new = dq + dt*ddq
    q_new = q + dt*dq_new

    return q_new,dq_new,ddq,dt

def step_rk4(q, dq, dt, dynamics):
    """Classical fourth-order Runge-Kutta."""
    # State vector x = [q, dq]
    x = np.concatenate([q, dq])

    def f(x):
        q = x[:6]
        dq = x[6:]
        ddq = dynamics(q, dq)
        return np.concatenate([dq, ddq])

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    q_new = x_new[:6]
    dq_new = x_new[6:]

    # Acceleration corresponding to the new state
    ddq_new = dynamics(q_new, dq_new)

    return q_new, dq_new, ddq_new, dt

def step_adaptive_rk4(q, dq, dt, dynamics, tol=1e-6):
    while True:
        # 1 großer Schritt
        q1, dq1, ddq1, _ = step_rk4(q, dq, dt, dynamics)

        # 2 kleine Schritte
        qh, dqh, ddqh, _ = step_rk4(q, dq, dt/2, dynamics)
        q2, dq2, ddq2, _ = step_rk4(qh, dqh, dt/2, dynamics)

        # Fehlerabschätzung
        err = max(
            np.linalg.norm(q2 - q1, np.inf),
            np.linalg.norm(dq2 - dq1, np.inf)
        )

        if err < tol:
            dt_new = dt * min(2.0, 0.9 * (tol / max(err, 1e-16))**0.2)
            ddq = dynamics(q2, dq2).reshape(-1)
            return q2, dq2, ddq, dt_new

        # Schritt war zu groß -> verkleinern
        dt *= max(0.1, 0.9 * (tol / err)**0.2)

def step_adaptive_semi_implicit_euler(q, dq, dt, dynamics, tol=1e-4):
    while True:
        # 1 großer Schritt
        q1, dq1, ddq1, _ = step_semi_implicit_euler(q, dq, dt, dynamics)

        # 2 kleine Schritte
        qh, dqh, ddqh, _ = step_semi_implicit_euler(q, dq, dt/2, dynamics)
        q2, dq2, ddq2, _ = step_semi_implicit_euler(qh, dqh, dt/2, dynamics)

        # Fehlerabschätzung
        err = max(
            np.linalg.norm(q2 - q1, np.inf),
            np.linalg.norm(dq2 - dq1, np.inf)
        )

        if err < tol:
            dt_new = dt * min(2.0, 0.9 * (tol / max(err, 1e-16))**0.2)
            ddq = dynamics(q2, dq2).reshape(-1)
            return q2, dq2, ddq, dt_new

        # Schritt war zu groß -> verkleinern
        dt *= max(0.1, 0.9 * (tol / err)**0.2)


Integrators_Dynamics = {
    "euler": step_euler,
    "semi_implicit_euler": step_semi_implicit_euler,
    "rk4": step_rk4,
    "adaptive_rk4": step_adaptive_rk4,
    "adaptive_semi_implicit_euler": step_adaptive_semi_implicit_euler,
}
