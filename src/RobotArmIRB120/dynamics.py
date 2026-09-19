"""
FUNCTIONS for TIME-STEPPING AND SIMULATING the IRB120
These are functions for implementing and running the simulation loop.
It has various time-stepping methods (euler, semi-implicit, rk4 and adaptive methods), helper functions
and dynamics functions describing the simulated dynamics.

Author: Sebastian Heckers
Based on: Exercise 2b from 'Robot Dynamics, ETHZ
"""
import os,sys
import numpy as np
from RobotArmIRB120 import control_pd_g,control_inv_dyn,control_op_space_hybrid

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def make_energies(eom):
    E_pot = lambda q_: (eom['E_pot'](*q_))[0,0]
    E_kin = lambda q_,dq_: (eom['E_kin'](*q_,*dq_))[0,0]
    H = lambda q_,dq_: (eom['H'](*q_,*dq_))[0,0]
    return H,E_kin,E_pot

def abb_eom(M, b, g, tau, phi, dphi, enable_friction:bool, friction, enable_g: bool):
    """Return joint acceleration of ABB robot by solving the forward dynamics."""
    A = M
    B = tau - b
    if enable_friction:
        B = B - np.diag(friction)@dphi.reshape((6,1))
    if enable_g:
        B = B - g
    ddphi = np.linalg.solve(A, B)
    return ddphi

def make_dynamics(f_ext, tau, eom, J_fun, dJe_fun, controller, flag_friction=False, friction=None, flag_gravity=True):

    if friction is None:
        friction = np.zeros(6)

    def dynamics(q,dq, controller=controller, f_ext=f_ext, tau=tau, eom=eom, J_fun=J_fun, dJe_fun=dJe_fun, flag_friction=flag_friction, friction=friction, flag_gravity=flag_gravity):
        # Evaluate robot dynamics
        M_i = np.asarray(eom["M"](*q), dtype=float)
        b_i = np.asarray(eom["b"](*q, *dq), dtype=float).reshape(-1, 1)
        g_i = np.asarray(eom["g"](*q), dtype=float).reshape(-1, 1)
        J_i = np.asarray(J_fun(*q), dtype=float)
        dJe_i = np.asarray(dJe_fun(*q, *dq), dtype=float)

        # evaluate controller and feedback
        if controller["type"] == "PDg":
            tau = control_pd_g(controller["q_des"], q, dq, g_i)
        elif controller["type"] == "INV":
            tau = control_inv_dyn(controller["r_des"], controller["eul_des"], q, dq, M_i, b_i, g_i, J_i, dJe_i)
        elif controller["type"] == "OPS":
            tau = control_op_space_hybrid(controller["r_des"], controller["eul_des"], q, dq, controller["I_F_Ex"], M_i, b_i, g_i, J_i, dJe_i)
            f_ext = controller["reaction_collision_model"](q, dq)
    
        assert tau.shape == (6,1), f"tau wrong shape: {tau.shape}"
    
        tau = np.asarray(tau, dtype=float)
        f_i = J_i @ f_ext + tau

        ddq = abb_eom(M_i, b_i, g_i, f_i, q, dq, flag_friction, friction, flag_gravity).reshape(-1)

        return ddq

    return dynamics

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
