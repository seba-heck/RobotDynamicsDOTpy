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
from .hopt import hopt
from .helper import get_position_EE,get_velocity_EE


repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

## REFERENCES
# Joint configurations and velocities
PD_G_Q_STAR_J = np.array([
        0.9, -1.5,
        0.9, -1.5,
        0.5, -0.8, 0.6
    ]).reshape((7,1))
PD_G_DQ_STAR_J = np.zeros((7,1))

FLOATING_BASE_Q0 = np.array([
        0.0, 0.50, 0.0,
        0.9, -1.5,
        0.9, -1.5,
        0.5, -0.8, 0.6
    ]).reshape((10,1))

# End-effector references
FLOATING_BASE_P_STAR_EE = np.array([0.5, 0.8, -1.3]).reshape((3,1))
FLOATING_BASE_W_STAR_EE = np.zeros((3,1))

# Selection matrix for seven actuated joints
SELECTION_MAT = np.hstack([
        np.zeros((7, 3)),
        np.eye(7)
    ])


def control_pd_g(q, dq, eom, *args,
                 S=SELECTION_MAT,q_des=PD_G_Q_STAR_J,dq_des=PD_G_DQ_STAR_J):
    """Joint-space PD controller with gravity compensation on zero velocity references."""
    g = eom['g_i']

    # Joint configurations and velocities
    q_j = q[3:10].reshape(-1,1)
    dq_j = dq[3:10]

    # PD gains
    Kp = np.concatenate([ 150.0 * np.ones((4, 1)), 20.0 * np.ones((3, 1)) ])
    Kd = np.concatenate([ 2.0 * np.ones((4, 1)), 0.2 * np.ones((3, 1)) ])

    # Joint torque command
    tau = np.multiply( Kp, q_des - q_j) + np.multiply( Kd, dq_des - dq_j)

    # Gravity compensation
    tau += S @ g

    return tau

def control_floating_base(q, dq, eom, base_motion_trajectory, t, 
                          S=SELECTION_MAT, p_star_EE=FLOATING_BASE_P_STAR_EE, w_star_EE=FLOATING_BASE_W_STAR_EE, q0=FLOATING_BASE_Q0, print_solution=False):
    """Joint-space PD controller with gravity compensation."""
    M_i = eom['M_i']
    b_i = eom['b_i']
    g_i = eom['g_i']
    I_J_Ba = eom['I_J_Ba']
    I_J_Ff = eom['I_J_Ff']
    I_J_Hf = eom['I_J_Hf']
    I_J_EE = eom['I_J_EE']
    I_Jd_Ba = eom['I_dJ_Ba']
    I_Jd_Ff = eom['I_dJ_Ff']
    I_Jd_Hf = eom['I_dJ_Hf']
    I_Jd_EE = eom['I_dJ_EE']
    I_T_Ff = eom['I_T_Ff']
    I_T_Hf = eom['I_T_Hf']
    I_T_EE = eom['I_T_EE']
    
    # Base position and velocity
    p_B = q[:3].copy()
    w_B = dq[:3].copy()

    # End-effector position & orientation and velocity
    p_EE = get_position_EE(I_T_EE)
    w_EE = get_velocity_EE(I_J_EE,dq)

    # Foot constraints: zero linear velocity in x and z
    J_c = np.vstack([
        I_J_Ff[:2, :],
        I_J_Hf[:2, :]
    ])

    dJ_c = np.vstack([
        I_Jd_Ff[:2, :],
        I_Jd_Hf[:2, :]
    ])

    # End-effector Jacobian: planar motion
    J_EE = I_J_EE[:3, :]
    dJ_EE = I_Jd_EE[:3, :]

    # Equations of motion
    A_eom = np.hstack([
        M_i,
        -J_c.T,
        -S.T
    ])

    b_eom = -b_i - g_i

    # No foot-contact motion
    A_c = np.hstack([
        J_c,
        np.zeros((4, 11))
    ])

    b_c = -dJ_c @ dq

    # BASE & END-EFFECTOR MOTION TASKS
    # Query base trajectory planner
    p_star_B, w_star_B = base_motion_trajectory(t)

    # Base motion
    kp_b = np.asarray(10.0).flatten()
    kd_b = 2.0 * np.sqrt(kp_b)

    A_b = np.hstack([
        I_J_Ba,
        np.zeros((3, 11))
    ])

    b_b = (
        kp_b * (p_star_B - p_B)
        + kd_b * (w_star_B - w_B)
        - I_Jd_Ba @ dq
    )

    # End-effector motion
    kp_ee = 5.0
    kd_ee = 2.0 * np.sqrt(kp_ee)

    A_ee = np.hstack([
        J_EE,
        np.zeros((3, 11))
    ])

    b_ee = (
        kp_ee * (p_star_EE - p_EE)
        + kd_ee * (w_star_EE - w_EE)
        - dJ_EE @ dq
    )

    ## ADDITIONAL TASKS
    # Kinematic null-space position control
    kp_0 = 1.0
    kd_0 = 2.0 * np.sqrt(kp_0)

    A_0 = np.hstack([
        np.eye(10),
        np.zeros((10, 11))
    ])

    b_0 = kp_0 * (q0 - q) - kd_0 * dq

    # Torque minimization
    A_tau = np.hstack([
        np.zeros((7, 14)),
        np.eye(7)
    ])

    b_tau = np.zeros(7)

    # Contact force minimization
    A_f_c = np.hstack([
        np.zeros((4, 10)),
        np.eye(4),
        np.zeros((4, 7))
    ])

    b_f_c = np.zeros(4)

    ## INEQUALITY CONSTRAINTS
    # Torque limits
    tau_max = 50.0

    C_tau_up = np.hstack([
        np.zeros((7, 14)),
        np.eye(7)
    ])

    d_tau_up = tau_max * np.ones(7)

    C_tau_low = -C_tau_up
    d_tau_low = d_tau_up.copy()

    # Friction cone
    mu = 0.5

    C_cone = np.array([
        [0.0, -1.0],
        [1.0, -mu],
        [-1.0, -mu]
    ])

    C_f_c = np.hstack([
        np.zeros((6, 10)),
        np.kron(np.eye(2), C_cone),
        np.zeros((6, 7))
    ])

    d_f_c = np.zeros(6)

    # Assemble inequality constraints
    C = np.vstack([
        C_tau_up,
        C_tau_low,
        C_f_c
    ])

    d = np.concatenate([
        d_tau_up,
        d_tau_low,
        d_f_c
    ])
    C = np.array(C, dtype=np.float64, copy=True)
    d = np.array(d, dtype=np.float64, copy=True)

    # ASSEMBLE AND SOLVE HOPT
    # Define hierarchy of tasks (highest priority first)
    A_tasks = [
        np.array(A_eom, dtype=np.float64, copy=True),
        np.array(A_c,   dtype=np.float64, copy=True),
        np.array(A_b,   dtype=np.float64, copy=True),
        np.array(A_ee,  dtype=np.float64, copy=True),
        np.array(A_0,   dtype=np.float64, copy=True),
        np.array(A_tau, dtype=np.float64, copy=True),
        np.array(A_f_c, dtype=np.float64, copy=True)
    ]

    b_tasks = [
        np.array(b_eom, dtype=np.float64, copy=True),
        np.array(b_c,   dtype=np.float64, copy=True),
        np.array(b_b,   dtype=np.float64, copy=True),
        np.array(b_ee,  dtype=np.float64, copy=True),
        np.array(b_0,   dtype=np.float64, copy=True),
        np.array(b_tau, dtype=np.float64, copy=True),
        np.array(b_f_c, dtype=np.float64, copy=True)
    ]

    # Solve hierarchical QPs
    try:
        x_opt = hopt(
            A_tasks,
            b_tasks,
            C,
            d,
            print_solution
        )

    except Exception as e:
        print("\n[ERROR] Hierarchical optimization failed!")
        print(f"Simulation time: {t:.6f} s")
        print(f"Exception: {type(e).__name__}: {e}")
        raise

    if x_opt is None:
        raise RuntimeError("Hierarchical optimization failed.")

    # Extract solution
    x_opt = np.asarray(x_opt, dtype=np.float64).copy()
    tau = x_opt[14:21].reshape((7, 1)).copy()

    # ddq = x_opt[:10]
    # f_c = x_opt[10:14]
    # tau = x_opt[14:21].reshape(-1,1)

    return tau

def control_hybrid_force(q, dq, eom, base_motion_trajectory, t, 
                          S=SELECTION_MAT, p_star_EE_z=0.6, f_star_EE=np.array([ [-0.7], [0.0], [0.0]]), q0=FLOATING_BASE_Q0, print_solution=False):
    """Joint-space PD controller with gravity compensation."""    
    M_i = eom['M_i']
    b_i = eom['b_i']
    g_i = eom['g_i']
    I_J_Ba = eom['I_J_Ba']
    I_J_Ff = eom['I_J_Ff']
    I_J_Hf = eom['I_J_Hf']
    I_J_EE = eom['I_J_EE']
    I_Jd_Ba = eom['I_dJ_Ba']
    I_Jd_Ff = eom['I_dJ_Ff']
    I_Jd_Hf = eom['I_dJ_Hf']
    I_Jd_EE = eom['I_dJ_EE']
    I_T_Ff = eom['I_T_Ff']
    I_T_Hf = eom['I_T_Hf']
    I_T_EE = eom['I_T_EE']
    
    # Base position and velocity
    p_B = q[:3].copy()
    w_B = dq[:3].copy()

    # End-effector position & orientation and velocity
    p_EE = get_position_EE(I_T_EE)
    w_EE = get_velocity_EE(I_J_EE,dq)

    # ============================================================
    # 5. Contact Jacobian
    # ============================================================
    # Foot constraints: zero linear velocity in x and z
    J_c = np.vstack([
        I_J_Ff[:2, :],
        I_J_Hf[:2, :]
    ])

    dJ_c = np.vstack([
        I_Jd_Ff[:2, :],
        I_Jd_Hf[:2, :]
    ])

    # ============================================================
    # 6. Control references
    # ============================================================
    # Base trajectory
    p_star_B, w_star_B = base_motion_trajectory(t)

    p_star_B = np.asarray(p_star_B).reshape(3, 1)
    w_star_B = np.asarray(w_star_B).reshape(3, 1)

    # ============================================================
    # 7. Optimization variables
    # ============================================================
    # x_opt = [ddq (10), f_c (4), tau (7), f_EE (3)]
    # n = 24

    # Z_10 = np.zeros((10, 14))
    # Z_7 = np.zeros((7, 17))
    # Z_3 = np.zeros((3, 21))

    # ============================================================
    # 8. Optimization tasks
    # ============================================================
    # ------------------------------------------------------------
    # Task 0: Equations of motion
    # ------------------------------------------------------------
    A_eom = np.hstack([
        M_i,
        -J_c.T,
        -S.T,
        -I_J_EE.T
    ])

    b_eom = -b_i - g_i

    # ------------------------------------------------------------
    # Task 1: No foot-contact motion
    # ------------------------------------------------------------
    A_c = np.hstack([
        J_c,
        np.zeros((4, 14))
    ])

    b_c = -dJ_c @ dq

    # ------------------------------------------------------------
    # Task 2: End-effector force
    # ------------------------------------------------------------
    A_force_ee = np.hstack([
        np.zeros((3, 21)),
        np.eye(3)
    ])

    b_force_ee = f_star_EE

    # ------------------------------------------------------------
    # Task 3: Constant EE position in z direction
    # ------------------------------------------------------------
    kp_ee = 5.0
    kd_ee = 2.0 * np.sqrt(kp_ee)

    A_pos_ee = np.hstack([
        I_J_EE[1:2, :],
        np.zeros((1, 14))
    ])

    b_pos_ee = (
        kp_ee * (p_star_EE_z - p_EE[1, 0])
        - kd_ee * w_EE[1, 0]
        - (I_Jd_EE[1:2, :] @ dq)[0, 0]
    )

    b_pos_ee = np.array([[b_pos_ee]])

    # ------------------------------------------------------------
    # Task 4: Base motion
    # ------------------------------------------------------------
    kp_b = 2.0
    kd_b = 2.0 * np.sqrt(kp_b)

    A_b = np.hstack([
        I_J_Ba,
        np.zeros((3, 14))
    ])

    b_b = (
        kp_b * (p_star_B - p_B)
        + kd_b * (w_star_B - w_B)
        - I_Jd_Ba @ dq
    )

    # ------------------------------------------------------------
    # Task 5: Kinematic null-space position control
    # ------------------------------------------------------------
    # q0 = np.array([
    #     0.0, 0.50, 0.0,
    #     0.9, -1.5,
    #     0.9, -1.5,
    #     0.9, 0.7, 0.4
    # ]).reshape(10, 1)

    kp_0 = 1.0
    kd_0 = 2.0 * np.sqrt(kp_0)

    A_0 = np.hstack([
        np.eye(10),
        np.zeros((10, 14))
    ])

    b_0 = kp_0 * (q0 - q) - kd_0 * dq

    # ------------------------------------------------------------
    # Task 6: Torque minimization
    # ------------------------------------------------------------
    A_tau = np.hstack([
        np.zeros((7, 14)),
        np.eye(7),
        np.zeros((7, 3))
    ])

    b_tau = np.zeros((7, 1))

    # ------------------------------------------------------------
    # Task 7: Contact force minimization
    # ------------------------------------------------------------
    A_f_c = np.hstack([
        np.zeros((4, 10)),
        np.eye(4),
        np.zeros((4, 10))
    ])

    b_f_c = np.zeros((4, 1))

    # ============================================================
    # 9. Inequality constraints
    # ============================================================
    # ------------------------------------------------------------
    # Torque limits
    # ------------------------------------------------------------
    tau_max = 50.0

    C_tau_up = np.hstack([
        np.zeros((7, 14)),
        np.eye(7),
        np.zeros((7, 3))
    ])

    d_tau_up = tau_max * np.ones((7, 1))

    C_tau_low = -C_tau_up
    d_tau_low = d_tau_up.copy()

    # ------------------------------------------------------------
    # Friction cone
    # ------------------------------------------------------------
    mu = 0.5

    C_cone = np.array([
        [0.0, -1.0],
        [1.0, -mu],
        [-1.0, -mu]
    ])

    C_f_c = np.hstack([
        np.zeros((6, 10)),
        np.kron(np.eye(2), C_cone),
        np.zeros((6, 10))
    ])

    d_f_c = np.zeros((6, 1))

    # ============================================================
    # 10. Assemble hierarchy
    # ============================================================
    A = [
        A_eom,
        A_c,
        A_force_ee,
        A_pos_ee,
        A_b,
        A_0,
        A_tau,
        A_f_c
    ]

    b_tasks = [
        b_eom,
        b_c,
        b_force_ee,
        b_pos_ee,
        b_b,
        b_0,
        b_tau,
        b_f_c
    ]

    # Inequality constraints
    C = np.vstack([
        C_tau_up,
        C_tau_low,
        C_f_c
    ])

    d = np.vstack([
        d_tau_up,
        d_tau_low,
        d_f_c
    ])

    # ============================================================
    # 11. Solve hierarchical QPs
    # ============================================================
    x_opt = hopt(
        A,
        b_tasks,
        C,
        d,
        print_solution
    )

    x_opt = np.asarray(x_opt).reshape(24, 1)

    # ============================================================
    # 12. Extract torques
    # ============================================================
    ddq = x_opt[:10]
    f_c = x_opt[10:14]
    tau = x_opt[14:21]
    F_EE = x_opt[21:24]

    return tau


ALL_CONTROL_FUNCS = {
    "PDg": control_pd_g,
    "FlB": control_floating_base,
    "HyF": control_hybrid_force,
}
