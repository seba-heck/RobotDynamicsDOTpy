"""
DYNAMICS of ROBOT ARM IRB120
This script implements the Equations of Motion (EOM) of the robot arm IRB120.
It has properties of the robot arm (IRB120) and computes the mass matrix M,
the coriolis & centrifugal terms b, and gravity terms g.
Helper functions to save (symbolic) EOM terms and calculate kinematic expression.

Example:
    Testing & saving the creation of EOM terms:
    $ uv run src/RobotArmIRB120/generateEOM.py

Notes:
    It uses sympy for symbolic expression using 'generalized_coordinates'.
    When this script is executed, the symbolic terms are dumped to binary
    files with dill (https://stackoverflow.com/questions/29079923/save-load-sympy-lambdifed-expressions).

Author: Sebastian Heckers
Based on: Exercise 2a from 'Robot Dynamics', ETHZ
"""
import os,sys
import dill
import numpy as np
import sympy as sp
from RobotArmIRB120 import spy

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

path_func_ = lambda f: repo_root+f"/bin/func_{f}.bin"


class IRB120:
    def __init__(self):
        self.mass = sp.Matrix( [3.0, 3.9, 2.9, 1.3, 0.55, 0.014] )
        self.com = sp.Matrix( [
            [0, 0, 0.062],
            [0, 0, 0.12],
            [0.5, 0, 0.03],
            [0, 0, 0.07],
            [0, 0, 0.03],
            [0, 0, 0]
        ] )
        self.inertia = [
            sp.diag(0.014, 0.010, 0.014),
            sp.diag(0.060, 0.026, 0.042),
            sp.diag(0.008, 0.013, 0.017),
            sp.diag(0.003, 0.005, 0.004),
            sp.diag(0.0004, 0.0008, 0.0009),
            sp.diag(0.002, 0.002, 0.003)
        ]

def dAdt(A, q, dq):
    """Calculate Jacobian matrix."""
    dA = sp.zeros(*A.shape)
    for i in range(A.rows):
        for j in range(A.cols):
            dA[i,j] = sum(
                sp.diff(A[i, j], q[k]) * dq[k]
                for k in range(len(q))
            )
    
    return dA

def generalized_coords():
    """Generates generalized coordinate symbolic variables (phi,dphi)."""
    phi = sp.Matrix(sp.symbols("phi1:7", real=True))
    dphi = sp.Matrix(sp.symbols("dphi1:7", real=True))
    return phi,dphi

def generate_kinematics_sp(phi):
    """
    Generate symbolic kinematic expressions.

    Input: phi - Generalized coordinates (sympy.Matrix)
    Output: kin - symbolic kinematic quantities (dict)
    """
    # Containers
    T_jk = [None] * 6
    T_Ik = [None] * 6
    R_Ik = [None] * 6
    k_n_k = [None] * 6

    # Homogeneous transformations
    T_jk[0] = spy.jointToTransform01(phi)
    T_jk[1] = spy.jointToTransform12(phi)
    T_jk[2] = spy.jointToTransform23(phi)
    T_jk[3] = spy.jointToTransform34(phi)
    T_jk[4] = spy.jointToTransform45(phi)
    T_jk[5] = spy.jointToTransform56(phi)

    # From frame k to inertial frame I
    T_Ik[0] = T_jk[0]

    for k in range(1, 6):
        T_Ik[k] = sp.simplify(T_Ik[k - 1] @ T_jk[k])

    # Rotation matrices
    for k in range(6):
        R_Ik[k] = T_Ik[k][:3, :3]

    # Joint rotation axes expressed in frame k
    k_n_k[0] = sp.Matrix([0, 0, 1])
    k_n_k[1] = sp.Matrix([0, 1, 0])
    k_n_k[2] = sp.Matrix([0, 1, 0])
    k_n_k[3] = sp.Matrix([1, 0, 0])
    k_n_k[4] = sp.Matrix([0, 1, 0])
    k_n_k[5] = sp.Matrix([1, 0, 0])

    # End effector
    T_Ie = sp.simplify(T_Ik[5])
    I_r_Ie = T_Ie[:3, 3]

    kin = {
        "T_jk": T_jk,
        "T_Ik": T_Ik,
        "R_Ik": R_Ik,
        "k_n_k": k_n_k,
        "T_Ie": T_Ie,
        "I_r_Ie": I_r_Ie,
    }
    return kin

def generate_jacobian_sp(phi, dphi, kin, bot):
    """
    Generate symbolic Jacobians for the IRB120.

    Input: phi & dphi - generalized coords, kin - symbolic kinematics, bot - robot parameters
    Output: jac - dictionary containing jacobians:
            I_Jp_s  - translational Jacobians of COMs
            I_Jr    - rotational Jacobians of links
            I_Jpe   - end-effector translational Jacobian
            I_Jre   - end-effector rotational Jacobian
            I_dJpe  - time derivative of translational Jacobian
            I_dJre  - time derivative of rotational Jacobian
    """
    # Setup
    T_Ik = kin["T_Ik"]
    R_Ik = kin["R_Ik"]
    k_n_k = kin["k_n_k"]
    I_r_Ie = kin["I_r_Ie"]

    k_r_ks = bot.com

    # Containers
    I_Jp_s = [None] * 6
    I_Jr = [None] * 6

    # Compute link Jacobians
    for k in range(6):
        # Position of center of mass
        r_ks_h = sp.Matrix.vstack(
            k_r_ks[k,:].reshape(3,1),
            sp.Matrix([1])
        )
        I_r_ks = ( T_Ik[k] @ r_ks_h )[:3, :]

        # Translational Jacobian
        Jp = I_r_ks.jacobian(phi)

        # Rotational Jacobian
        Jr = sp.zeros(3, 6)

        # if k > 0:
        #     # Copy previous columns
        #     Jr[:, :k] = I_Jr[k - 1][:, :k]
        for j in range(k + 1):
            n = R_Ik[j] * k_n_k[j]
            Jr[:, j] = n

        # New column
        n = R_Ik[k] * sp.Matrix(k_n_k[k])

        Jr[:, k] = n

        # Simplify
        Jp = sp.simplify(Jp)
        Jr = sp.simplify(Jr)

        I_Jp_s[k] = Jp
        I_Jr[k]  = Jr

    # End-effector Jacobians
    I_Jpe = sp.simplify( sp.Matrix(I_r_Ie).jacobian(phi) )
    I_Jre = I_Jr[5]

    # Time derivatives
    I_dJpe = dAdt(I_Jpe, phi, dphi)
    I_dJre = dAdt(I_Jre, phi, dphi)

    I_dJpe = sp.simplify(I_dJpe)
    I_dJre = sp.simplify(I_dJre)

    # Full Jacobian
    I_Je = sp.Matrix.vstack(
        I_Jpe,
        I_Jre
    )

    I_dJe = sp.Matrix.vstack(
        I_dJpe,
        I_dJre
    )

    jac = {
        "I_Jp_s": I_Jp_s,
        "I_Jr": I_Jr,
        "I_Jpe": I_Jpe,
        "I_Jre": I_Jre,
        "I_dJpe": I_dJpe,
        "I_dJre": I_dJre,
        "I_Je": I_Je,
        "I_dJe": I_dJe,
    }
    return jac

def generate_eom():
    """
    Returns matrices and vector necessary to compute the equation of motion (EOM).
    """
    bot = IRB120()
    phi,dphi = generalized_coords()
    kin = generate_kinematics_sp(phi)
    jac = generate_jacobian_sp(phi,dphi,kin,bot)

    N = 6
    m = bot.mass
    k_r_ks = bot.com
    k_I_s = bot.inertia
    I_Jp_s = jac["I_Jp_s"]
    I_Jr = jac["I_Jr"]
    R_Ik = kin["R_Ik"]
    T_Ik = kin["T_Ik"]
    I_g_acc = sp.Matrix([0, 0, -9.81])

    ## Compute MASS MATRIX
    print("Computing mass matrix M ...", end="", flush=True)
    M = sp.zeros(N,N)
    for i in range(N):
        M = M + I_Jp_s[i].T * m[i] * I_Jp_s[i] + I_Jr[i].T * R_Ik[i] * k_I_s[i] * R_Ik[i].T * I_Jr[i]

    # print(" simplify ...", end="", flush=True)
    # M = sp.simplify(M)
    print(" done.", flush=True)

    ## Compute GRAVITY TERMS
    print("Computing gravity vector g ...", end="", flush=True)
    g = sp.zeros(N,1)
    for i in range(N):
        g = g - I_Jp_s[i].T * m[i] * I_g_acc

    print(" simplify ...", end="", flush=True)
    g = sp.simplify(g)
    print(" done.", flush=True)

    ## Compute NONLINEAR TERMS VECTOR
    print("Computing coriolis and centrifugal vector b ...", end="", flush=True)
    b = sp.zeros(N,1)
    for i in range(N):
        w = I_Jr[i] * dphi
        d_I_Jp_s_i = dAdt(I_Jp_s[i], phi, dphi)
        d_I_Jr_i = dAdt(I_Jr[i], phi, dphi)
        k_I_b = R_Ik[i] * k_I_s[i] * R_Ik[i].T

        b = b + I_Jp_s[i].T * m[i] * d_I_Jp_s_i * dphi \
              + I_Jr[i].T * k_I_b * d_I_Jr_i * dphi \
              + I_Jr[i].T * w.cross(k_I_b * w)
        
    # print(" simplify ...", end="", flush=True)
    # g = sp.simplify(g)
    print(" done.", flush=True)
    
    ## Compute ENERGY
    print("Computing total energy E ...", end="", flush=True)
    E_pot = sp.zeros(1)
    for i in range(N):
        r_ks_h = sp.Matrix.vstack(
            k_r_ks[i,:].reshape(3,1),
            sp.Matrix([1])
        )
        I_r_ks = ( T_Ik[i] @ r_ks_h )[:3, :]

        E_pot = E_pot - m[i]*I_g_acc.T * I_r_ks
    
    print(" simplify ...", end="", flush=True)
    E_pot = sp.simplify(E_pot)
    E_kin = 0.5*dphi.T * M * dphi
    # E_kin = sp.simplify(0.5*dphi.T * M * dphi)
    H = E_pot + E_kin
    print(" done.", flush=True)

    eom = {
        "M": M,
        "b": b,
        "g": g,
        "E_kin": E_kin,
        "E_pot": E_pot,
        "H": H
    }
    return eom

def load_EOM_func(func_:str):
    """Load function for EOM from binary file."""
    assert func_ in ("M", "b", "g", "E_kin", "E_pot", "H")
    path_ = path_func_(func_)
    phi,dphi = generalized_coords()
    sym_ = dill.load(open(path_, "rb"))
    if func_ in ("M", "g", "E_pot"):
        f_ = sp.lambdify(phi, sym_, "numpy")
    elif func_ in ("b", "E_kin", "H"):
        f_ = sp.lambdify((*phi,*dphi), sym_, "numpy")
    return f_

def load_EOM():
    """Load all functions of EOM from binary files."""
    eom = {}
    for func_ in ("M", "b", "g", "E_kin", "E_pot", "H"):
        f_fun = load_EOM_func(func_)
        eom[func_] = f_fun
    return eom


if __name__ == "__main__":
    eom = generate_eom()

    dill.settings['recurse'] = True
    for func_ in eom.keys():
        path_ = path_func_(func_)
        dill.dump(eom[func_], open(path_, "wb"))
        print(f"Saved: {func_}")

    M = dill.load(open(repo_root+"/bin/M_func.bin", "rb"))

    phi,dphi = generalized_coords()

    M_fun = sp.lambdify(phi, M, "numpy")

    for it in range(10):
        q = np.random.rand(6)

        # symmetric
        M_num = np.array(M_fun(*q), dtype=float)

        assert np.max(np.abs(M_num - M_num.T)) < 1e-12

        # positive definite
        eigval = np.linalg.eigvalsh(M_num)

        assert (eigval > -1e3).all(), f"Error: Eigenvalues {eigval}"

        # kinetic energy
        q_dot = np.random.randn(6)

        T = 0.5 * q_dot @ M_num @ q_dot

        assert T > 0
