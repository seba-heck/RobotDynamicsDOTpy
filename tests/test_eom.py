import os
import dill
import sympy as sp
import numpy as np
import pytest
from RobotArmIRB120 import load_EOM_func



repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
N_TESTS = 10


def test_generateEOM_symmetrie(tol=1e-12):
    print(f"Testing generateEOM symmetrie  ...", end=" ", flush=True)

    M_fun = load_EOM_func("M")

    for _ in range(N_TESTS):
        q = np.random.rand(6)

        # symmetric
        M_num = np.array(M_fun(*q), dtype=float)
    
        diff_ = np.max(np.abs(M_num - M_num.T))
        assert diff_ <= tol, f"Error: mass matrix M not symmetric. ({diff_})"

    print(f"passed.")

def test_generateEOM_posdef():
    print(f"Testing generateEOM positive definite  ...", end=" ", flush=True)

    M_fun = load_EOM_func("M")

    for _ in range(N_TESTS):
        q = np.random.rand(6)
        M_num = np.array(M_fun(*q), dtype=float)
    
        # positive definite
        eigval = np.linalg.eigvalsh(M_num)
        
        assert (eigval > -1e3).all(), f"Error: mass matrix M not positive definite. ({eigval})"

    print(f"passed.")

def test_generateEOM_kinetic():
    print(f"Testing generateEOM kinetic energy  ...", end=" ", flush=True)

    M_fun = load_EOM_func("M")
    
    for _ in range(N_TESTS):
        q = np.random.rand(6)
        q_dot = np.random.rand(6)
        M_num = np.array(M_fun(*q), dtype=float)
    
        # kinetic energy
        T = 0.5 * q_dot @ M_num @ q_dot

        assert T > 0, f"Error: mass matrix M not positive definite. ({T})"

    print(f"passed.")

test_generateEOM_symmetrie()
test_generateEOM_posdef()
test_generateEOM_kinetic()

# M,g,b,H,E_kin,E_pot = generate_eom()

# phi,dphi = generalized_coords()

# M_fun = sp.lambdify(phi, M, "numpy")

# for it in range(10):


#     assert , f"Error: Eigenvalues {eigval}"

#     # kinetic energy
#     q_dot = np.random.randn(6)


#     assert T > 0